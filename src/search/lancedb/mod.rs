use anyhow::{bail, Result};
use async_trait::async_trait;
use futures::TryStreamExt;

use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{ArrowError, SchemaRef};
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};

use crate::search::interface::{Doc, Hit, Meta, VectorDbRead, VectorDbSetup, VectorDbWrite};
use std::sync::Arc;
use arrow_array::{types::Float32Type, ArrayRef};
use arrow_schema::{DataType, Field, Schema};
    
/// LanceDB 実装
pub struct LanceVectorDb {
	pub db_uri: String,
	pub table_name: String,
	pub embedding_dim: i32,
}

impl LanceVectorDb {
	pub fn new(db_uri: impl Into<String>, table_name: impl Into<String>, embedding_dim: i32) -> Self {
		Self {
			db_uri: db_uri.into(),
			table_name: table_name.into(),
			embedding_dim,
		}
	}
}

#[async_trait]
impl VectorDbWrite for LanceVectorDb {
	async fn upsert(&self, docs: &[Doc]) -> Result<()> {
		if docs.is_empty() {
			return Ok(());
		}
		let dim = self.embedding_dim;
		for d in docs {
			if d.vector.len() as i32 != dim {
				bail!("vector dim {} != expected {}", d.vector.len(), dim);
			}
		}

		let db = connect(&self.db_uri).execute().await?;
		let schema = build_docs_schema(dim);

		let ids: Vec<i64> = docs.iter().map(|d| d.id).collect();
		let texts_refs: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();
		let mut vectors_flat: Vec<f32> = Vec::with_capacity(docs.len() * dim as usize);
		for d in docs {
			vectors_flat.extend_from_slice(&d.vector);
		}
		let batch = build_docs_batch(schema.clone(), &ids, &texts_refs, &vectors_flat, dim)?;

		let table = match db.open_table(&self.table_name).execute().await {
			Ok(t) => t,
			Err(_) => {
				let reader = arrow_array::RecordBatchIterator::new(
					vec![batch.clone()]
						.into_iter()
						.map(|rb| Ok::<RecordBatch, ArrowError>(rb)),
					schema.clone(),
				);
				db.create_table(&self.table_name, reader).execute().await?
			}
		};

		let reader = arrow_array::RecordBatchIterator::new(
			vec![batch]
				.into_iter()
				.map(|rb| Ok::<RecordBatch, ArrowError>(rb)),
			schema,
		);
		table.add(reader).execute().await?;
		Ok(())
	}
}

#[async_trait]
impl VectorDbRead for LanceVectorDb {
	async fn nearest(&self, query: &[f32], top_k: usize) -> Result<Vec<Hit>> {
		if query.len() as i32 != self.embedding_dim {
			bail!("query dim {} != expected {}", query.len(), self.embedding_dim);
		}
		let db = connect(&self.db_uri).execute().await?;
		let table = db.open_table(&self.table_name).execute().await?;

		let stream = table
			.query()
			.nearest_to(query.to_vec())?
			.limit(top_k)
			.execute()
			.await?;
		let batches: Vec<RecordBatch> = stream.try_collect().await?;

		let mut hits: Vec<Hit> = Vec::new();
		for batch in &batches {
			let schema: SchemaRef = batch.schema();
			let id_idx = schema.index_of("id")?;
			let text_idx = schema.index_of("text")?;
			let vec_idx = schema.index_of("vector")?;

			let ids = batch.column(id_idx).as_any().downcast_ref::<Int64Array>().unwrap();
			let texts = batch.column(text_idx).as_any().downcast_ref::<StringArray>().unwrap();
			let vectors = batch.column(vec_idx).as_any().downcast_ref::<FixedSizeListArray>().unwrap();
			let values = vectors.values().as_any().downcast_ref::<Float32Array>().unwrap();
			let dim = self.embedding_dim as usize;

			for i in 0..batch.num_rows() {
				let id = ids.value(i);
				let text = texts.value(i).to_string();
				let base = i * dim;
				let mut v: Vec<f32> = Vec::with_capacity(dim);
				for k in 0..dim {
					v.push(values.value(base + k));
				}
				let score = cosine(query, &v);
				hits.push(Hit { id, text, score });
			}
		}

		// 既に top_k で取得しているが、cosine 再計算したので念のため整列
		hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
		hits.truncate(top_k);
		Ok(hits)
	}

	fn dim(&self) -> usize { self.embedding_dim as usize }
}

#[async_trait]
impl VectorDbSetup for LanceVectorDb {
	async fn ensure_created(&self) -> Result<()> {
		let db = connect(&self.db_uri).execute().await?;
		match db.open_table(&self.table_name).execute().await {
			Ok(_) => Ok(()),
			Err(_) => {
				let schema = build_docs_schema(self.embedding_dim);
				let empty = Vec::<RecordBatch>::new();
				let reader = arrow_array::RecordBatchIterator::new(
					empty
						.into_iter()
						.map(|rb| Ok::<RecordBatch, ArrowError>(rb)),
					schema.clone(),
				);
				db.create_table(&self.table_name, reader).execute().await?;
				Ok(())
			}
		}
	}

	async fn write_meta(&self, meta: Meta) -> Result<()> {
		ensure_meta_row(&self.db_uri, meta).await
	}

	async fn latest_meta(&self) -> Result<Option<Meta>> {
		read_latest_meta(&self.db_uri).await
	}
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
	let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
	let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
	let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
	if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

// ---- 以下、元 vec.rs の LanceDB ヘルパ群 ----
/// docs テーブル用のスキーマを生成します。
/// - カラム: id(Int64), text(Utf8), vector(FixedSizeList(Float32, dim))
pub fn build_docs_schema(dim: i32) -> Arc<Schema> {
	Arc::new(Schema::new(vec![
		Field::new("id", DataType::Int64, false),
		Field::new("text", DataType::Utf8, false),
		Field::new(
			"vector",
			DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
			false,
		),
	]))
}

/// docs テーブル向けの `RecordBatch` を生成します。
/// - `vectors_flat` はフラット化済み (行数×dim) の長さである必要があります。
pub fn build_docs_batch(
	schema: Arc<Schema>,
	ids: &[i64],
	texts: &[&str],
	vectors_flat: &[f32],
	dim: i32,
) -> anyhow::Result<RecordBatch> {
	use anyhow::bail;
	let num_rows = ids.len();
	if texts.len() != num_rows {
		bail!("texts length {} != ids length {}", texts.len(), num_rows);
	}
	if vectors_flat.len() != num_rows * (dim as usize) {
		bail!(
			"vectors_flat length {} != num_rows({}) * dim({})",
			vectors_flat.len(),
			num_rows,
			dim
		);
	}

	let ids_arr = Int64Array::from(ids.to_vec());
	let texts_arr = StringArray::from(texts.to_vec());
	let list_iter = vectors_flat
		.chunks(dim as usize)
		.map(|chunk| Some(chunk.iter().copied().map(Some).collect::<Vec<Option<f32>>>()));
	let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(list_iter, dim);

	let batch = RecordBatch::try_new(
		schema,
		vec![
			Arc::new(ids_arr) as ArrayRef,
			Arc::new(texts_arr) as ArrayRef,
			Arc::new(vectors) as ArrayRef,
		],
	)?;
	Ok(batch)
}

/// `_meta` テーブルのスキーマ
pub fn build_meta_schema() -> Arc<Schema> {
	Arc::new(Schema::new(vec![
		Field::new("version", DataType::Utf8, false),
		Field::new("source_dir", DataType::Utf8, false),
		Field::new("model_name", DataType::Utf8, false),
		Field::new("embedding_dim", DataType::Int32, false),
		Field::new("created_at_ms", DataType::Int64, false),
	]))
}

/// `_meta` 用の 1 行 `RecordBatch` を作成
pub fn build_meta_batch(schema: Arc<Schema>, m: &Meta) -> RecordBatch {
	let version = StringArray::from(vec![m.version.clone()]);
	let source_dir = StringArray::from(vec![m.source_dir.clone()]);
	let model_name = StringArray::from(vec![m.model_name.clone()]);
	let embedding_dim = Int32Array::from(vec![m.embedding_dim]);
	let created_at_ms = Int64Array::from(vec![m.created_at_ms]);

	RecordBatch::try_new(
		schema,
		vec![
			Arc::new(version) as ArrayRef,
			Arc::new(source_dir) as ArrayRef,
			Arc::new(model_name) as ArrayRef,
			Arc::new(embedding_dim) as ArrayRef,
			Arc::new(created_at_ms) as ArrayRef,
		],
	)
	.expect("failed to build meta record batch")
}

/// `_meta` テーブルに 1 行追記（無ければ作成）
pub async fn ensure_meta_row(db_uri: &str, meta: Meta) -> anyhow::Result<()> {
	use arrow_array::RecordBatchIterator;
	let db = connect(db_uri).execute().await?;
	let schema = build_meta_schema();

	let table = match db.open_table("_meta").execute().await {
		Ok(t) => t,
		Err(_) => {
			let batch = build_meta_batch(schema.clone(), &meta);
			let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
			db.create_table("_meta", reader).execute().await?
		}
	};

	let batch = build_meta_batch(schema.clone(), &meta);
	let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
	table.add(reader).execute().await?;
	Ok(())
}

/// `_meta` テーブルから最新の 1 行を読み出し
pub async fn read_latest_meta(db_uri: &str) -> anyhow::Result<Option<Meta>> {
	let db = connect(db_uri).execute().await?;
	let table = match db.open_table("_meta").execute().await {
		Ok(t) => t,
		Err(_) => return Ok(None),
	};

	let stream = table.query().execute().await?;
	let batches: Vec<RecordBatch> = stream.try_collect().await?;

	let mut best: Option<Meta> = None;
	for batch in &batches {
		let v = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
		let s = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
		let m = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
		let d = batch.column(3).as_any().downcast_ref::<Int32Array>().unwrap();
		let t = batch.column(4).as_any().downcast_ref::<Int64Array>().unwrap();

		for i in 0..batch.num_rows() {
			let cand = Meta {
				version: v.value(i).to_string(),
				source_dir: s.value(i).to_string(),
				model_name: m.value(i).to_string(),
				embedding_dim: d.value(i),
				created_at_ms: t.value(i),
			};
			if best
				.as_ref()
				.map(|b| cand.created_at_ms > b.created_at_ms)
				.unwrap_or(true)
			{
				best = Some(cand);
			}
		}
	}
	Ok(best)
}

#[cfg(test)]
mod lancedb_tests {
	use super::*;
	use tempfile::tempdir;
	use anyhow::Result;
	use std::time::{SystemTime, UNIX_EPOCH};
	use crate::search::interface::{Doc, VectorDbRead, VectorDbSetup, VectorDbWrite};
	use crate::search::interface::Meta;

	#[tokio::test]
	async fn test_setup_and_meta() -> Result<()> {
		let tmp = tempdir().expect("tmpdir");
		let uri = tmp.path().to_str().unwrap();

		let vdb = LanceVectorDb::new(uri, "docs", 2);
		vdb.ensure_created().await?;

		let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;
		let meta = Meta {
			version: "v0.0.1".into(),
			source_dir: "/tmp".into(),
			model_name: "test".into(),
			embedding_dim: 2,
			created_at_ms: now_ms,
		};
		vdb.write_meta(meta.clone()).await?;
		let latest = vdb.latest_meta().await?.expect("meta exists");
		assert_eq!(latest.version, meta.version);
		assert_eq!(latest.embedding_dim, 2);
		Ok(())
	}

	#[tokio::test]
	async fn test_upsert_and_nearest() -> Result<()> {
		let tmp = tempdir().expect("tmpdir");
		let uri = tmp.path().to_str().unwrap();

		let vdb = LanceVectorDb::new(uri, "docs", 2);
		vdb.ensure_created().await?;

		let docs = vec![
			Doc { id: 1, text: "りんご".into(),   vector: vec![1.0, 0.0] },
			Doc { id: 2, text: "バナナ".into(),   vector: vec![0.9, 0.1] },
			Doc { id: 3, text: "さくらんぼ".into(), vector: vec![0.0, 1.0] },
		];
		vdb.upsert(&docs).await?;

		let query = vec![0.95_f32, 0.05_f32];
		let hits = vdb.nearest(&query, 2).await?;
		assert!(!hits.is_empty());
		assert_eq!(hits[0].id, 1);
		Ok(())
	}
}



