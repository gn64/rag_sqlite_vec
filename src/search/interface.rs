use anyhow::Result;
use async_trait::async_trait;

/// RAG 向け Vector DB の最小データ型
#[derive(Debug, Clone)]
pub struct Doc {
	pub id: i64,
	pub text: String,
	pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Hit {
	pub id: i64,
	pub text: String,
	pub score: f32, // 大きいほど類似（cosine）
}

/// DB 全体のメタデータ
#[derive(Debug, Clone)]
pub struct Meta {
	pub version: String,
	pub source_dir: String,
	pub model_name: String,
	pub embedding_dim: i32,
	pub created_at_ms: i64,
}

#[async_trait]
pub trait VectorDbRead: Send + Sync {
	async fn nearest(&self, query: &[f32], top_k: usize) -> Result<Vec<Hit>>;
	fn dim(&self) -> usize;
}

#[async_trait]
pub trait VectorDbWrite: Send + Sync {
	async fn upsert(&self, docs: &[Doc]) -> Result<()>;
}

/// 作成（テーブル確保）とメタ読み書きを含む管理 I/F
#[async_trait]
pub trait VectorDbSetup: Send + Sync {
	async fn ensure_created(&self) -> Result<()>;
	async fn write_meta(&self, meta: Meta) -> Result<()>;
	async fn latest_meta(&self) -> Result<Option<Meta>>;
}


