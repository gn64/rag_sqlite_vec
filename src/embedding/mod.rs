use anyhow::Result;
use ndarray::Array2;
use ort::{session::builder::{GraphOptimizationLevel, SessionBuilder}, session::{Session, SessionInputValue}, value::Value};
use std::borrow::Cow;
use rust_tokenizers::tokenizer::{SentencePieceTokenizer, Tokenizer, TruncationStrategy};

pub struct Embedder {
    tokenizer: SentencePieceTokenizer,
    session: Session,
    max_len: usize,
    pad_id: i64,
    bos_id: i64,
    eos_id: i64,
}

impl Embedder {
    pub fn new(onnx_path: &str, sp_model_path: &str, max_len: usize, pad_id: i64, bos_id: i64, eos_id: i64) -> Result<Self> {
        // SentencePiece tokenizer
        let tokenizer = SentencePieceTokenizer::from_file(sp_model_path, false)?;

        // ORT session
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(onnx_path)?;

        Ok(Self { tokenizer, session, max_len, pad_id, bos_id, eos_id })
    }

    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let (ids, mask) = self.encode_batch(texts)?;
        let b = ids.shape()[0];
        let s = ids.shape()[1];
        let ids_vec = ids.into_raw_vec();
        let mask_vec = mask.into_raw_vec();
        let ids_v = Value::from_array((vec![b as i64, s as i64], ids_vec))?;
        let mask_v = Value::from_array((vec![b as i64, s as i64], mask_vec))?;
        let inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
            (Cow::Borrowed("input_ids"), ids_v.into()),
            (Cow::Borrowed("attention_mask"), mask_v.into()),
        ];
        let outputs = self.session.run(inputs)?;

        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let b = texts.len();
        let total = data.len();
        let h = if b > 0 { total / b } else { 0 };
        let mut result = Vec::with_capacity(b);
        for i in 0..b {
            let start = i * h;
            let end = start + h;
            result.push(data[start..end].to_vec());
        }
        Ok(result)
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<(Array2<i64>, Array2<i64>)> {
        let b = texts.len();
        let s = self.max_len;

        let mut ids: Vec<i64> = Vec::with_capacity(b * s);
        let mut mask: Vec<i64> = Vec::with_capacity(b * s);

        for t in texts {
            // 1+3 prefix は呼び出し側で文頭に付与してください
            let encoding = self.tokenizer.encode(
                t,
                None,
                self.max_len.saturating_sub(2),
                &TruncationStrategy::LongestFirst,
                0,
            );

            let mut vi: Vec<i64> = encoding.token_ids.iter().map(|&x| x as i64).collect();
            // add BOS/EOS to match tokenizer config
            vi.insert(0, self.bos_id);
            vi.push(self.eos_id);
            // attention mask: tokens=1, paddings=0
            let mut vm: Vec<i64> = vec![1; vi.len()];
            if vi.len() < s {
                vi.resize(s, self.pad_id);
                vm.resize(s, 0);
            } else if vi.len() > s {
                vi.truncate(s);
                vm.truncate(s);
            }
            ids.extend(vi);
            mask.extend(vm);
        }

        let ids_arr = Array2::from_shape_vec((b, s), ids)?;
        let mask_arr = Array2::from_shape_vec((b, s), mask)?;
        Ok((ids_arr, mask_arr))
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;

    #[test]
    fn test_embedder_ruri_v3_30m() -> Result<()> {
        let onnx = std::env::var("RURI_ONNX")
            .unwrap_or_else(|_| "tools/onnx_transform/out/ruri-v3-30m-onnx/model.onnx".to_string());
        let spm = std::env::var("RURI_SPM")
            .unwrap_or_else(|_| "tools/onnx_transform/out/ruri-v3-30m-onnx/tokenizer.model".to_string());

        let mut embedder = Embedder::new(&onnx, &spm, 512, 3, 1, 2)?;
        let sentences = vec![
            "川べりでサーフボードを持った人たちがいます",
            "サーファーたちが川べりに立っています",
            "トピック: 瑠璃色のサーファー",
            "検索クエリ: 瑠璃色はどんな色？",
            "検索文書: 瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。",
        ];
        let batch: Vec<&str> = sentences.iter().map(|s| s.as_ref()).collect();

        let emb = embedder.embed(&batch)?;
        assert_eq!(emb.len(), 5);
        println!("emb size = {} x {}", emb.len(), emb[0].len());

        for i in 0..emb.len() {
            let mut row = Vec::with_capacity(emb.len());
            for j in 0..emb.len() {
                row.push(format!("{:.4}", super::cosine(&emb[i], &emb[j])));
            }
            println!("[{}]", row.join(", "));
        }
        Ok(())
        // ---- embedding::embedding_tests::test_embedder_ruri_v3_30m stdout ----
        // emb size = 5 x 256
        // [1.0000, 0.9588, 0.8587, 0.7479, 0.7329]
        // [0.9588, 1.0000, 0.8609, 0.7578, 0.7389]
        // [0.8587, 0.8609, 1.0000, 0.8974, 0.8651]
        // [0.7479, 0.7578, 0.8974, 1.0000, 0.9477]
        // [0.7329, 0.7389, 0.8651, 0.9477, 1.0000]
    }
}

