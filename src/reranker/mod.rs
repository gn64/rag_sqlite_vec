use anyhow::Result;
use ndarray::Array2;
use ort::{
    session::builder::{GraphOptimizationLevel, SessionBuilder},
    session::{Session, SessionInputValue},
    value::Value,
};
use rust_tokenizers::tokenizer::{SentencePieceTokenizer, Tokenizer, TruncationStrategy};
use std::borrow::Cow;

pub struct Reranker {
    tokenizer: SentencePieceTokenizer,
    session: Session,
    max_len: usize,
    pad_id: i64,
    bos_id: i64,
    eos_id: i64,
}

impl Reranker {
    pub fn new(
        onnx_path: &str,
        sp_model_path: &str,
        max_len: usize,
        pad_id: i64,
        bos_id: i64,
        eos_id: i64,
    ) -> Result<Self> {
        let tokenizer = SentencePieceTokenizer::from_file(sp_model_path, false)?;

        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(onnx_path)?;

        Ok(Self { tokenizer, session, max_len, pad_id, bos_id, eos_id })
    }

    pub fn score(&mut self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        let (ids, mask) = self.encode_pairs(pairs)?;
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
        Ok(data.to_vec())
    }

    pub fn rank(&mut self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        let pairs: Vec<(&str, &str)> = documents.iter().map(|d| (query, *d)).collect();
        let scores = self.score(&pairs)?;
        let mut idx_scores: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        idx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(idx_scores)
    }

    pub fn score_sigmoid(&mut self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        let logits = self.score(pairs)?;
        Ok(logits.into_iter().map(sigmoid).collect())
    }

    pub fn rank_sigmoid(&mut self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        let pairs: Vec<(&str, &str)> = documents.iter().map(|d| (query, *d)).collect();
        let probs = self.score_sigmoid(&pairs)?;
        let mut idx_scores: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        idx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(idx_scores)
    }

    pub fn score_logit(&mut self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        // ロジットはそのまま raw 出力（= self.score）
        self.score(pairs)
    }

    pub fn rank_logit(&mut self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        // ロジットでの順位（= rank と同等）
        self.rank(query, documents)
    }

    fn encode_pairs(&self, pairs: &[(&str, &str)]) -> Result<(Array2<i64>, Array2<i64>)> {
        let b = pairs.len();
        let s = self.max_len;

        let mut ids: Vec<i64> = Vec::with_capacity(b * s);
        let mut mask: Vec<i64> = Vec::with_capacity(b * s);

        for (q, d) in pairs {
            let enc_q = self.tokenizer.encode(
                q,
                None,
                self.max_len,
                &TruncationStrategy::DoNotTruncate,
                0,
            );
            let enc_d = self.tokenizer.encode(
                d,
                None,
                self.max_len,
                &TruncationStrategy::DoNotTruncate,
                0,
            );
            let mut q_ids: Vec<i64> = enc_q.token_ids.iter().map(|&x| x as i64).collect();
            let mut d_ids: Vec<i64> = enc_d.token_ids.iter().map(|&x| x as i64).collect();

            // ModernBERT/RoBERTa style: <s> q </s></s> d </s>
            let special = 4; // <s>, </s>, </s>, </s>
            let budget = s.saturating_sub(special);
            truncate_longest_first_pair(&mut q_ids, &mut d_ids, budget);

            let mut vi: Vec<i64> = Vec::with_capacity(s);
            vi.push(self.bos_id);
            vi.extend(q_ids);
            vi.push(self.eos_id);
            vi.push(self.eos_id);
            vi.extend(d_ids);
            vi.push(self.eos_id);

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

fn truncate_longest_first_pair(a: &mut Vec<i64>, b: &mut Vec<i64>, budget: usize) {
    while a.len() + b.len() > budget {
        if a.len() >= b.len() {
            a.pop();
        } else {
            b.pop();
        }
    }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[cfg(test)]
mod reranker_tests {
    use super::*;

    #[test]
    fn test_reranker_scoring_and_ranking() -> Result<()> {
        let onnx = std::env::var("RERANKER_ONNX")
            .unwrap_or_else(|_| "tools/onnx_transform/out/japanese-reranker-xsmall-v2-onnx/model.onnx".to_string());
        let spm = std::env::var("RERANKER_SPM")
            .unwrap_or_else(|_| "tools/onnx_transform/out/japanese-reranker-xsmall-v2-onnx/tokenizer.model".to_string());

        let pad_id = std::env::var("RERANKER_PAD_ID").ok()
            .and_then(|s| s.parse::<i64>().ok()).unwrap_or(3);
        let bos_id = std::env::var("RERANKER_BOS_ID").ok()
            .and_then(|s| s.parse::<i64>().ok()).unwrap_or(1);
        let eos_id = std::env::var("RERANKER_EOS_ID").ok()
            .and_then(|s| s.parse::<i64>().ok()).unwrap_or(2);

        let mut rr = Reranker::new(&onnx, &spm, 512, pad_id, bos_id, eos_id)?;

        let inputs = vec![
            ("瑠璃色はどんな色？",
             "瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。"),
            ("瑠璃色 なに",
             "瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。"),
            ("瑠璃色はどんな色？",
             "ワシ、タカ、ハゲワシ、ハヤブサ、コンドル、フクロウが代表的である。これらの猛禽類はリンネ前後の時代(17~18世紀)には鷲類・鷹類・隼類及び梟類に分類された。ちなみにリンネは狩りをする鳥を単一の目(もく)にまとめ、vultur(コンドル、ハゲワシ)、falco(ワシ、タカ、ハヤブサなど)、strix(フクロウ)、lanius(モズ)の4属を含めている。"),
            ("ワシやタカのように、鋭いくちばしと爪を持った大型の鳥類を総称して「何類」というでしょう?",
             "ワシ、タカ、ハゲワシ、ハヤブサ、コンドル、フクロウが代表的である。これらの猛禽類はリンネ前後の時代(17~18世紀)には鷲類・鷹類・隼類及び梟類に分類された。ちなみにリンネは狩りをする鳥を単一の目(もく)にまとめ、vultur(コンドル、ハゲワシ)、falco(ワシ、タカ、ハヤブサなど)、strix(フクロウ)、lanius(モズ)の4属を含めている。"),
            ("ワシやタカのように、鋭いくちばしと爪を持った大型の鳥類を総称して「何類」というでしょう?",
             "瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。"),
        ];

        let scores = rr.score(&inputs)?;
        assert_eq!(scores.len(), 5);
        println!("scores = {:?}", scores);

        let probs = rr.score_sigmoid(&inputs)?;
        println!("probs  = {:?}", probs);

        let query = "瑠璃色はどんな色？";
        let documents = vec![
            "ワシ、タカ、ハゲワシ、ハヤブサ、コンドル、フクロウが代表的である。これらの猛禽類はリンネ前後の時代(17~18世紀)には鷲類・鷹類・隼類及び梟類に分類された。ちなみにリンネは狩りをする鳥を単一の目(もく)にまとめ、vultur(コンドル、ハゲワシ)、falco(ワシ、タカ、ハヤブサなど)、strix(フクロウ)、lanius(モズ)の4属を含めている。",
            "瑠璃、または琉璃（るり）は、仏教の七宝の一つ。サンスクリットの vaiḍūrya またはそのプラークリット形の音訳である。金緑石のこととも、ラピスラズリであるともいう[1]。",
            "瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。",
        ];
        let ranked = rr.rank(query, &documents)?;
        println!("ranked = {:?}", ranked);
        let ranked_sigmoid = rr.rank_sigmoid(query, &documents)?;
        println!("ranked_sigmoid = {:?}", ranked_sigmoid);
        let ranked_logit = rr.rank_logit(query, &documents)?;
        println!("ranked_logit = {:?}", ranked_logit);

        assert_eq!(ranked[0].0, 2);
        //scores = [6.5505004, 4.4772873, -6.05822, 2.1553423, -5.4889135]
        //probs  = [0.9985726, 0.98876345, 0.002333104, 0.8961669, 0.0041153254]
        //ranked = [(2, 6.5505004), (1, -0.008894408), (0, -6.05822)]
        //ranked_sigmoid = [(2, 0.9985726), (1, 0.49777642), (0, 0.002333104)]
        //ranked_logit = [(2, 6.5505004), (1, -0.008894408), (0, -6.05822)]
        Ok(())
    }
}


