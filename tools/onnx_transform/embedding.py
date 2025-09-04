import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class EmbeddingWrapper(torch.nn.Module):
    """
    HF Transformer の隠れ状態から埋め込みを作る薄いラッパー。
    - pooling: mean / cls / max
    - normalize: L2 正規化（cosine 類似度向け）
    """

    def __init__(
        self, base_model: torch.nn.Module, pooling: str = "mean", normalize: bool = True
    ):
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        self.normalize = normalize

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        last_hidden = outputs.last_hidden_state  # [B, S, H]

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, S, 1]
            summed = (last_hidden * mask).sum(dim=1)  # [B, H]
            lengths = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            emb = summed / lengths
        elif self.pooling == "cls":
            emb = last_hidden[:, 0, :]  # [B, H]
        elif self.pooling == "max":
            mask = (attention_mask == 0).unsqueeze(-1)  # [B, S, 1]
            masked = last_hidden.masked_fill(mask, float("-inf"))
            emb, _ = masked.max(dim=1)  # [B, H]
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)

        return emb


def _ensure_pad_token(tokenizer, model):
    """
    pad_token が無い場合にできる限り安全に補う。
    必要なら語彙拡張し、埋め込みもリサイズする。
    """
    if tokenizer.pad_token_id is not None:
        return

    for candidate_attr in ["eos_token", "bos_token", "cls_token", "sep_token"]:
        tok = getattr(tokenizer, candidate_attr, None)
        if tok:
            tokenizer.pad_token = tok
            return

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass


def export_to_onnx(
    model_id: str,
    output_dir: Path,
    pooling: str = "mean",
    normalize: bool = True,
    opset: int = 17,
    max_length: int = 512,
    trust_remote_code: bool = True,
    test_sentences: Optional[List[str]] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust_remote_code
    )
    base_model = AutoModel.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    base_model.eval().to(device)

    _ensure_pad_token(tokenizer, base_model)

    wrapper = (
        EmbeddingWrapper(base_model, pooling=pooling, normalize=normalize)
        .eval()
        .to(device)
    )

    dummy = tokenizer(
        ["hello world"],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"].to(device)
    attention_mask = dummy["attention_mask"].to(device)

    onnx_path = output_dir / "model.onnx"

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        onnx_path.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "embeddings": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    if test_sentences:
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(
                (output_dir / ("model.onnx")).as_posix(),
                providers=["CPUExecutionProvider"],
            )
            tokens = tokenizer(
                test_sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            feeds = {
                "input_ids": tokens["input_ids"].numpy(),
                "attention_mask": tokens["attention_mask"].numpy(),
            }
            outs = sess.run(["embeddings"], feeds)
            print(f"ONNX output shape: {outs[0].shape}")
        except Exception as e:
            print(f"[warn] validation skipped: {e}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert HF embedding model to ONNX (with pooling)."
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="cl-nagoya/ruri-v3-30m",
        help="Hugging Face model id",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="tools/onnx_transform/out/ruri-v3-30m-onnx",
        help="Output directory",
    )
    p.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "cls", "max"],
        default="mean",
        help="Pooling strategy",
    )
    p.add_argument(
        "--no-normalize", action="store_true", help="Disable L2 normalization"
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for export and test",
    )
    p.add_argument(
        "--no-trust-remote-code", action="store_true", help="Disable trust_remote_code"
    )
    p.add_argument(
        "--test",
        type=str,
        nargs="*",
        default=["テストです", "今日はいい天気ですね"],
        help="Sentences for a quick validation",
    )
    return p.parse_args()


def main():
    args = parse_args()
    export_to_onnx(
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        pooling=args.pooling,
        normalize=not args.no_normalize,
        opset=args.opset,
        max_length=args.max_length,
        trust_remote_code=not args.no_trust_remote_code,
        test_sentences=args.test,
    )
    print("Done.")


if __name__ == "__main__":
    main()
