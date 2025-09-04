import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RerankerWrapper(torch.nn.Module):
    """
    CrossEncoder (sequence-classification) をスコア出力だけに絞った薄いラッパー。
    入力は query と document のペアをテンプレートで結合した単一系列。
    出力は [B] のスコア（float32）。
    """

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits  # [B, 1] or [B]
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return logits


def _ensure_pad_token(tokenizer, model):
    if tokenizer.pad_token_id is not None:
        return
    # 既存の special token を転用
    for candidate in ["eos_token", "bos_token", "cls_token", "sep_token"]:
        tok = getattr(tokenizer, candidate, None)
        if tok:
            tokenizer.pad_token = tok
            return
    # どうしても無ければ追加
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass


def _pair_encode(
    tokenizer,
    pairs: List[Tuple[str, str]],
    max_length: int,
):
    texts_a = [q for q, _ in pairs]
    texts_b = [d for _, d in pairs]
    return tokenizer(
        texts_a,
        texts_b,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def export_to_onnx(
    model_id: str,
    output_dir: Path,
    opset: int = 17,
    max_length: int = 512,
    trust_remote_code: bool = True,
    test_pairs: Optional[List[Tuple[str, str]]] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust_remote_code
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    base_model.eval().to(device)

    _ensure_pad_token(tokenizer, base_model)

    wrapper = RerankerWrapper(base_model).eval().to(device)

    # ダミー入力（query, document のペア）
    dummy_inputs = _pair_encode(
        tokenizer,
        pairs=[("query", "document")],
        max_length=max_length,
    )
    input_ids = dummy_inputs["input_ids"].to(device)
    attention_mask = dummy_inputs["attention_mask"].to(device)

    onnx_path = output_dir / "model.onnx"

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        onnx_path.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=["scores"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "scores": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # 任意の簡易検証（ONNX Runtime）
    if test_pairs:
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(
                onnx_path.as_posix(), providers=["CPUExecutionProvider"]
            )
            tokens = _pair_encode(tokenizer, test_pairs, max_length=max_length)
            feeds = {
                "input_ids": tokens["input_ids"].numpy(),
                "attention_mask": tokens["attention_mask"].numpy(),
            }
            outs = sess.run(["scores"], feeds)
            print(f"ONNX scores shape: {outs[0].shape}")
        except Exception as e:
            print(f"[warn] validation skipped: {e}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert HF cross-encoder reranker model to ONNX."
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="cl-nagoya/ruri-v3-reranker-310m",
        help="Hugging Face model id",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="tools/onnx_transform/out/ruri-v3-reranker-310m-onnx",
        help="Output directory",
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
        default=None,
        help=(
            "Test pairs given as: q1 d1 q2 d2 ... (even number). If provided, "
            "will run a quick ONNX Runtime forward pass."
        ),
    )
    return p.parse_args()


def _parse_test_pairs(flat: Optional[List[str]]) -> Optional[List[Tuple[str, str]]]:
    if not flat:
        return None
    if len(flat) % 2 != 0:
        raise ValueError("--test expects an even number of strings: q d q d ...")
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(flat), 2):
        pairs.append((flat[i], flat[i + 1]))
    return pairs


def main():
    args = parse_args()
    pairs = _parse_test_pairs(args.test)
    export_to_onnx(
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        opset=args.opset,
        max_length=args.max_length,
        trust_remote_code=not args.no_trust_remote_code,
        test_pairs=pairs,
    )
    print("Done.")


if __name__ == "__main__":
    main()
