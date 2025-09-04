# ONNX Transform

`cl-nagoya/ruri-v3-30m` のような HF の文字列埋め込みモデルを、プーリングと L2 正規化込みで ONNX へ変換します。オプションで動的量子化（int8, weight-only）も可能です。

## セットアップ

```bash
# プロジェクト直下で（必要に応じて venv を有効化）
pip install -U transformers torch onnx onnxruntime sentencepiece huggingface_hub
```

## 変換

```bash
python tools/onnx_transform/embedding.py \
  --model-id cl-nagoya/ruri-v3-30m \
  --output-dir tools/onnx_transform/out/ruri-v3-30m-onnx \
  --pooling mean \
  --opset 17 \
  --quantize
```

- `--pooling`: `mean`（推奨）/ `cls` / `max`
- `--no-normalize`: L2 正規化を無効化（既定は有効）
- `--quantize`: 動的量子化（int8, weight-only）を実施

出力:

- `model.onnx`（および `model.int8.onnx` 量子化時）
- トークナイザは HF 側から読み込んで実行時に使用してください

## 簡易検証

`--test` で与えた文を用いて ONNX Runtime で前向き実行し、出力形状を表示します。

## 注意

- モデルに `pad_token` が無い場合は自動で補います（必要に応じて語彙を拡張）。
- 既定のプーリングは多くの埋め込みモデルで一般的な mean pooling です。モデル仕様に合わせて変更してください。

### 使い方（最小例）

```bash
python tools/onnx_transform/main.py --model-id cl-nagoya/ruri-v3-30m --quantize
```

- 実行後、`tools/onnx_transform/out/ruri-v3-30m-onnx/model.onnx`（および `model.int8.onnx`）が生成されます。
- 実装は CPU 向けです。GPU は不要です。
- Rust 側で実行する際は `onnxruntime` などから `input_ids`, `attention_mask` を与えれば `embeddings` 出力が得られます。
