# Qwen 3.5 4B Text Encoder for Anima 2B — Forge Neo Extension

A Forge Neo extension that adds support for the **Qwen 3.5 4B** hybrid (Mamba2 + Attention) text encoder for use with the **Anima 2B** diffusion model.

The base Anima 2B ships with a Qwen 3 0.6B text encoder. This extension enables the larger Qwen 3.5 4B variant from [cosmos-qwen3.5](https://huggingface.co/nightknocker/cosmos-qwen3.5/tree/main/4b), which uses a hybrid SSM/attention architecture for improved text understanding — roughly 7× more parameters dedicated to reading your prompt.

Ported from the [ComfyUI custom node](https://github.com/GumGum10/comfyui-qwen35-anima) with identical model logic.

## Architecture

Qwen 3.5 4B is **not** a standard transformer — it's a hybrid model alternating between Mamba2-style selective state space (SSM) blocks and gated self-attention:

- **32 layers** total: 24 SSM + 8 self-attention (at positions 3, 7, 11, 15, 19, 23, 27, 31)
- **Hidden size**: 2560, **Output dim**: 1024 (matching Anima's expected embedding size)
- **Vocab**: 248,320 tokens
- **FP8 quantized** (F8_E4M3) weights with BF16 norms

## Installation

All model files are available at: **[lylogummy/anima2b-qwen-3.5-4b](https://huggingface.co/lylogummy/anima2b-qwen-3.5-4b)**

1. Clone or copy this extension into your Forge Neo `extensions` directory:

   ```
   sd-webui-forge-neo/extensions/sd_forge_qwen35_encoder/
   ```

2. Download `qwen35_4b.safetensors` from [text_encoders/](https://huggingface.co/lylogummy/anima2b-qwen-3.5-4b/tree/main/text_encoders) and place it in:

   ```
   sd-webui-forge-neo/models/text_encoder/qwen35_4b.safetensors
   ```

3. The calibration and alignment files (`calibration_params.safetensors`, `rotation_matrix.safetensors`) and the tokenizer folder (`qwen35_tokenizer/`) are **bundled with the extension** — no extra download needed.

4. Restart Forge Neo. The extension will auto-install `transformers` and `safetensors` if not already present.

## Usage

1. Load an **Anima 2B** checkpoint in Forge Neo
2. Keep **`qwen_3_06b_base.safetensors`** selected in the top VAE / Text Encoder dropdown — the LLM adapter on the 0.6B model is still needed
3. In the generation tab, expand **"Qwen3.5 Text Encoder (Anima)"** and check the enable box
4. Select **`qwen35_4b.safetensors`** in the extension's Model File dropdown
5. Generate as normal — the extension intercepts text encoding automatically

> **Do NOT** put `qwen35_4b.safetensors` in the top VAE/Text Encoder dropdown. That dropdown is for the stock 0.6B encoder only. Forge will crash if it tries to load the 4B file as a 0.6B model.

### Recommended Starting Settings

| Setting | Value |
|---------|-------|
| Use Alignment | ON |
| Alignment Strength | 0.5 |
| Use Calibration | OFF |
| Output Scale | 1.0 |

## Settings Reference

| Setting | What It Does | When to Use |
|---|---|---|
| **Use Calibration** | Per-dimension affine scaling to match 0.6B magnitude distribution | Optional fine-grained magnitude calibration |
| **Use Alignment** | Procrustes rotation to align 4B concept directions with 0.6B | Recommended — fixes pose, viewpoint, spatial understanding |
| **Alignment Strength** (0–1) | Controls bias shift: 0 = keep 4B magnitude, 1 = shift to 0.6B magnitude. Rotation is always applied when alignment is on. | Start at 0.5, adjust to taste |
| **Output Scale** | Uniform multiplier on the final output | Usually leave at 1.0 |

## VRAM Usage

The 4B encoder weights are FP8 quantized, requiring roughly ~4 GB of VRAM for the text encoder portion (vs ~0.6 GB for the stock 0.6B). The extension manages GPU memory through Forge's model patcher system and reserves the extra memory during sampling to prevent OOM.

## Requirements

- Forge Neo (sd-webui-forge-neo)
- An Anima 2B checkpoint (e.g. `animaFp8_preview.safetensors`)
- The stock `qwen_3_06b_base.safetensors` text encoder (for the LLM adapter)
- The Qwen 3.5 4B text encoder weights (`qwen35_4b.safetensors`)
- Python packages: `transformers>=4.40.0`, `safetensors` (auto-installed)

## How It Works

The extension hooks into Forge's processing pipeline at the `process_batch` stage (before conditioning is computed). It:

1. Loads the Qwen 3.5 4B hybrid model and its dedicated tokenizer
2. Creates a CLIP patcher wrapper for GPU memory management
3. Monkeypatches `sd_model.get_learned_conditioning` to route text encoding through the 4B model
4. Uses the original 0.6B model's LLM adapter for the final embedding preprocessing (the adapter was trained against the 0.6B's output format)
5. Applies optional Procrustes alignment and/or per-dimension calibration to map the 4B output into the 0.6B's embedding space
6. Restores the original text encoder after generation completes

Generation parameters (`q35_model`, `q35_alignment`, etc.) are saved in the image metadata for reproducibility.

## Credits

- **Anima 2B**: [circlestone-labs](https://huggingface.co/circlestone-labs/Anima)
- **Qwen 3.5 4B for Anima**: [nightknocker/cosmos-qwen3.5](https://huggingface.co/nightknocker/cosmos-qwen3.5)
- **Original ComfyUI Node & Alignment**: [GumGum10](https://github.com/GumGum10/comfyui-qwen35-anima)

## License

MIT
