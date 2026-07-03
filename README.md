ComfyUI-zyk-IndexTTS2
=================

Lightweight ComfyUI wrapper for IndexTTS 2 (voice cloning + emotion control) — **zyk fork**. Nodes call the upstream inference code so behaviour stays matched with the original repo.

Original repo: https://github.com/index-tts/index-tts

![zyk-ComfyUI-IndexTTS2 nodes](images/overview.png)

## Updates
- 2026-06-10: Added **zyk-IndexTTS2 Volume Adjust** (gain control + peak limiting) and **zyk-IndexTTS2 Denoise** (AI-powered noise reduction via DeepFilterNet) nodes.
- 2026-06-07: 增加自定义停顿功能的支持，允许用户在文本中增加 <rf_pause:100ms> 作为停顿表示，100ms表示停顿时间是100毫秒，停顿时长单位仅支持毫秒。
- 2025-10-13: Save Audio node now acts as an output node with an embedded player overlay for instant preview inside the graph (no need for downstream preview nodes).
- 2025-10-08: Default FP32 with optional FP16 toggle, output gain control, and a Save Audio helper node (wav/mp3 + quality parameters).
- 2025-09-22: Added zyk-IndexTTS2 Advanced node exposing sampling, speed, seed, and other generation controls.

## Install
- Clone this repository into `ComfyUI/custom_nodes/`
- Inside your ComfyUI Python environment:
  ```bash
  pip install wetext
  pip install -r requirements.txt
  ```
- **Optional** — The **Denoise** node requires `deepfilternet`, which needs the **Rust** toolchain on Windows:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  pip install deepfilternet
  ```
  If you don't need the Denoise node, you can skip this step — the other nodes work fine without it.

## Models
- Create `checkpoints/` in the repo root and copy the IndexTTS-2 release there (https://huggingface.co/IndexTeam/IndexTTS-2/tree/main). Missing files will be cached from Hugging Face automatically.
- The **Denoise** node automatically downloads its models from ModelScope (China, fast) or GitHub (fallback) on first use into `checkpoints/audio_denoise/`.

## Nodes

### TTS Nodes
- **zyk-IndexTTS2 Simple** - speaker audio, text, optional emotion audio/vector; outputs audio + status string. Default FP32, optional FP16 toggle, output gain control.
- **zyk-IndexTTS2 Advanced** - Simple inputs plus overrides for sampling, speech speed, pauses, CFG, seed, FP16 toggle, and output gain.
- **zyk-IndexTTS2 Emotion Vector** – eight sliders (0.0–1.4, sum <= 1.5) producing an emotion vector.
- **zyk-IndexTTS2 Emotion From Text** – requires ModelScope and local QwenEmotion; turns short text into an emotion vector + summary.
- **zyk-IndexTTS2 Save Audio** - saves generated audio tensors to disk with wav/mp3 options and surfaces an inline player directly on the node after execution.

### Audio Processing Nodes
- **zyk-IndexTTS2 Volume Adjust** - Adjust audio volume with dB gain control (-20 to +20 dB) and optional peak limiting to prevent clipping. Includes an `enabled` toggle to bypass the node.
- **zyk-IndexTTS2 Denoise** - AI-powered noise reduction using DeepFilterNet (v1/v2/v3). Parameters: model selection (3 versions), noise reduction strength (0-2), post-filter toggle, and `enabled` bypass switch. Models auto-download from ModelScope (China) or HuggingFace on first use.

### Parameters

| Node | Parameter | Type | Range | Default | Description |
|------|-----------|------|-------|---------|-------------|
| Volume Adjust | `enabled` | BOOLEAN | — | True | Disable to pass audio through unchanged |
| Volume Adjust | `gain_db` | FLOAT | -20 ~ +20 (step 0.5) | 0.0 | Volume adjustment in decibels |
| Volume Adjust | `peak_limit` | BOOLEAN | — | True | Prevent clipping when boosting volume |
| Denoise | `enabled` | BOOLEAN | — | True | Disable to pass audio through unchanged |
| Denoise | `model` | COMBO | DeepFilterNet/2/3 | DeepFilterNet2 | Denoising model version |
| Denoise | `strength` | FLOAT | 0.0 ~ 2.0 (step 0.1) | 1.0 | Noise reduction intensity |
| Denoise | `post_filter` | BOOLEAN | — | True | Extra residual noise suppression |

## Examples

### 自定义停顿标签

在文本中插入 `<rf_pause:Xms>` 标签可以控制语音停顿，其中 `X` 是停顿毫秒数。

| 用法 | 示例 | 效果 |
|------|------|------|
| 开头停顿 | `"<rf_pause:1000ms>你好"` | 先停顿 1 秒，再开始说话 |
| 句间停顿 | `"你好<rf_pause:500ms>世界"` | 说"你好"后停顿 500ms，再说"世界" |
| 结尾停顿 | `"你好<rf_pause:1000ms>"` | 说完"你好"后停顿 1 秒再结束 |
| 连续停顿 | `"你好<rf_pause:300ms><rf_pause:200ms>世界"` | 连续标签的停顿时间累加（共 500ms） |

> 注意：停顿时长单位仅支持毫秒（ms），标签大小写不敏感。支持在文本的**开头、中间、结尾**任意位置使用。

### Workflow 示例
- Speaker audio -> zyk-IndexTTS2 Simple -> Preview/Save Audio
- Speaker + emotion audio -> zyk-IndexTTS2 Simple -> Save
- Emotion Vector -> zyk-IndexTTS2 Simple -> Save
- Emotion From Text -> zyk-IndexTTS2 Simple -> Save
- TTS output -> Volume Adjust (+6dB, peak_limit) -> Save
- TTS output -> Denoise (DeepFilterNet2, strength 1.2) -> Save

## Troubleshooting
- Windows only so far; DeepSpeed is disabled.
- Install `wetext` if the module is missing on first launch.
- Emotion vector sum must stay <= 1.5.
- If `deepfilternet` fails to install, ensure Rust is installed (`curl https://sh.rustup.rs | sh`).
