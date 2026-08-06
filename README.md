# ComfyUI Krea Multi-Shot Stills

Generate **multiple Krea 2 still images** from a single character reference sheet — one text prompt per shot — and save them automatically to `ComfyUI/input/` for use in **LTX Director**, Advanced I2V, AntiDrift, or any image-to-video workflow.

This pack adds one node: **`KreaMultiShotStills`** (display name: **Krea Multi-Shot Stills**).

It does **not** modify LTX Director or run video generation. It only batch-generates stills so you can drop them onto a Director timeline yourself.

---

## What it does

| Step | What happens |
|------|----------------|
| 1 | You connect Krea 2 model, CLIP, VAE, reference sheet, negative, and empty latent |
| 2 | You fill `shot_1` … `shot_8` with **image** prompts (leave unused shots empty) |
| 3 | Queue → one Krea render per filled shot |
| 4 | Files written to `input/<prefix>_1.png`, `input/<prefix>_2.png`, … (default prefix: `multishot`) |

Typical pipeline:

```
Reference sheet → Krea Multi-Shot Stills → multishot_1.png … multishot_7.png
                                              ↓
                                    LTX Director (image guides + video prompts)
                                              ↓
                                         final MP4
```

---

## Requirements

### ComfyUI

- ComfyUI with **native Krea 2** support (recent build)
- **Python 3.10+**

### Required custom node (install first)

This pack **depends on** [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) for grounded Krea encoding (`Krea2EditGroundedEncode`). Without it, the node will error on run.

### Models (not included — download separately)

| Asset | Example filename | Folder |
|-------|------------------|--------|
| Krea 2 diffusion model | `krea2_turbo_fp8_scaled.safetensors` | `models/diffusion_models/` or `models/unet/` |
| Qwen3-VL text encoder (Krea 2) | `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders/` |
| Krea 2 VAE | per your Krea workflow | `models/vae/` |
| **Identity Edit LoRA** | `krea2_identity_edit_v1_2.safetensors` | `models/loras/` |
| Optional realism LoRA | e.g. `Krea2-realism-V2.safetensors` | `models/loras/` |

### Optional (for video after stills)

- [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI) — **LTX Director** timeline
- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) — LTX 2.3 nodes
- LTX 2.3 checkpoint / fp8 weights and spatial upscaler (see your LTX workflow)

---

## Installation

### Method A — ComfyUI Manager (recommended)

1. Open ComfyUI → **Manager** → **Install Custom Nodes**
2. Choose **Install via Git URL**
3. Paste:
   ```
   https://github.com/CodingWithShahzaib/comfyui-multishot-plan
   ```
4. Install the dependency if you have not already:
   ```
   https://github.com/lbouaraba/comfyui-krea2edit
   ```
5. **Restart ComfyUI**

### Method B — Git clone

From your ComfyUI root:

```bash
cd ComfyUI/custom_nodes

# Required dependency (do this first)
git clone https://github.com/lbouaraba/comfyui-krea2edit

# This pack
git clone https://github.com/CodingWithShahzaib/comfyui-multishot-plan
```

Restart ComfyUI.

### Method C — Manual copy

1. Download or copy the `comfyui-multishot-plan` folder into `ComfyUI/custom_nodes/`
2. Ensure `comfyui-krea2edit` is in `custom_nodes/` beside it
3. Restart ComfyUI

### Verify installation

After restart, search the node menu for:

- **Krea Multi-Shot Stills** (category: `multishot`)

If it is missing, check the terminal for import errors and confirm `comfyui-krea2edit` is installed.

---

## Quick start (minimal workflow)

1. **Load models** (same as any Krea Identity Edit workflow):
   - Krea 2 UNET
   - Qwen3-VL CLIP (`type: krea2`)
   - VAE
   - Apply **Krea2EditModelPatch** + Identity Edit LoRA to the model path

2. **Add nodes**:
   - `Load Image` → your character reference sheet
   - `Empty Latent Image` → e.g. **1280×720** or **768×512** (width and height divisible by **32**)
   - Negative prompt → `CLIP Text Encode` or your usual Krea negative
   - **`Krea Multi-Shot Stills`**

3. **Wire**:
   - `model`, `clip`, `vae`, `reference_image`, `negative`, `latent_image` → Multi-Shot Stills

4. **Fill prompts** (example — one scene per shot):

   | Shot | Image prompt (still) |
   |------|----------------------|
   | shot_1 | Medium wide, same woman at rain-streaked loft window at dawn, silk robe, blue hour, photorealistic, exact face identity |
   | shot_2 | Medium close-up, black turtleneck, desk phone, warm lamp, photorealistic, exact face identity |
   | shot_3 | Full body hallway, charcoal coat, keys by door, photorealistic, exact face identity |
   | … | Leave shot_4–shot_8 empty until needed |

5. **Settings** (defaults are fine to start):
   - `filename_prefix`: `multishot` → writes `input/multishot_1.png`, etc.
   - `steps`: 12 (turbo) or higher for quality
   - `cfg`: 1.0
   - `grounding_px`: 768
   - `overwrite`: ON (replace existing files)

6. **Queue** → check `ComfyUI/input/` for your PNGs.

---

## Using stills with LTX Director

1. Open your LTX Director workflow (e.g. Krea Multi-Scene + LTX Director).
2. On the **MAIN** timeline, add **image segments** in order.
3. Drag or assign `multishot_1.png` … `multishot_N.png` to each segment.
4. Put **video / motion prompts** in each segment (I2V style — verbs + camera, not full scene re-description).
5. Set **global prompt** on Director for identity lock, e.g.  
   *Same woman throughout, exact face identity from reference sheet. Hard cuts between scenes. No dissolve, no morph.*
6. Recommended Director settings for photoreal multi-cut:
   - `divisible_by`: **32**
   - `resize_method`: **crop** (match your still aspect)
   - Guide strength: **0.75–0.85** per image segment
   - Avoid stacking MSR / VBVR LoRAs on top of many hard cuts

**Frame counts:** LTX wants total frames = **8n + 1** (e.g. 497 frames ≈ 20.7 s @ 24 fps for seven ~3 s scenes).

---

## Node reference

### Inputs

| Input | Type | Notes |
|-------|------|--------|
| `model` | MODEL | Krea 2 with Identity Edit patch / LoRA applied |
| `clip` | CLIP | Qwen3-VL for Krea 2 |
| `vae` | VAE | Krea VAE |
| `reference_image` | IMAGE | Character reference sheet |
| `negative` | CONDITIONING | Standard negative |
| `latent_image` | LATENT | Empty latent (target resolution) |
| `shot_1` … `shot_8` | STRING | Image prompts; **empty = skip** |
| `filename_prefix` | STRING | Output base name (default `multishot`) |
| `seed` | INT | Base seed; each shot uses `seed + index` |
| `steps` | INT | Default 12 |
| `cfg` | FLOAT | Default 1.0 |
| `sampler_name` | COMBO | Any ComfyUI sampler |
| `scheduler` | COMBO | Any ComfyUI scheduler |
| `denoise` | FLOAT | Default 1.0 |
| `grounding_px` | INT | Passed to Krea2Edit grounded encode (default 768) |
| `overwrite` | BOOLEAN | Replace existing PNGs in `input/` |

### Outputs

| Output | Description |
|--------|-------------|
| `images` | Batch preview of all generated shots |
| `shot_count` | Number of shots generated |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `comfyui-krea2edit is required` | Install [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) and restart |
| Node not in menu | Restart ComfyUI; confirm folder name is `comfyui-multishot-plan` under `custom_nodes/` |
| Face drift between shots | Stronger identity LoRA; keep “same woman, exact face identity” in every shot prompt; use one reference sheet |
| Wrong output size | Set empty latent to your target 32-aligned resolution before generating |
| Files not updating | Set `overwrite` to **true** or delete old `multishot_N.png` in `input/` |
| LTX artifacts on hard cuts | Lower guide strength; use `LTXDirectorCropGuides`; fewer scenes per pass; see LTX 2.3 prompting guide |

---

## Updating

**Manager:** Manager → update custom nodes → restart.

**Git:**

```bash
cd ComfyUI/custom_nodes/comfyui-multishot-plan
git pull
```

Restart ComfyUI.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Credits

- **Krea 2 Identity Edit** encoding: [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) by Conrad Locke
- Designed to pair with **LTX Director** ([WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI)) and LTX 2.3 video workflows

Issues and feature requests: [GitHub Issues](https://github.com/CodingWithShahzaib/comfyui-multishot-plan/issues)
