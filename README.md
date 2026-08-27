# ComfyUI Krea Multi-Shot Stills (+ LTX 2.5 MSR)

Custom nodes for a **one photo → character sheet → multi-scene video** pipeline in ComfyUI.

**GitHub:** https://github.com/OsamaAtiq12/ComfyUI-Krea-MultiShot-Stills

---

## What you get

| Node | What it does |
|------|----------------|
| **Krea Photo → Character Sheet** | One reference photo → 4-view lookbook character sheet |
| **Krea CharSheet Scenes** | Full sheet → first/last stills per scene (FLF path) |
| **LTX 2.5 Multi-Scene MSR** | Character sheet + scene prompts → N videos → combined MP4 (Licon MSR) |
| **LTX 2.5 Multi-Scene FLF2V** | First/last stills → N videos (first-last frame path) |

Web UI: **+ Add scene** buttons for dynamic scene counts.

---

## Dependencies (install these too)

1. [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) — Identity Edit / grounded encode  
2. [ComfyUI-LTX2.5-MSR](https://github.com/liconstudio/ComfyUI-LTX2.5-MSR) — Licon MSR LoRA loader + guide (for MSR node)

ComfyUI itself must support **Krea 2** and **LTX 2.5** natively (recent build).

---

## Install

### A — Git URL (easiest)

In a terminal, from your `ComfyUI/custom_nodes` folder:

```bash
git clone https://github.com/OsamaAtiq12/ComfyUI-Krea-MultiShot-Stills.git
git clone https://github.com/lbouaraba/comfyui-krea2edit.git
git clone https://github.com/liconstudio/ComfyUI-LTX2.5-MSR.git
```

Restart ComfyUI.

### B — ComfyUI Manager

**Install via Git URL** → paste:

```
https://github.com/OsamaAtiq12/ComfyUI-Krea-MultiShot-Stills
```

Also install the two dependency repos above. Restart ComfyUI.

---

## Models (download separately)

### Stage 1 — Krea character sheet

| Model | Typical folder |
|-------|----------------|
| `krea2_turbo_fp8_scaled.safetensors` | `models/diffusion_models/` or `unet/` |
| `qwen3vl_4b_fp8_scaled.safetensors` (type **krea2**) | `models/text_encoders/` |
| `qwen_image_vae.safetensors` | `models/vae/` |
| `krea2_identity_edit_v1_2.safetensors` | `models/loras/Krea2/` |

### Stage 2 — LTX 2.5 MSR video

| Model | Typical folder |
|-------|----------------|
| LTX 2.5 distilled transformer | `models/diffusion_models/` |
| LTX 2.5 video VAE | `models/vae/` |
| LTX 2.5 audio VAE | `models/vae/` |
| Gemma text encoder for LTX 2.5 (type **ltxv**) | `models/text_encoders/` |
| `LTX-2.5-Licon-MSR-V1.safetensors` | `models/loras/` |

Exact filenames can vary by download source — match what your ComfyUI model picker shows.

---

## Ready-made workflow

Use the Gumroad / product pack file:

**`CharSheet_Krea_LTX25_MSR.json`**

1. Open ComfyUI  
2. Load that workflow  
3. Put your photo in **LOAD PHOTO**  
4. Queue  

Flow:

```
One photo
   → Krea (+ Identity Edit) → character sheet
   → LTX 2.5 + Licon MSR → multi-scene video + audio
```

---

## MSR prompting tip

In each scene prompt, mention **Image 1** as the person from the character sheet.  
Describe action + camera + `Audible …` for sound.

---

## License

See `LICENSE` in this repository.
