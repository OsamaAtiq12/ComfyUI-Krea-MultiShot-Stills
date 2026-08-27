"""LTX 2.5 multi-scene with Licon MSR (Multiple Subject Reference) LoRA.

Pipeline (per docs: ComfyUI-LTX2.5-MSR):
  UNET → MSR IC-LoRA Loader → for each scene:
    Empty video latent → MSR Multi-Reference Guide (pic1.. + sheet)
    → ConcatAV → sample → Separate → CropGuides → decode
  → stitch scenes with audio

Prompts must label refs as Image 1 / Image 2 / … (MSR training convention).
"""
from __future__ import annotations

import re
from pathlib import Path

import folder_paths
import torch

from .ltx_multiscene import DISTILLED_SIGMAS, LTX_NEGATIVE, _call, _parse_motions

MAX_SCENES = 8

MSR_REF_PREFIX = (
    "Image 1 is the exact young woman from the character reference "
    "(same face, hair, body, and outfit). "
)

DEFAULT_MSR_SCENES = {
    1: (
        MSR_REF_PREFIX
        + "She stands in a luxury hotel lobby in warm evening light, then walks "
        "toward the sliding glass exit doors with a natural gait. Slight camera track. "
        "Audible soft lobby ambience, footsteps on hard floor, distant murmur."
    ),
    2: (
        MSR_REF_PREFIX
        + "At a rooftop bar at night with city lights bokeh, she sits holding a glass, "
        "then stands and turns toward the skyline. Audible night city ambience, soft bar music, light wind."
    ),
    3: (
        MSR_REF_PREFIX
        + "She walks through a bright airport terminal with a small suitcase toward "
        "the departure gate screens. Camera tracks alongside. Audible terminal ambience, "
        "rolling suitcase wheels, distant PA."
    ),
    4: (
        MSR_REF_PREFIX
        + "Seated by an airplane window in soft cabin light, she looks at the clouds, "
        "then slowly turns toward camera with a small smile. Audible low cabin hum, seatbelt chime."
    ),
    5: (
        MSR_REF_PREFIX
        + "She steps out of a black taxi onto a sunlit city street with her suitcase, "
        "then walks into a boutique hotel entrance as a doorman holds the door. "
        "Audible city traffic, suitcase wheels, lobby doors."
    ),
}


def _parse_scene_prompts(scene_prompts: str, scene_count: int, **scene_kwargs) -> list[str]:
    """Prefer per-widget scene_N_prompt; fall back to --- blob."""
    out = []
    for i in range(1, scene_count + 1):
        w = (scene_kwargs.get(f"scene_{i}_prompt") or "").strip()
        if w:
            out.append(w)
        else:
            out.append("")
    if any(out):
        for i, p in enumerate(out):
            if not p:
                out[i] = DEFAULT_MSR_SCENES.get(
                    i + 1,
                    MSR_REF_PREFIX + f"Natural cinematic motion for scene {i + 1}, audible matching ambience.",
                )
        return out
    # blob from linked STRING (optional)
    return _parse_motions(scene_prompts, scene_count)


class LTX25MultiSceneMSR:
    """Dynamic N-scene LTX 2.5 video using Licon MSR multi-reference LoRA."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "audio_vae": ("VAE",),
            "pic1": ("IMAGE",),
            "scene_count": ("INT", {"default": 5, "min": 1, "max": MAX_SCENES}),
        }
        for i in range(1, MAX_SCENES + 1):
            required[f"scene_{i}_prompt"] = (
                "STRING",
                {
                    "multiline": True,
                    "default": DEFAULT_MSR_SCENES.get(i, ""),
                },
            )
        required.update(
            {
                "lora_name": (loras,),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "msr_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_frames": (["25", "33"],),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 32}),
                "duration_sec": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "save_each_scene": ("BOOLEAN", {"default": True}),
                "scene_prompts": ("STRING", {"multiline": True, "default": ""}),
            }
        )
        return {
            "required": required,
            "optional": {
                "pic2": ("IMAGE",),
                "pic3": ("IMAGE",),
                "pic4": ("IMAGE",),
                "background": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "INT")
    RETURN_NAMES = ("combined_video", "all_frames", "scene_count")
    FUNCTION = "generate"
    CATEGORY = "multishot"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "LTX 2.5 + Licon MSR LoRA: character refs (Image 1..) → N scene videos → "
        "one combined clip with audio. Requires ComfyUI-LTX2.5-MSR + LTX-2.5-Licon-MSR-V1."
    )

    def generate(
        self,
        model,
        clip,
        vae,
        audio_vae,
        pic1,
        scene_count,
        lora_name,
        lora_strength,
        msr_strength,
        reference_frames,
        seed,
        width,
        height,
        duration_sec,
        fps,
        save_each_scene=True,
        scene_prompts="",
        pic2=None,
        pic3=None,
        pic4=None,
        background=None,
        **scene_kwargs,
    ):
        n = max(1, min(int(scene_count), MAX_SCENES))
        prompts = _parse_scene_prompts(scene_prompts, n, **scene_kwargs)

        # MSR requires single-frame refs
        def _one(img):
            if img is None:
                return None
            if img.shape[0] > 1:
                return img[:1]
            return img

        pic1 = _one(pic1)
        pic2 = _one(pic2)
        pic3 = _one(pic3)
        pic4 = _one(pic4)
        background = _one(background)

        model_msr, msr_parameters = _call(
            "ComfyUILTX25MSRICLoRALoader",
            model=model,
            lora_name=lora_name,
            strength_model=float(lora_strength),
        )

        length = int(round(float(duration_sec) * float(fps))) + 1
        if (length - 1) % 8 != 0:
            length = ((length - 1) // 8) * 8 + 1

        all_frames = []
        all_audio = []

        for i in range(n):
            prompt = prompts[i]
            if "Image 1" not in prompt and "image 1" not in prompt.lower():
                prompt = MSR_REF_PREFIX + prompt

            pos = _call("CLIPTextEncode", clip=clip, text=prompt)[0]
            neg = _call("CLIPTextEncode", clip=clip, text=LTX_NEGATIVE)[0]
            pos, neg = _call(
                "LTXVConditioning",
                positive=pos,
                negative=neg,
                frame_rate=float(fps),
            )

            video_latent = _call(
                "EmptyLTXVLatentVideo",
                width=width,
                height=height,
                length=length,
                batch_size=1,
            )[0]
            base_t = int(video_latent["samples"].shape[2])

            guide_kwargs = dict(
                positive=pos,
                negative=neg,
                vae=vae,
                latent=video_latent,
                pic1=pic1,
                strength=float(msr_strength),
                reference_frames=str(reference_frames),
                use_tiled_encode=False,
                tile_size=256,
                tile_overlap=64,
                msr_parameters=msr_parameters,
            )
            if pic2 is not None:
                guide_kwargs["pic2"] = pic2
            if pic3 is not None:
                guide_kwargs["pic3"] = pic3
            if pic4 is not None:
                guide_kwargs["pic4"] = pic4
            if background is not None:
                guide_kwargs["background"] = background

            pos, neg, video_latent = _call(
                "ComfyUILTX25MSRMultiReferenceGuide", **guide_kwargs
            )

            audio_latent = _call(
                "LTXVEmptyLatentAudio",
                frames_number=length,
                frame_rate=float(fps),
                batch_size=1,
                audio_vae=audio_vae,
            )[0]
            av = _call(
                "LTXVConcatAVLatent",
                video_latent=video_latent,
                audio_latent=audio_latent,
            )[0]

            noise = _call("RandomNoise", noise_seed=int(seed) + i)[0]
            sigmas = _call("ManualSigmas", sigmas=DISTILLED_SIGMAS)[0]
            sampler = _call("SamplerEulerAncestral", eta=0.0, s_noise=1.0)[0]
            guider = _call(
                "LTXVDualCFGGuider",
                model=model_msr,
                positive=pos,
                negative=neg,
                video_cfg=1.0,
                audio_cfg=1.0,
            )[0]
            sampled = _call(
                "SamplerCustomAdvanced",
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=av,
            )
            sampled_latent = sampled[1] if len(sampled) > 1 else sampled[0]

            video_lat, audio_lat = _call(
                "LTXVSeparateAVLatent", av_latent=sampled_latent
            )
            _p, _n, video_lat = _call(
                "LTXVCropGuides", positive=pos, negative=neg, latent=video_lat
            )

            vt = video_lat["samples"]
            if vt.shape[2] > base_t:
                video_lat = dict(video_lat)
                video_lat["samples"] = vt[:, :, :base_t].contiguous()
                nm = video_lat.get("noise_mask")
                if nm is not None and nm.shape[2] > base_t:
                    video_lat["noise_mask"] = nm[:, :, :base_t].contiguous()
                print(
                    f"[LTX25MSR] scene {i+1}: trimmed guide slots "
                    f"{vt.shape[2]} -> {base_t}"
                )

            frames = _call(
                "VAEDecodeTiled",
                samples=video_lat,
                vae=vae,
                tile_size=512,
                overlap=64,
                temporal_size=64,
                temporal_overlap=16,
            )[0]
            audio = _call(
                "LTXVAudioVAEDecode", samples=audio_lat, audio_vae=audio_vae
            )[0]

            if frames.shape[0] > length:
                frames = frames[:length].contiguous()
            if isinstance(audio, dict) and "waveform" in audio:
                max_samples = max(
                    1, int(round(frames.shape[0] / float(fps) * audio["sample_rate"]))
                )
                if audio["waveform"].shape[-1] > max_samples:
                    audio = dict(audio)
                    audio["waveform"] = audio["waveform"][..., :max_samples].contiguous()

            video = _call(
                "CreateVideo",
                images=frames,
                fps=float(fps),
                audio=audio,
                bit_depth=8,
            )[0]
            all_frames.append(frames)
            all_audio.append(audio)

            if save_each_scene:
                try:
                    _call(
                        "SaveVideo",
                        video=video,
                        filename_prefix=f"video/scene_{i + 1}_msr",
                        format="auto",
                        codec="auto",
                    )
                except Exception as e:
                    print(f"[LTX25MSR] SaveVideo scene {i+1} skipped: {e}")

        combined_frames = torch.cat(all_frames, dim=0)
        audio = all_audio[0]
        for a in all_audio[1:]:
            audio = _call("AudioConcat", audio1=audio, audio2=a, direction="after")[0]
        combined = _call(
            "CreateVideo",
            images=combined_frames,
            fps=float(fps),
            audio=audio,
            bit_depth=8,
        )[0]
        try:
            _call(
                "SaveVideo",
                video=combined,
                filename_prefix="video/charsheet_msr_combined",
                format="auto",
                codec="auto",
            )
        except Exception as e:
            print(f"[LTX25MSR] SaveVideo combined skipped: {e}")

        return (combined, combined_frames, n)


NODE_CLASS_MAPPINGS = {
    "LTX25MultiSceneMSR": LTX25MultiSceneMSR,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX25MultiSceneMSR": "LTX 2.5 Multi-Scene MSR (Licon LoRA)",
}
