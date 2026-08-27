"""LTX 2.5 multi-scene FLF2V — fully dynamic loop over Krea still batches.

Takes interleaved first/last stills (2 * scene_count) + motion_prompts blob
from KreaCharSheetScenes and generates one video per scene, then a combined
video with audio. No muted graph clones.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

import folder_paths

DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

LTX_NEGATIVE = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, unnatural skin tones, deformed facial features, asymmetrical face, "
    "missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts, "
    "camera shake, background clutter, harsh shadows, cartoonish rendering, 3D CGI look, "
    "uncanny valley effect, silent or muted audio, distorted voice, robotic voice, "
    "jittery movement, unnatural transitions, AI artifacts, character sheet, turnaround, "
    "multiple poses, collage, grid, triptych"
)


def _nodes():
    import nodes

    return nodes.NODE_CLASS_MAPPINGS


def _unwrap(out):
    """Normalize classic tuple returns and V3 NodeOutput to a tuple."""
    if out is None:
        return ()
    if hasattr(out, "args") and isinstance(getattr(out, "args"), tuple):
        return out.args
    if isinstance(out, dict) and "result" in out:
        r = out["result"]
        return r if isinstance(r, tuple) else (r,)
    if isinstance(out, tuple):
        return out
    return (out,)


def _call(class_name: str, **kwargs):
    mapping = _nodes()
    if class_name not in mapping:
        raise RuntimeError(f"Missing Comfy node class: {class_name}")
    cls = mapping[class_name]
    # V3 nodes expose EXECUTE_NORMALIZED as a classmethod; classic nodes use instance.FUNCTION
    func_name = cls.FUNCTION
    fn = getattr(cls, func_name, None)
    if callable(fn) and not isinstance(fn, property):
        try:
            out = fn(**kwargs)
            return _unwrap(out)
        except TypeError:
            pass
    obj = cls()
    out = getattr(obj, func_name)(**kwargs)
    return _unwrap(out)


def _parse_motions(motion_prompts: str, scene_count: int) -> list[str]:
    text = (motion_prompts or "").strip()
    parts = re.split(r"\n---\n", text) if text else []
    motions = []
    for i in range(scene_count):
        if i < len(parts):
            block = parts[i]
            block = re.sub(r"^\[scene_\d+\]\s*", "", block.strip(), flags=re.I)
            motions.append(block.strip() or f"Motion and camera only for scene {i + 1}.")
        else:
            motions.append(
                f"Motion and camera only: natural movement between the two frames for scene {i + 1}, "
                "audible matching ambience. Keep identity fixed."
            )
    return motions


def _resize_image(image, width: int, height: int):
    # image: [B,H,W,C]
    try:
        out = _call(
            "ImageScale",
            image=image,
            upscale_method="lanczos",
            width=width,
            height=height,
            crop="center",
        )
        return out[0]
    except Exception:
        samples = image.movedim(-1, 1)
        import comfy.utils

        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
        return samples.movedim(1, -1)


def _run_one_flf2v(
    *,
    model,
    clip,
    vae,
    audio_vae,
    first,
    last,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    length: int,
    fps: float,
    guide_strength: float,
    img_compression: int,
):
    first = _resize_image(first, width, height)
    last = _resize_image(last, width, height)
    first = _call("LTXVPreprocess", image=first, img_compression=img_compression)[0]
    last = _call("LTXVPreprocess", image=last, img_compression=img_compression)[0]

    pos = _call("CLIPTextEncode", clip=clip, text=prompt)[0]
    neg = _call("CLIPTextEncode", clip=clip, text=LTX_NEGATIVE)[0]
    pos, neg = _call("LTXVConditioning", positive=pos, negative=neg, frame_rate=float(fps))

    # Official FLF2V order: AddGuide on VIDEO latent only, THEN ConcatAV.
    # Guiding a NestedTensor AV latent raises: expected Tensor, got NestedTensor.
    video_latent = _call(
        "EmptyLTXVLatentVideo",
        width=width,
        height=height,
        length=length,
        batch_size=1,
    )[0]
    # Guide frames are APPENDED to the latent; they must be cropped before decode
    # or they appear as frozen still frames at the end of the clip.
    base_t = int(video_latent["samples"].shape[2])

    pos, neg, video_latent = _call(
        "LTXVAddGuide",
        positive=pos,
        negative=neg,
        vae=vae,
        latent=video_latent,
        image=first,
        frame_idx=0,
        strength=guide_strength,
    )
    pos, neg, video_latent = _call(
        "LTXVAddGuide",
        positive=pos,
        negative=neg,
        vae=vae,
        latent=video_latent,
        image=last,
        frame_idx=-1,
        strength=guide_strength,
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

    noise = _call("RandomNoise", noise_seed=seed)[0]
    sigmas = _call("ManualSigmas", sigmas=DISTILLED_SIGMAS)[0]
    sampler = _call("SamplerEulerAncestral", eta=0.0, s_noise=1.0)[0]
    guider = _call(
        "LTXVDualCFGGuider",
        model=model,
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
    # Gallery FLF uses denoised output (slot 1); fall back to slot 0 if absent.
    sampled_latent = sampled[1] if len(sampled) > 1 else sampled[0]

    # Separate first, then CropGuides on the VIDEO stream only (matches gallery).
    video_lat, audio_lat = _call("LTXVSeparateAVLatent", av_latent=sampled_latent)
    _pos2, _neg2, video_lat = _call(
        "LTXVCropGuides", positive=pos, negative=neg, latent=video_lat
    )

    # Hard guarantee: never decode the appended guide stills.
    vt = video_lat["samples"]
    if vt.shape[2] != base_t:
        video_lat = dict(video_lat)
        video_lat["samples"] = vt[:, :, :base_t].contiguous()
        nm = video_lat.get("noise_mask")
        if nm is not None and getattr(nm, "shape", None) is not None and nm.shape[2] > base_t:
            video_lat["noise_mask"] = nm[:, :, :base_t].contiguous()
        print(
            f"[LTX25MultiScene] trimmed video latent {tuple(vt.shape)} -> "
            f"{tuple(video_lat['samples'].shape)} (removed guide still frames)"
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
    audio = _call("LTXVAudioVAEDecode", samples=audio_lat, audio_vae=audio_vae)[0]

    # Decode can overshoot; clamp to requested pixel length so stills aren't held.
    if frames.shape[0] > length:
        frames = frames[:length].contiguous()
    if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
        max_samples = max(1, int(round(frames.shape[0] / float(fps) * audio["sample_rate"])))
        wf = audio["waveform"]
        if wf.shape[-1] > max_samples:
            audio = dict(audio)
            audio["waveform"] = wf[..., :max_samples].contiguous()

    video = _call("CreateVideo", images=frames, fps=float(fps), audio=audio, bit_depth=8)[0]
    return video, frames, audio


class LTX25MultiSceneFLF2V:
    """Animate N Krea scene pairs with LTX 2.5 FLF2V in one node (fully dynamic)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "images": ("IMAGE",),
                "scene_count": ("INT", {"default": 3, "min": 1, "max": 8}),
                "motion_prompts": ("STRING", {"multiline": True, "default": ""}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 32}),
                "duration_sec": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "guide_strength": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
                "img_compression": ("INT", {"default": 18, "min": 0, "max": 100}),
                "save_each_scene": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "INT")
    RETURN_NAMES = ("combined_video", "all_frames", "scene_count")
    FUNCTION = "generate"
    CATEGORY = "multishot"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Fully dynamic LTX 2.5 FLF2V: scene_count still-pairs → that many videos → "
        "one combined video with audio. Wire from KreaCharSheetScenes."
    )

    def generate(
        self,
        model,
        clip,
        vae,
        audio_vae,
        images,
        scene_count,
        motion_prompts,
        seed,
        width,
        height,
        duration_sec,
        fps,
        guide_strength,
        img_compression,
        save_each_scene=True,
    ):
        n = max(1, min(int(scene_count), 8))
        need = n * 2
        if images.shape[0] < need:
            raise ValueError(
                f"Need {need} stills (first/last × {n} scenes), got batch of {images.shape[0]}. "
                "Increase scene_count on Krea CharSheet Scenes and re-queue."
            )

        length = int(round(duration_sec * fps)) + 1
        # LTX prefers 8k+1
        if (length - 1) % 8 != 0:
            length = ((length - 1) // 8) * 8 + 1

        motions = _parse_motions(motion_prompts, n)
        out_dir = Path(folder_paths.get_output_directory()) / "video"
        out_dir.mkdir(parents=True, exist_ok=True)

        videos = []
        all_frames = []
        all_audio = []

        for i in range(n):
            first = images[i * 2 : i * 2 + 1]
            last = images[i * 2 + 1 : i * 2 + 2]
            video, frames, audio = _run_one_flf2v(
                model=model,
                clip=clip,
                vae=vae,
                audio_vae=audio_vae,
                first=first,
                last=last,
                prompt=motions[i],
                seed=int(seed) + i,
                width=width,
                height=height,
                length=length,
                fps=fps,
                guide_strength=guide_strength,
                img_compression=img_compression,
            )
            videos.append(video)
            all_frames.append(frames)
            all_audio.append(audio)

            if save_each_scene:
                try:
                    _call(
                        "SaveVideo",
                        video=video,
                        filename_prefix=f"video/scene_{i + 1}_flf2v",
                        format="auto",
                        codec="auto",
                    )
                except Exception as e:
                    print(f"[LTX25MultiScene] SaveVideo scene {i+1} skipped: {e}")

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
                filename_prefix="video/charsheet_combined_final",
                format="auto",
                codec="auto",
            )
        except Exception as e:
            print(f"[LTX25MultiScene] SaveVideo combined skipped: {e}")

        return (combined, combined_frames, n)


NODE_CLASS_MAPPINGS = {
    "LTX25MultiSceneFLF2V": LTX25MultiSceneFLF2V,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX25MultiSceneFLF2V": "LTX 2.5 Multi-Scene FLF2V (dynamic)",
}
