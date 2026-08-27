"""Krea CharSheet multi-scene stills — full sheet identity + dynamic scenes.

- character_sheet = FULL reference (face + body + outfit), not a face crop.
- scene_count + web "+ Add scene" / Update: 1–8 scenes (default 3).
- Each scene: still_first, still_last, motion (motion is stored for LTX; stills are generated here).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

import comfy.samplers
import folder_paths
from nodes import SaveImage, VAEDecode, common_ksampler

MAX_SCENES = 8

IDENTITY_SYSTEM = (
    "The attached image is a FULL CHARACTER REFERENCE sheet for ONE person "
    "(face, hair, body proportions, outfit, accessories). Use it like an identity LoRA: "
    "match this exact person and clothing. "
    "CRITICAL OUTPUT RULE: produce ONE new photoreal photograph — single figure, "
    "single camera angle, single environment. "
    "Do NOT copy the sheet layout, white studio backdrop, turnaround, grid, triptych, "
    "front/side/back panels, or multiple poses in one frame."
)

STILL_PREFIX = (
    "Using the full character reference for identity and outfit, generate ONE new "
    "photoreal image of this exact young woman. Single figure, single frame, "
    "single camera — not a character sheet or turnaround. Scene: "
)

DEFAULT_SCENES = {
    1: {
        "first": STILL_PREFIX
        + "the girl standing in a luxury hotel lobby, warm evening light, "
        "three-quarter angle, calm expression, vertical 9:16.",
        "last": STILL_PREFIX
        + "the same girl in that hotel lobby near the sliding glass exit doors, "
        "mid-stride walking toward the exit, camera slightly closer, vertical 9:16.",
        "motion": (
            "Motion and camera only: the girl walks from standing in the hotel lobby "
            "toward the sliding exit doors, natural walking gait, slight camera track, "
            "soft lobby ambience. Keep her face, hair, and outfit fixed to the frames. "
            "No character sheet, collage, or grid."
        ),
    },
    2: {
        "first": STILL_PREFIX
        + "the girl sitting at a rooftop bar at night, city lights bokeh behind her, "
        "holding a glass, medium shot, vertical 9:16.",
        "last": STILL_PREFIX
        + "the same girl at the rooftop bar now standing and turning toward the "
        "city skyline view, vertical 9:16.",
        "motion": (
            "Motion and camera only: at the rooftop bar the girl stands up and turns "
            "toward the city skyline, subtle hand motion with the glass, soft night "
            "ambience. Keep identity fixed. No face/hair/outfit redesign. No sheet."
        ),
    },
    3: {
        "first": STILL_PREFIX
        + "the girl walking through a bright airport terminal with a small suitcase, "
        "medium-wide shot, vertical 9:16.",
        "last": STILL_PREFIX
        + "the same girl in the airport terminal nearer the departure gate screens, "
        "still walking forward, vertical 9:16.",
        "motion": (
            "Motion and camera only: the girl walks through the airport terminal toward "
            "the gate screens, suitcase rolling beside her, camera tracks alongside, "
            "terminal ambience. Keep identity fixed. No character sheet or collage."
        ),
    },
}


def _load_krea_encoder():
    import importlib.util

    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "comfyui-krea2edit" / "__init__.py",
        here.parents[2] / "ComfyUI" / "custom_nodes" / "comfyui-krea2edit" / "__init__.py",
        Path(r"D:\Comfyui\ComfyUI\custom_nodes\comfyui-krea2edit\__init__.py"),
        Path(r"D:\Comfyui\custom_nodes\comfyui-krea2edit\__init__.py"),
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise RuntimeError(
            "comfyui-krea2edit is required. Install it under custom_nodes."
        )
    spec = importlib.util.spec_from_file_location("comfyui_krea2edit_runtime", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Krea2EditGroundedEncode()


def _save_image_tensor(image_tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    img.save(path)


def _default(scene_i: int, key: str) -> str:
    return DEFAULT_SCENES.get(scene_i, {}).get(key, "")


class KreaCharSheetScenes:
    """Full character-sheet identity → dynamic first/last stills per scene."""

    @classmethod
    def INPUT_TYPES(cls):
        scenes = {}
        for i in range(1, MAX_SCENES + 1):
            scenes[f"scene_{i}_first"] = (
                "STRING",
                {
                    "multiline": True,
                    "default": _default(i, "first"),
                    "dynamicPrompts": False,
                },
            )
            scenes[f"scene_{i}_last"] = (
                "STRING",
                {
                    "multiline": True,
                    "default": _default(i, "last"),
                    "dynamicPrompts": False,
                },
            )
            scenes[f"scene_{i}_motion"] = (
                "STRING",
                {
                    "multiline": True,
                    "default": _default(i, "motion"),
                    "dynamicPrompts": False,
                },
            )
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "character_sheet": ("IMAGE",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "scene_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": MAX_SCENES,
                        "step": 1,
                        "tooltip": "How many scenes to generate. Use + Add scene, then fill first/last/motion for each.",
                    },
                ),
                **scenes,
                "seed": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 12, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grounding_px": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 64}),
                "overwrite": ("BOOLEAN", {"default": True}),
                "identity_system_prompt": (
                    "STRING",
                    {"multiline": True, "default": IDENTITY_SYSTEM},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "motion_prompts", "scene_count")
    FUNCTION = "generate"
    CATEGORY = "multishot"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Full character sheet → single-figure scene stills (first/last per scene). "
        "Set scene_count (default 3); use + Add scene to unlock more (max 8). "
        "Each scene needs still_first, still_last, and motion (for LTX)."
    )

    def generate(
        self,
        model,
        clip,
        vae,
        character_sheet,
        negative,
        latent_image,
        scene_count,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        grounding_px,
        overwrite=True,
        identity_system_prompt=IDENTITY_SYSTEM,
        **scene_kwargs,
    ):
        n = max(1, min(int(scene_count), MAX_SCENES))
        encoder = _load_krea_encoder()
        decoder = VAEDecode()
        images_out = []
        motions = []
        input_dir = Path(folder_paths.get_input_directory())
        sys_prompt = (identity_system_prompt or IDENTITY_SYSTEM).strip()
        gen_i = 0

        for scene_i in range(1, n + 1):
            first = (scene_kwargs.get(f"scene_{scene_i}_first") or "").strip()
            last = (scene_kwargs.get(f"scene_{scene_i}_last") or "").strip()
            motion = (scene_kwargs.get(f"scene_{scene_i}_motion") or "").strip()
            if not first and not last:
                raise ValueError(
                    f"Scene {scene_i}: fill scene_{scene_i}_first and/or scene_{scene_i}_last."
                )
            if not first:
                first = last
            if not last:
                last = first
            motions.append(motion or f"Motion only for scene {scene_i}.")

            for tag, prompt in (("first", first), ("last", last)):
                positive = encoder.encode(
                    clip,
                    prompt,
                    image=character_sheet,
                    grounding_px=grounding_px,
                    system_prompt=sys_prompt,
                )[0]
                sampled = common_ksampler(
                    model,
                    seed + gen_i,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent_image,
                    denoise=denoise,
                )[0]
                image = decoder.decode(vae, sampled)[0]
                images_out.append(image)
                path = input_dir / f"scene_{scene_i}_{tag}.png"
                if overwrite or not path.is_file():
                    _save_image_tensor(image[0], path)
                gen_i += 1

        motion_blob = "\n---\n".join(
            f"[scene_{i + 1}]\n{m}" for i, m in enumerate(motions)
        )
        return (torch.cat(images_out, dim=0), motion_blob, n)


# Backward-compatible alias used by older workflows
class KreaMultiShotStills(KreaCharSheetScenes):
    """Alias — prefer KreaCharSheetScenes (full sheet + dynamic scenes)."""

    pass


SHEET_SYSTEM = (
    "The attached image is a PHOTO of ONE real person. "
    "Preserve the exact facial identity of this person — same face geometry, eyes, nose, "
    "mouth, skin tone, age, and hair. "
    "Restyle ONLY the composition into a new lookbook character sheet. "
    "Do not invent a different person."
)

SHEET_PROMPT = (
    "Preserve the exact facial identity of the person in the attached photo "
    "(same face, eyes, nose, lips, skin, hair length and style, age). "
    "Keep the same outfit and accessories from the photo in every view.\n\n"
    "Edit the image into a professional character reference sheet, commercial lookbook "
    "photography, clean pure white seamless studio background in EVERY panel, "
    "even soft high-key lighting, no outdoor/park/cemetery background.\n\n"
    "Exactly FOUR equal panels side-by-side in ONE wide horizontal image, "
    "no gaps, labels, or borders:\n"
    "1) Head-and-shoulders close-up facing camera on white studio (not the original outdoor photo)\n"
    "2) Full-body front standing straight, arms at sides, feet visible, white studio\n"
    "3) Full-body right-side profile standing straight, feet visible, white studio\n"
    "4) Full-body back view facing away, feet visible, white studio\n\n"
    "Same exact person and same outfit in all four panels. Photoreal, natural skin texture, "
    "sharp focus. No text, watermarks, collage frames, beauty filter, or different faces."
)

SHEET_NEGATIVE = (
    "different person, different face, generic face, beauty filter face, "
    "single portrait only, copy of source photo background, outdoor background, "
    "park, cemetery, busy background, inconsistent identity between panels, "
    "different hair, outfit change between views, colored backdrop, plastic skin, "
    "extra limbs, cropped feet, text, watermark, unequal panels, anime, illustration, CGI"
)


class KreaPhotoToCharSheet:
    """One photo → one 4-view character sheet (for MSR / identity pipeline)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "photo": ("IMAGE",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 60}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Higher = stronger face likeness (krea2edit: try 768–1024 for people).
                "grounding_px": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 64}),
                "sheet_prompt": ("STRING", {"multiline": True, "default": SHEET_PROMPT}),
                "system_prompt": ("STRING", {"multiline": True, "default": SHEET_SYSTEM}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("character_sheet",)
    FUNCTION = "generate"
    CATEGORY = "multishot"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "One photo only → Krea 4-view lookbook sheet with identity lock "
        "(use with Identity Edit LoRA + ModelPatch). Shows on CHARACTER SHEET OUTPUT."
    )

    def generate(
        self,
        model,
        clip,
        vae,
        photo,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        grounding_px,
        sheet_prompt,
        system_prompt,
    ):
        encoder = _load_krea_encoder()
        decoder = VAEDecode()
        positive = encoder.encode(
            clip,
            (sheet_prompt or SHEET_PROMPT).strip(),
            image=photo,
            grounding_px=grounding_px,
            system_prompt=(system_prompt or SHEET_SYSTEM).strip(),
        )[0]
        sampled = common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )[0]
        image = decoder.decode(vae, sampled)[0]
        saver = SaveImage()
        saved = saver.save_images(image, filename_prefix="character_sheet")
        print(f"[KreaPhotoToCharSheet] UI images: {saved.get('ui', {}).get('images')}")
        return {"ui": saved.get("ui", {}), "result": (image,)}


NODE_CLASS_MAPPINGS = {
    "KreaCharSheetScenes": KreaCharSheetScenes,
    "KreaMultiShotStills": KreaMultiShotStills,
    "KreaPhotoToCharSheet": KreaPhotoToCharSheet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KreaCharSheetScenes": "Krea CharSheet Scenes (+ Add scene)",
    "KreaMultiShotStills": "Krea Multi-Shot Stills (legacy alias)",
    "KreaPhotoToCharSheet": "Krea Photo → Character Sheet",
}
