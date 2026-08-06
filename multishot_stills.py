from pathlib import Path

import numpy as np
import torch
from PIL import Image

import comfy.samplers
import folder_paths
from nodes import VAEDecode, common_ksampler

MAX_SHOTS = 8


def _load_krea_encoder():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "comfyui-krea2edit" / "__init__.py"
    if not path.is_file():
        raise RuntimeError("comfyui-krea2edit is required for Krea Multi-Shot Stills.")
    spec = importlib.util.spec_from_file_location("comfyui_krea2edit_runtime", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Krea2EditGroundedEncode()


def _save_image_tensor(image_tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    img.save(path)


class KreaMultiShotStills:
    """Generate Krea stills from image prompts only. Does not touch LTX Director."""

    @classmethod
    def INPUT_TYPES(cls):
        shots = {}
        defaults = {
            "shot_1": "16:9 widescreen close-up of the same woman at a messy desk, black hoodie, ring light, looking at camera, hands out of frame, photorealistic, exact face identity.",
            "shot_2": "16:9 widescreen medium shot of the same woman on a sunny balcony, white sundress, city skyline behind her, golden light, photorealistic, exact face identity.",
            "shot_3": "16:9 widescreen cafe booth, same woman in a red blazer over a cream blouse, latte on the table, warm window light, photorealistic, exact face identity.",
            "shot_4": "16:9 widescreen three-quarter night street, same woman in a brown leather jacket, neon signs, looking toward camera, photorealistic, exact face identity.",
            "shot_5": "16:9 widescreen kitchen counter, same woman in a mustard yellow oversized tee, phone on a stand, daylight, looking at camera, photorealistic, exact face identity.",
        }
        for i in range(1, MAX_SHOTS + 1):
            key = f"shot_{i}"
            shots[key] = (
                "STRING",
                {
                    "multiline": True,
                    "default": defaults.get(key, ""),
                    "dynamicPrompts": False,
                },
            )
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                **shots,
                "filename_prefix": ("STRING", {"default": "multishot"}),
                "seed": ("INT", {"default": 1000, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "shot_count")
    FUNCTION = "generate"
    CATEGORY = "multishot"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Generate one Krea still per filled shot prompt and save input/<prefix>_N.png. "
        "Leave a shot blank to skip it. LTX Director is not modified — drop stills there yourself."
    )

    def generate(
        self,
        model,
        clip,
        vae,
        reference_image,
        negative,
        latent_image,
        filename_prefix,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        grounding_px,
        overwrite=True,
        **shot_kwargs,
    ):
        prompts = []
        for i in range(1, MAX_SHOTS + 1):
            text = (shot_kwargs.get(f"shot_{i}") or "").strip()
            if text:
                prompts.append((i, text))
        if not prompts:
            raise ValueError("Enter at least one shot image prompt.")

        encoder = _load_krea_encoder()
        decoder = VAEDecode()
        images_out = []
        prefix = (filename_prefix or "multishot").strip() or "multishot"
        input_dir = Path(folder_paths.get_input_directory())

        for gen_i, (shot_num, prompt) in enumerate(prompts):
            positive = encoder.encode(
                clip, prompt, image=reference_image, grounding_px=grounding_px
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

            path = input_dir / f"{prefix}_{shot_num}.png"
            if overwrite or not path.is_file():
                _save_image_tensor(image[0], path)

        return (torch.cat(images_out, dim=0), len(prompts))
