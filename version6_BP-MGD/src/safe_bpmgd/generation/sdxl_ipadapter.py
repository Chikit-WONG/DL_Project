from __future__ import annotations

from pathlib import Path

import torch


def build_sdxl_ipadapter_pipeline(cfg, device: str):
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        cfg["paths"]["sdxl_turbo"],
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        variant="fp16" if "cuda" in device else None,
    )
    adapter_root = Path(cfg["paths"]["ip_adapter_root"])
    # The local SDXL image encoder projects to 1280 dims, so it must pair with
    # the 1280-dim adapter weights instead of the 1024-dim vit-h variant.
    pipe.load_ip_adapter(str(adapter_root), subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors")
    pipe.set_ip_adapter_scale(float(cfg["generation"].get("ip_adapter_scale", 0.75)))
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
