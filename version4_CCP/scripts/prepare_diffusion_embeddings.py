from __future__ import annotations

import argparse

from src.cogcappro.generate_image.generator import IPAdapterGenerator, prepare_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CogCapPro diffusion embeddings from the adapted dataset.")
    parser.add_argument("--config", type=str, default="configs/cogcappro.yaml")
    parser.add_argument("--data-type", type=str, default="EEG", choices=["EEG", "MEG"])
    parser.add_argument("--sd-path", type=str, default=None)
    parser.add_argument("--ip-adapter-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    generator = IPAdapterGenerator(
        sd_path=args.sd_path,
        ip_adapter_path=args.ip_adapter_path,
        device=args.device,
        seed=42,
        num_inference_steps=5,
        guidance_scale=0.0,
        modalities=["image", "depth", "edge"],
    )
    prepare_embedding(
        generator=generator,
        modalities=["image", "depth", "edge"],
        config_path=args.config,
        data_type=args.data_type,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
