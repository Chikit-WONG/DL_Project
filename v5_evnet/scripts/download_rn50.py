"""
Download OpenCLIP RN50 (openai pretrained) weights and save as open_clip_pytorch_model.bin
"""
import argparse
import open_clip
import torch
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default=os.path.join(REPO_DIR, "data", "weights"),
        help="Directory to save open_clip_pytorch_model.bin.",
    )
    args = parser.parse_args()

    save_path = os.path.join(args.save_dir, "open_clip_pytorch_model.bin")

    if os.path.exists(save_path):
        print(f"Already exists: {save_path}")
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        print("Downloading OpenCLIP RN50 (openai pretrained)...")
        model, _, _ = open_clip.create_model_and_transforms('RN50', pretrained='openai')
        model.eval()
        torch.save(model.state_dict(), save_path)
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"Saved to {save_path} ({size_mb:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
