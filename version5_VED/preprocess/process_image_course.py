"""
Generate multi-blur CLIP (RN50) image features for course data.
Adapted from process_image.py: Linux paths, course image dirs.
"""
import torch
import cv2
from PIL import Image
import numpy as np
import torch.nn.functional as F
import os
import argparse
from torchvision import transforms
import open_clip
import torch.nn as nn
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.path.join(REPO_DIR, "data", "things-eeg", "Image_set")
DEFAULT_RN50_WEIGHTS = os.environ.get(
    "VED_RN50_WEIGHTS",
    os.path.join(REPO_DIR, "data", "weights", "open_clip_pytorch_model.bin"),
)
DEFAULT_SAVE_DIR = os.path.join(REPO_DIR, "output", "Image_feature")


class BlurringPipeline:
    def __init__(self, blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_blur = cv2.GaussianBlur(img_np, (self.blur_kernel_size, self.blur_kernel_size), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_blur)


class Make_dataset(nn.Module):
    def __init__(self, checkpoint_path, batch_size=128):
        super().__init__()
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                "Missing local RN50 weights. Download the checkpoint and pass "
                f"--clip_checkpoint, or set VED_RN50_WEIGHTS. Tried: {checkpoint_path}"
            )

        self.batch_size = batch_size
        # Use a local checkpoint so runs do not depend on external network access.
        self.vlmodel, _, _ = open_clip.create_model_and_transforms(
            'RN50',
            pretrained=checkpoint_path
        )
        self.vlmodel.eval()

        self.blur_transform = {}
        for kernel, tag in zip(
            [1, 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63],
            ['1', '3', '9', '15', '21', '27', '33', '39', '45', '51', '57', '63']
        ):
            self.blur_transform[tag] = BlurringPipeline(kernel)

        process_terms = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ]
        self.process_transform = transforms.Compose(process_terms)

    @torch.no_grad()
    def ImageEncoder(self, image_paths, blur_transform):
        self.vlmodel.eval()
        image_features_list = []
        dev = next(self.vlmodel.parameters()).device

        for i in tqdm.tqdm(range(0, len(image_paths), self.batch_size)):
            batch = image_paths[i:i + self.batch_size]
            imgs = []
            for rel_path in batch:
                full_path = os.path.join(BASE_PATH, rel_path)
                img = Image.open(full_path).convert("RGB").resize((224, 224))
                imgs.append(self.process_transform(blur_transform(img)))
            inputs = torch.stack(imgs).to(dev)
            feats = self.vlmodel.encode_image(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            image_features_list.append(feats.float().cpu())

        image_features = torch.cat(image_features_list, dim=0)
        return {image_paths[i]: image_features[i] for i in range(len(image_paths))}


def collect_image_paths(split_dir):
    """Return sorted relative paths from Image_set/ for images in split_dir."""
    rel_paths = []
    split_full = os.path.join(BASE_PATH, split_dir)
    for cat in sorted(os.listdir(split_full)):
        cat_dir = os.path.join(split_full, cat)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                rel_paths.append(os.path.join(split_dir, cat, fname))
    return rel_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_checkpoint",
        default=DEFAULT_RN50_WEIGHTS,
        help="Path to OpenCLIP RN50 checkpoint, usually open_clip_pytorch_model.bin.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--save_dir",
        default=DEFAULT_SAVE_DIR,
        help="Directory for generated MultiBlur_RN50_train/test.pt files.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train_save = os.path.join(args.save_dir, "MultiBlur_RN50_train.pt")
    test_save = os.path.join(args.save_dir, "MultiBlur_RN50_test.pt")

    print(f"Using RN50 checkpoint: {args.clip_checkpoint}")
    print(f"Saving image features to: {args.save_dir}")
    model = Make_dataset(args.clip_checkpoint, args.batch_size).to(device)

    blur_keys = ['1', '3', '9', '15', '21', '27', '33', '39', '45', '51', '57', '63']

    # --- Training features ---
    if os.path.exists(train_save):
        print(f"Train features already exist: {train_save}")
    else:
        print("Collecting training image paths...")
        train_paths = collect_image_paths("train_images")
        print(f"Found {len(train_paths)} training images")
        train_features = {}
        for key in blur_keys:
            print(f"Encoding blur level {key}...")
            train_features[key] = model.ImageEncoder(train_paths, model.blur_transform[key])
        torch.save(train_features, train_save)
        print(f"Saved: {train_save}")

    # --- Test features ---
    if os.path.exists(test_save):
        print(f"Test features already exist: {test_save}")
    else:
        print("Collecting test image paths...")
        test_paths = collect_image_paths("test_images")
        print(f"Found {len(test_paths)} test images")
        test_features = {}
        for key in blur_keys:
            print(f"Encoding blur level {key}...")
            test_features[key] = model.ImageEncoder(test_paths, model.blur_transform[key])
        torch.save(test_features, test_save)
        print(f"Saved: {test_save}")

    print("All done.")
