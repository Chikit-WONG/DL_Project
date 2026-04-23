import torch
import numpy as np
import os

DATA_ROOT = "/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data"

def inspect(path, name):
    print(f"\n{'='*60}")
    print(f"Inspecting: {name}")
    data = torch.load(path, weights_only=False)
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor shape={tuple(v.shape)}, dtype={v.dtype}")
            elif isinstance(v, np.ndarray):
                print(f"  {k}: ndarray shape={v.shape}, dtype={v.dtype}")
                if v.size > 0:
                    flat = v.flat
                    sample = [next(flat) for _ in range(min(3, v.size))]
                    print(f"    sample values: {sample}")
            else:
                t = type(v).__name__
                try:
                    print(f"  {k}: {t}, len={len(v)}")
                    if hasattr(v, '__getitem__') and len(v) > 0:
                        print(f"    first elem: {v[0]!r}")
                except Exception:
                    print(f"  {k}: {t}")

inspect(os.path.join(DATA_ROOT, "train.pt"), "train.pt")
inspect(os.path.join(DATA_ROOT, "test.pt"), "test.pt")

print("\nImage directories:")
for split in ["training_images", "test_images"]:
    p = os.path.join(DATA_ROOT, split)
    cats = sorted(os.listdir(p)) if os.path.isdir(p) else []
    print(f"  {split}/: {len(cats)} folders, first 3: {cats[:3]}")
    if cats:
        first_cat = os.path.join(p, cats[0])
        imgs = sorted(os.listdir(first_cat))
        print(f"    {cats[0]}/: {imgs[:3]}")
