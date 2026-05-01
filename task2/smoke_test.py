#!/usr/bin/env python3
"""Smoke test: verify imports and key external path resolution (no GPU required)."""
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

PASSED = []
FAILED = []

def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {e}")

print("=" * 60)
print("CognitionCapturerPro Smoke Test")
print("=" * 60)

print("\n[1] Python imports")
def check_imports():
    import src.cogcappro.data.eeg
    import src.cogcappro.models.brain_backbone
    import src.cogcappro.models.fusion_backbone
    import src.cogcappro.training.module
    import src.cogcappro.runtime.paths
    import src.cogcappro.align.data
    import src.cogcappro.align.diffusion_pipe
    import src.cogcappro.generate_image.generator
check("All module imports", check_imports)

print("\n[2] Config loading")
def check_configs():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(REPO, "configs/cogcappro.yaml"))
    local = OmegaConf.load(os.path.join(REPO, "configs/local.yaml"))
    merged = OmegaConf.merge(cfg, local)
    assert hasattr(merged, "paths"), "merged config missing 'paths'"
check("Config loading (cogcappro.yaml + local.yaml)", check_configs)

print("\n[3] External path resolution")
from omegaconf import OmegaConf
local = OmegaConf.load(os.path.join(REPO, "configs/local.yaml"))
paths_to_check = {
    "data_root": local.paths.data_root,
    "weights_root": local.paths.weights_root,
    "sdxl_root": local.paths.sdxl_root,
    "ip_adapter_root": local.paths.ip_adapter_root,
    "runs_root": local.paths.runs_root,
}
for name, path in paths_to_check.items():
    check(f"{name} = {path}", lambda p=path: (_ for _ in ()).throw(FileNotFoundError(p)) if not os.path.exists(p) else None)

print("\n[4] Result files")
result_files = [
    "results/reconstruction_metrics_all_before.json",
    "results/reconstruction_metrics_all.json",
    "results/retrieval_test_results.json",
    "results/summary_metrics.json",
]
for rel in result_files:
    full = os.path.join(REPO, rel)
    check(rel, lambda p=full: (_ for _ in ()).throw(FileNotFoundError(p)) if not os.path.exists(p) else None)

print("\n[5] Output images")
comparison_dir = os.path.join(REPO, "outputs/comparison")
check("comparison dir exists", lambda: (_ for _ in ()).throw(FileNotFoundError(comparison_dir)) if not os.path.exists(comparison_dir) else None)
n_gen = len([f for f in os.listdir(os.path.join(REPO, "outputs/generated_all_before")) if f.endswith(".jpg")])
check(f"generated_all_before has {n_gen} images (expect 200)", lambda: (_ for _ in ()).throw(AssertionError(f"got {n_gen}")) if n_gen != 200 else None)

print("\n" + "=" * 60)
print(f"Results: {len(PASSED)} passed, {len(FAILED)} failed")
print("=" * 60)
sys.exit(1 if FAILED else 0)
