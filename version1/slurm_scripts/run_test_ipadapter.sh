#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/test_ipa_%j.out
#SBATCH -e temp/test_ipa_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project
#SBATCH --time=00:30:00

set -eo pipefail
mkdir -p temp outputs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

echo "Job started at $(date) on $(hostname)"

python -u - << 'PYEOF'
import sys, torch
sys.path.insert(0, 'codes')
from config import DEFAULT_CONFIG
from model import UnifiedModel
from data import load_eeg_dataset
import torch.nn.functional as F

cfg = DEFAULT_CONFIG
device = torch.device('cuda')
print("CUDA:", torch.cuda.get_device_name(0))

# Load model
model = UnifiedModel(cfg, alpha=1.0, beta=1.0).to(device).eval()
state = torch.load('checkpoints/phase2_main_best.pt', map_location=device, weights_only=False)
model.load_state_dict(state['model'], strict=False)

# Get 1 test EEG
test_ds = load_eeg_dataset(data_directory='image-eeg-data', split='test',
                           avg_trials=True, eeg_channel_jsonl='image-eeg-data/EEG_CHANNELS.jsonl')
test_ds = test_ds.with_format('torch')
eeg = test_ds[0]['eeg'].float().unsqueeze(0).to(device)  # [1, 63, 250]
with torch.no_grad():
    emb = model.encode(eeg)  # [1, 1024]
print(f"EEG embedding: shape={tuple(emb.shape)} norm={emb.norm().item():.4f}")

# Load IP-Adapter + SD
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    str(cfg.sd_model_path), torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe.load_ip_adapter(str(cfg.ip_adapter_root), subfolder=cfg.ip_adapter_subfolder,
                     weight_name=cfg.ip_adapter_weight)
pipe.set_ip_adapter_scale(0.7)
pipe.set_progress_bar_config(disable=True)
print("Pipeline loaded")

# Test format 1: [B, num_tokens, embed_dim] with B=1
emb_fp16 = emb.to(dtype=torch.float16)
ip_embeds = [emb_fp16.unsqueeze(1)]  # [1, 1, 1024]
print(f"ip_embeds[0] shape: {ip_embeds[0].shape}")
try:
    out = pipe(prompt="", ip_adapter_image_embeds=ip_embeds,
               guidance_scale=7.5, num_inference_steps=5,
               generator=torch.Generator('cuda').manual_seed(0)).images[0]
    out.save('outputs/test_ipadapter.png')
    print(f"SUCCESS: image saved as outputs/test_ipadapter.png ({out.size})")
except Exception as e:
    print(f"Format 1 failed: {e}")
    # Try format 2: direct embedding
    try:
        out = pipe(prompt="", ip_adapter_image_embeds=[emb_fp16],
                   guidance_scale=7.5, num_inference_steps=5,
                   generator=torch.Generator('cuda').manual_seed(0)).images[0]
        out.save('outputs/test_ipadapter.png')
        print(f"Format 2 SUCCESS")
    except Exception as e2:
        print(f"Format 2 also failed: {e2}")

PYEOF
echo "Job ended at $(date)"
conda deactivate
