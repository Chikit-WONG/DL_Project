#!/bin/bash
#SBATCH -p debug
#SBATCH -o temp/test_ipa2_%j.out
#SBATCH -e temp/test_ipa2_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project
#SBATCH --time=00:30:00

set -eo pipefail
mkdir -p temp outputs

source /hpc2hdd/home/ckwong627/miniconda3/etc/profile.d/conda.sh
conda activate test
module load cuda/12.6

python -u - << 'PYEOF'
import sys, torch, inspect
sys.path.insert(0, 'codes')
from config import DEFAULT_CONFIG
from model import UnifiedModel
from data import load_eeg_dataset
from diffusers import StableDiffusionPipeline

cfg = DEFAULT_CONFIG
device = torch.device('cuda')

model = UnifiedModel(cfg, alpha=1.0, beta=1.0).to(device).eval()
state = torch.load('checkpoints/phase2_main_best.pt', map_location=device, weights_only=False)
model.load_state_dict(state['model'], strict=False)

test_ds = load_eeg_dataset(data_directory='image-eeg-data', split='test',
    avg_trials=True, eeg_channel_jsonl='image-eeg-data/EEG_CHANNELS.jsonl').with_format('torch')
eeg = test_ds[0]['eeg'].float().unsqueeze(0).to(device)
with torch.no_grad():
    emb = model.encode(eeg)  # [1, 1024]
emb_fp16 = emb.to(dtype=torch.float16)
zeros = torch.zeros_like(emb_fp16)

pipe = StableDiffusionPipeline.from_pretrained(
    str(cfg.sd_model_path), torch_dtype=torch.float16, safety_checker=None
).to(device)
pipe.load_ip_adapter(str(cfg.ip_adapter_root), subfolder=cfg.ip_adapter_subfolder,
                     weight_name=cfg.ip_adapter_weight)
pipe.set_ip_adapter_scale(0.7)
pipe.set_progress_bar_config(disable=True)

# Try different formats with guidance_scale=1 (no CFG) to find the right shape
formats = {
    'no_cfg_3d': (1.0, [emb_fp16.unsqueeze(1)]),       # [1,1,1024]
    'no_cfg_2d_batch': (1.0, [emb_fp16]),               # [1,1024]
    'cfg_concat_3d': (7.5, [torch.cat([zeros,emb_fp16]).unsqueeze(1)]),  # [2,1,1024] — original plan
    'no_cfg_4d': (1.0, [emb_fp16.unsqueeze(0).unsqueeze(0)]),  # [1,1,1,1024]
}
for name, (gs, ip_e) in formats.items():
    try:
        out = pipe(prompt="", ip_adapter_image_embeds=ip_e,
                   guidance_scale=gs, num_inference_steps=5,
                   height=512, width=512,
                   generator=torch.Generator('cuda').manual_seed(0)).images[0]
        out.save(f'outputs/test_ipa_{name}.png')
        print(f"OK  {name}  gs={gs}  emb shape {ip_e[0].shape}")
    except Exception as e:
        print(f"FAIL {name}: {e}")
PYEOF
echo "done"
