# Version 2 实施计划：基于共享数据目录的 THINGS-EEG 检索与重建升级

## Summary
- 以 `version1` 为可运行基线，不修改 `version1` 的结果目录与行为，在 `version2` 下独立实现升级版流水线。
- 数据目录固定为：
  - `/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data`
- 主线方案固定为：`增强 EEG encoder + 多视觉目标对齐 + Prior Diffusion + 单路 IP-Adapter + img2img + SDXL-Turbo`。
- 交付物包括：代码、SLURM 脚本、日志、结果、Task 2 拼图，以及中英文计划与中英文结果汇总。

## Key Changes
- `version2/codes/config.py` 作为唯一配置源，硬编码共享数据目录。
- `version2` 不复制数据集，所有训练、评估、重建都直接读取共享数据目录。
- `version2/codes` 提供以下核心脚本：
  - `config.py`
  - `data.py`
  - `cache_backbone_features.py`
  - `model.py`
  - `train_encoder.py`
  - `train_prior.py`
  - `reconstruct.py`
  - `evaluate.py`
  - `make_task2_montage.py`
  - `summarize_results.py`
- 模型固定采用：
  - `Semantic head`：主对齐 `CLIP ViT-H/14`，辅对齐 `ViT-B/32` 与 `RN50`
  - `Structural head`：回归 `SD VAE latent`
  - 三阶段训练：
    - `warmup`：`H14 InfoNCE + 0.5*MSE`
    - `multitarget`：加 `B32/RN50 InfoNCE + SmoothL1(VAE latent)`
    - `finetune`：加 `hard-negative InfoNCE + supervised contrastive`

## Execution Steps
- Milestone 0：固化基线与计划文档
- Milestone 1：缓存 `H14/B32/RN50/VAE latent`
- Milestone 2：训练升级版 EEG encoder
- Milestone 3：训练 Prior Diffusion
- Milestone 4：打通 `Prior + single IP-Adapter + img2img`
- Milestone 5：评估、生成 Task 2 拼图、输出中英文结果汇总

## Notes
- 默认环境：`test`
- 默认优先：`debug` 分区做 smoke test
- 若 `SDXL-Turbo` 无法立即稳定跑通，允许先回退兼容生成链路，不阻塞 encoder/prior 主线
