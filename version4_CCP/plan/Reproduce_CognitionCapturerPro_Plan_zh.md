# CognitionCapturerPro 课程数据复现计划

## 目标
- 在课程固定的 `image-eeg-data` 数据上跑通 `CognitionCapturerPro`
- 优先完成 `Brain-to-Image Retrieval` 和 `Brain-to-Image Reconstruction`
- 输出课程要求的检索与重建指标，并保留可复现实验脚本

## 实施路径
1. 复用 `test` conda 环境，记录版本偏差：`Python 3.10 / torch 2.10`
2. 将课程数据适配为 CogCapPro 期望的 `ThingsEEG` 目录布局
3. 由课程原图生成 `Image_depth_set_Resize` 与 `Image_edge_set_Resize`
4. 修改 EEG loader，直接接受课程 `train.pt/test.pt` 的真实字段与 trial 维度
5. 使用现有的 `sdxl-turbo`、`IP-Adapter`、`OpenCLIP` 权重完成训练、对齐和生成
6. 用课程 sample code 同口径实现重建评测脚本

## 关键默认值
- subject 固定为 `sub-01`
- 训练和测试都从文件真实 shape 推断重复次数，不再硬编码
- 文本分支优先使用课程数据自带 `text` 字段；没有 BLIP2 私有文件时不阻塞训练
- 主跑使用 `ViT-H-14` 视觉骨干，原因是本地已有权重；若后续补齐 `RN50` 可再做对照
- smoke run 优先验证端到端可运行，再决定是否延长正式训练

## 评测
- Retrieval: `Top-1`、`Top-5`
- Reconstruction:
  - 必做：`SSIM`、`CLIP`
  - 扩展：`PixCorr`、`AlexNet2`、`AlexNet5`、`Inception`、`EffNe0`、`SwAV`

## 产物
- `configs/local.yaml`
- `scripts/prepare_course_data.py`
- `scripts/prepare_diffusion_embeddings.py`
- `scripts/evaluate_reconstruction.py`
- `slurm_scripts/*.sh`
- `runs/...` 下的训练、对齐、重建与评测输出
