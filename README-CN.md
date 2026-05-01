# DSAA2012 深度学习期末项目：基于 EEG 的图像解码

[English README](README.md)

本仓库是 DSAA2012 深度学习课程期末项目，主题是在 THINGS-EEG 数据集上进行基于脑电信号的视觉解码。项目主要包含两个任务：

1. **图像检索**：将 EEG 信号映射到视觉 embedding 空间，并在 200 类候选图像中检索正确图像。
2. **图像重建**：利用 EEG 预测出的视觉特征，通过扩散模型和 IP-Adapter 等条件生成方法重建图像。

仓库中保留了多个实验版本，因为这个项目经历了原始方案、LLM 共同规划方案，以及两个论文/开源项目复现方向。每个版本的具体运行命令、模型结构和结果说明放在对应版本目录的 README 中。

## 仓库结构

```text
DL_Project/
├── Final_Project_Instructions/   # 课程项目 PDF
├── image-eeg-data/               # 本地 THINGS-EEG 数据，已被 git 忽略
├── model_score_compare/          # 各版本分数对比摘要
├── plan/                         # 项目计划文档，Markdown 文件允许提交
├── references/                   # 论文和本地参考材料
├── sample_codes/                 # 原始示例 notebook
├── version1/                     # 一开始的原计划/基线
├── version2/                     # ChatGPT、Claude、Gemini 三方讨论出的计划，结果不理想
├── version3_ATM/                 # 复现 EEG_Image_decode (ATM)
├── version4_CCP/                 # 复现 CognitionCapturerPro (CCP)
├── version5_VED/                 # 复现并适配 VisualEEGDecoding
├── version6_BP-MGD/              # Safe BP-MGD：防泄漏的任务 2 重建分支
└── version7_VED_plus_EVNet/      # VED + EVNet：加入仿生视觉前端的检索增强分支
```

## 各版本说明

| 版本 | 主要思路 | 当前定位 | 本地主要结果 |
|---|---|---|---|
| [`version1`](version1/README-CN.md) | 一开始的原计划：自定义 EEG encoder 对齐 CLIP ViT-H/14，再用 SD v1.5 + IP-Adapter 做重建 | 基线与参考实现 | 完整重跑：Top-1 24.5%，Top-5 53.0%，SSIM 0.2633，CLIP 0.7836 |
| [`version2`](version2/README-CN.md) | 由 ChatGPT、Claude、Gemini 三方互相讨论后制定的方案，包含更强的双路径 encoder、多目标视觉监督和轻量 prior | 探索性尝试；最终结果很不理想，没有达到预期 | 完整重跑：Top-1 20.0%，Top-5 50.5%，SSIM 0.3753，CLIP 0.2755 |
| [`version3_ATM`](version3_ATM/README-CN.md) | 复现并适配 `EEG_Image_decode`，核心路线是 ATM/ATMS | 检索效果较强，并包含完整评估脚本 | 完整重跑：Top-1 33.5%，Top-5 63.5%，SSIM 0.2709，CLIP 0.6089 |
| [`version4_CCP`](version4_CCP/README-CN.md) | 复现并适配 `CognitionCapturerPro` (CCP)，包含多模态 embedding、alignment 和 SDXL-Turbo 生成 | 后期主要的 CCP 重建/适配分支 | 完整重跑：Any-modality Top-1 61.5%，Top-5 89.0%；重建（`all`）SSIM 0.3732，CLIP 0.8981 |
| [`version5_VED`](version5_VED/README-CN.md) | 复现并适配 `VisualEEGDecoding`，task 1 使用 multi-blur OpenCLIP RN50 检索路线，task 2 使用检索增强的 IP-Adapter 重建路线 | 完整的 task1/task2 流程基线，已被 version7 进一步扩展 | Task 1：当前选定提交分数 Top-1 86.85% ± 0.63%，Top-5 98.10% ± 0.52%；Task 2：SSIM 0.2977 ± 0.0066，CLIP 0.7610 ± 0.0148 |
| [`version6_BP-MGD`](version6_BP-MGD/README.md) | Safe BP-MGD：基于仅训练集原型记忆库的防泄漏任务 2 重建流程，使用 SDXL-Turbo 生成和 IP-Adapter 条件控制 | 严格防泄漏的任务 2 重建分支（测试图像在生成阶段完全不可见） | Task 2：SSIM 0.3799，CLIP 0.5346（全量训练 hybrid SDXL 最终运行） |
| [`version7_VED_plus_EVNet`](version7_VED_plus_EVNet/README-CN.md) | 在 version5_VED 基础上引入仿生视觉前端 EVNet（SubcorticalBlock + VOneBlock，全参数冻结），通过可学习权重与多尺度模糊 CLIP 特征融合 | 当前任务 1 检索效果最好的分支；包含模糊级别数、训练集大小、适配层初始化方式、CLIP 主干的消融实验 | Task 1：Best-test Top-1 87.85% ± 0.82%（8-blur + EVNet，RN50，全量训练，10 seeds） |

## 数据和模型文件

原始数据和转换后的数据体积较大，不提交到 GitHub。默认本地数据目录为：

```text
image-eeg-data/
├── train.pt
├── test.pt
├── EEG_CHANNELS.jsonl
├── training_images/
└── test_images/
```

部分版本还会在 `image-eeg-data/converted_for_cogcappro/` 下生成适配 CognitionCapturerPro 的转换数据。

预训练模型权重也不提交到 GitHub，通常放在：

```text
/hpc2hdd/home/ckwong627/workdir/models/
```

常见依赖包括 CLIP ViT-H/14、OpenCLIP RN50、CLIP ViT-H/14（LAION-2B，用于 version7 消融实验）、Stable Diffusion v1.5、SDXL-Turbo 和 IP-Adapter 权重。精确路径以各版本 README 和配置文件为准。

## 结果文件和提交策略

仓库会保留轻量级结果摘要和部分可视化图片，例如：

- 各版本输出目录中的 metrics JSON/CSV；
- Task 2 的 montage 或 comparison 图片；
- Markdown 结果总结和计划文档。

以下内容默认不提交：模型 checkpoint、`.pt` tensor、缓存、日志、临时文件、本地数据集和体积较大的参考仓库。

## Git 管理说明

顶层 `.gitignore` 的目标是避免把大文件推送到 GitHub，同时保留对组员有用的文档和结果：

- 忽略：数据集、模型权重、tensor 缓存、日志、SLURM 输出、临时文件、大型参考仓库；
- 保留：源代码、README、`plan/` 下的 Markdown 计划、指标摘要和部分结果图。

推送前建议先检查：

```bash
git status --short
git add -n README.md README-CN.md .gitignore plan/*.md
```

其中 `git add -n` 是 dry-run，只检查哪些文件会被暂存，不会真的修改暂存区。

## 建议阅读顺序

1. 先阅读本根目录 README，了解整体结构。
2. 阅读 [`version1/README-CN.md`](version1/README-CN.md)，了解一开始的原计划/基线流程。
3. 阅读 [`version3_ATM/README-CN.md`](version3_ATM/README-CN.md) 和 [`version4_CCP/README-CN.md`](version4_CCP/README-CN.md)，了解两个偏重重建的复现方向。
4. 阅读 [`version5_VED/README-CN.md`](version5_VED/README-CN.md)，了解 VisualEEGDecoding 基线，包括检索增强的 task 2 重建流程。
5. 阅读 [`version6_BP-MGD/README.md`](version6_BP-MGD/README.md)，了解防泄漏的任务 2 重建分支。
6. 阅读 [`version7_VED_plus_EVNet/README-CN.md`](version7_VED_plus_EVNet/README-CN.md)，了解当前任务 1 效果最好的分支，即在 version5_VED 基础上融入 EVNet 仿生视觉前端的扩展方案。
7. 查看 [`plan/`](plan/) 了解计划历史和实现决策。
8. 查看 [`model_score_compare/`](model_score_compare/) 了解各版本简明分数对比。
