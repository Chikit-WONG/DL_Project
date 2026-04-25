# version5_VED Task 2 计划：检索增强重建

## 总结

task 2 不重新设计一套新模型，而是在现有 `version5_VED` task 1 检索模型基础上扩展。最终系统需要：

1. 保留当前模糊感知 EEG 检索主干，
2. 在同一个 OpenCLIP RN50 空间里加入 class-text prototype 监督，
3. 对每个测试 EEG 检索 top-k 训练图像，
4. 按类别聚合检索结果并选出一个 prompt 类别，
5. 用固定模板 prompt 和 IP-Adapter 参考图生成图像，
6. 用课程口径的 `SSIM` 和 `CLIP` 做评估。

整个实现需要可复现、统一写入 `output/`，并能在课程 HPC 上运行。

## 核心设计

### 检索侧修改

- 继续使用当前 `Brain_Visual_Encoder_EEG` 主干和图像对齐分支。
- 不额外增加 text head，直接复用同一个 1024 维 OpenCLIP RN50 空间。
- 为每个训练类别构造一个文本 prototype：
  - `a realistic photo of a {class_name}`
- 从 task 1 checkpoint 初始化，按下面的联合 loss 微调：
  - `L_total = 0.7 * L_image + 0.3 * L_class`
- task 2 checkpoint 选择依据为 validation prompt-class retrieval 表现。

### Prompt 与检索逻辑

- 对每个 EEG 查询检索 top-k 训练图像。
- 默认 `top-k = 20`。
- 对检索结果按类别聚合得分。
- 得分最高的类别作为 prompt 类别。
- 得分最高的检索图像作为 IP-Adapter 参考图。
- prompt 模板固定为：
  - `a realistic photo of a {class_name}`

### 生成路径

- 第一版生成器使用 Stable Diffusion v1.5 + IP-Adapter。
- 条件输入：
  - text：聚合后选出的 prompt 类别
  - image：top-1 检索训练图
- 第一版不实现 `T2I-Adapter`、自由 prompt 生成和多参考图融合。

## 需要产出的结果

- task 2 微调 checkpoint
- 类别文本 prototype cache
- 训练图像 bank cache
- 测试集生成图像
- ground-truth 图像拷贝
- 逐样本检索元数据
- 逐 seed 重建评估 JSON
- 跨 seed 的 mean ± std 汇总
- 8–12 个定性例子或一张定性拼图

所有结果都必须放在 `version5_VED/output/` 下。

## 测试计划

- 验证 class-text prototype 生成后，训练集中每个类别都有一个 prototype。
- 验证 task 2 微调能无 shape mismatch 地加载 task 1 checkpoint。
- 验证检索元数据中包含：
  - prompt 类别，
  - top 检索图像，
  - top 检索类别，
  - ground-truth 图像路径。
- 验证 `--epoch 1 --n_seeds 1` 的 smoke run 能完整结束。
- 验证最终评估 JSON 至少包含：
  - `eval_ssim`
  - `eval_clip`

## 假设

- 仍然只使用 `sub-01`。
- task 1 checkpoint 已经存在。
- 第一版实现优先保证语义效果提升和课程复现性，不优先追求更复杂的新结构。
- 之所以优先使用 `IP-Adapter`，是因为当前可用条件是“检索到的相似参考图”，而不是可靠的 EEG 推断 control map。
