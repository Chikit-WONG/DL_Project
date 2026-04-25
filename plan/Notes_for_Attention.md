1. 我是在slurm集群上工作的，所以要使用GPU资源的话，要用计算节点。使用计算节点的方法，可以参考文件/hpc2hdd/home/ckwong627/workdir/models/Qwen3-VL-8B-Instruct/run_qwen3_quick_start.sh和网站[香港科技大学（广州） HPC AI智算平台 知识库 | 香港科技大学（广州） HPC AI智算平台 知识库](https://docs.hpc.hkust-gz.edu.cn/)。`debug` 分区免费，最多只能使用 2 张 A40 且运行 30 分钟，也可以跑短 CPU 任务。除了 smoke test / env check 之外，如果正式任务预计也能在这个限制内完成，那么也应优先考虑 `debug`。但要注意 `debug` 还有按用户的 QOS 限制，所以不一定能无限并发；如果 `debug` 实际排队明显更慢，或者被用户级 QOS 限额卡住，再按照分区选择策略切换到别的分区。

2. 正式任务请按“先看 `debug` 是否够用，再按资源够用 + 低优先级优先 + 实时排队快”的原则选分区，不要默认用 `long_*` 分区。官网把分区分成三档：共享(低)/独占(中)/应急(高)。对于预计能在 `debug` 限制内完成的正式任务，不管是 CPU 还是 GPU，都先看 `debug`；如果 `debug` 没有明显优势，再按下面的顺序判断。建议顺序：
   - CPU 任务先看共享(低)：`i64m512u` / `a128m512u` / `i64m512r`，再看独占(中)：`i64m512ue` / `a128m512ue` / `i64m512re`，再看应急(高)：`emergency_cpu`。`long_cpu` 只在确实需要 14 天或当前明显更快时考虑。
   - A40 GPU 任务先看共享(低)：`i64m1tga40u`，再看独占(中)：`i64m1tga40ue`，再看应急(高)：`emergency_gpua40`。
   - A800 GPU 任务先看共享(低)：`i64m1tga800u` / `i64m1tga800u8ka`，再看独占(中)：`i64m1tga800ue`，再看应急(高)：`emergency_gpu`。`long_gpu` 只在确实需要 14 天或当前明显更快时考虑。
   - 大内存任务先看共享(低)：`i96m3tu`，再看独占(中)：`i96m3tue`。
   - 能用 A40 跑通的，不要默认上 A800；能用 CPU 的，不要默认上 GPU。
   如果一个分区等太久了，就换一个分区，并结合官网分区说明与实时 `squeue/sinfo` 情况重新判断。

3. `QOS` 可以理解为调度器的资源配额/限制策略层，可能限制单用户最多能同时提交多少作业、最多能同时用多少 CPU/GPU、最长能跑多久等。这类限制不只存在于 `debug`，其他分区/QOS 也可能有各自的 `MaxTRESPU`、`MaxJobsPU`、`MaxSubmitPU` 等限制。`debug` 只是因为限制更紧，所以更容易显现出来。更完整的分区选择策略已经整理到这个文件：/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/plan/Slurm_Partition_Selection_Strategy.md

4. 如果需要下载模型，请先在路径/hpc2hdd/home/ckwong627/workdir/models 创建对应文件夹，再用命令

   ```bash
   hf download xxx --local-dir “刚刚创建的模型对应文件夹”
   ```

   去下载模型。下载了什么模型，模型大概占用多少储存空间，运行需要多少资源，也请告诉我。

5. 按照课程要求的指标来评估模型得分。课程要求在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/Final_Project_Instructions。

6. 课程数据在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data。

7. 如果复现后发现模型得分和论文中的不一样，请分析原因。

8. 因为我感觉我的名为test的conda环境依赖挺多的，你看看可不可以复用。如果会有依赖冲突的话，就新建一个conda环境。

9. 可以参考这里sample code读取数据的方式 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/sample_codes
