# Reproduce_VisualEEGDecoding_Plan

## 跑通仓库

将这个路径下的仓库跑通/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding。

优先跑通**Brain-to-Image Retrieval**和**Brain-to-Image Reconstruction**任务（如果有的话），如果能把其他任务也跑通，那就更好了。

先不用太管DL_Project文件夹中的文件，特别是version那里的文件夹，那是我用其他方法完成**Brain-to-Image Retrieval**和**Brain-to-Image Reconstruction**任务的尝试。

## 注意事项

1. 我是在slurm集群上工作的，所以要使用GPU资源的话，要用计算节点。使用计算节点的方法，可以参考文件/hpc2hdd/home/ckwong627/workdir/models/Qwen3-VL-8B-Instruct/run_qwen3_quick_start.sh和网站[香港科技大学（广州） HPC AI智算平台 知识库 | 香港科技大学（广州） HPC AI智算平台 知识库](https://docs.hpc.hkust-gz.edu.cn/)。尽量使用debug分区，但debug分区最多只能使用2张A40和运行30分钟，如果需要更多的资源，可以使用其它分区但要告诉我。

2. 创建的用于提交slurm作业的.sh脚本放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding/slurm_scripts。

3. 如果需要下载模型，请先在路径/hpc2hdd/home/ckwong627/workdir/models 创建对应文件夹，再用命令

   ```bash
   hf download xxx --local-dir “刚刚创建的模型对应文件夹”
   ```

   去下载模型。下载了什么模型，模型大概占用多少储存空间，运行需要多少资源，也请告诉我。

4. 如果需要下载数据集（不过这个任务应该不用下载数据集），请先在路径/hpc2hdd/home/ckwong627/workdir/dataset 创建对应文件夹，再用命令

   ```bash
   hf download xxx --local-dir “刚刚创建的模型对应文件夹”
   ```

   去下载数据集。下载了什么数据集，数据集大概占用多少储存空间，也请告诉我。

5. 按照课程要求的指标来评估模型得分。课程要求在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/Final_Project_Instructions。

6. 课程数据在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data。用课程数据来训练模型和评估模型得分。

7. 读取课程数据和评估模型得分，可以参考这个路径下的文件/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/sample_codes。

8. 如果复现后发现模型得分和论文中的不一样，请分析原因。论文在这个路径/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/paper/Liu 等 - 2026 - Leveraging Visual Blur Perception Characteristics for EEG Decoding.pdf。

9. 因为我感觉我的名为test的conda环境依赖挺多的，你看看可不可以复用。如果会有依赖冲突的话，就新建一个conda环境。

10. 可以参考这里sample code读取数据的方式 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/sample_codes

11. 用plan mode弄出来的计划，保存为两份markdwon，一份英文，一份中文。放在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding/plan

12. 要等待的任务就继续等待，等要完成的任务完成了就继续，不要停，直到完成所有任务。如果一个分区等太久了，就换一个分区。

13. 如果5h的token额度用完了，要使用extra usage了，请先停下来，等5h token额度重置了，再继续任务。我会使用/codex:rescue这个skill，所以你也会用到codex的额度。

14. 我看/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/VisualEEGDecoding这个仓库好多都是.ipynb文件，不知道slurm集群好不好允许.ipynb文件，如果你觉得不好运行，可以将它们先转成.py文件再运行。