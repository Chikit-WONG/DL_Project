# Reproduce_CognitionCapturerPro_Plan

## 跑通仓库

将这个路径下的仓库跑通/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/CognitionCapturerPro。

优先跑通**Brain-to-Image Retrieval**和**Brain-to-Image Reconstruction**任务，如果能把其他任务也跑通，那就更好了。

先不用管DL_Project文件夹中，我没说要注意的文件。

## 注意事项

1. 我是在slurm集群上工作的，所以要使用GPU资源的话，要用计算节点。使用计算节点的方法，可以参考文件/hpc2hdd/home/ckwong627/workdir/models/Qwen3-VL-8B-Instruct/run_qwen3_quick_start.sh和网站[香港科技大学（广州） HPC AI智算平台 知识库 | 香港科技大学（广州） HPC AI智算平台 知识库](https://docs.hpc.hkust-gz.edu.cn/)。尽量使用debug分区，但debug分区最多只能使用2张A40和运行30分钟，如果需要更多的资源，可以使用其它分区但要告诉我。

2. 创建的用于提交slurm作业的.sh脚本放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/CognitionCapturerPro/slurm_scripts。

3. 如果需要下载模型，请先在路径/hpc2hdd/home/ckwong627/workdir/models 创建对应文件夹，再用命令

   ```bash
   hf download xxx --local-dir “刚刚创建的模型对应文件夹”
   ```

   去下载模型。下载了什么模型，模型大概占用多少储存空间，运行需要多少资源，也请告诉我。

4. 按照课程要求的指标来评估模型得分。课程要求在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/Final_Project_Instructions。

5. 课程数据在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data。

6. 如果复现后发现模型得分和论文中的不一样，请分析原因。

7. 因为我感觉我的名为test的conda环境依赖挺多的，你看看可不可以复用。如果会有依赖冲突的话，就新建一个conda环境。

8. 用plan mode弄出来的计划，保存为两份markdwon，一份英文，一份中文。放在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository/CognitionCapturerPro/plan
