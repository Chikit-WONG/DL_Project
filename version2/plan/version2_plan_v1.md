# **Brain-to-Image Retrieval & Reconstruction** version 2 计划

## 这个项目之前做过的事情

这个项目是在/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project 文件夹下进行的。

之前根据/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/Final_Project_Instructions中的文件，完成了version 1，但效果不理想。具体代码和结果在/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version1路径下。

## 现在要做的事情

在这路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2，在做一版，看看能不能冲击更高的得分

这次我找了ChatGPT, Claude, Gemini三个AI互相讨论方案，最终讨论出来的计划和可以用来参考的论文、仓库链接放在了这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/plan。请根据他们讨论的结果，完成任务。

## 注意事项

1. 我是在slurm集群上工作的，所以要使用GPU资源的话，要用计算节点。使用计算节点的方法，可以参考文件/hpc2hdd/home/ckwong627/workdir/models/Qwen3-VL-8B-Instruct/run_qwen3_quick_start.sh和网站[香港科技大学（广州） HPC AI智算平台 知识库 | 香港科技大学（广州） HPC AI智算平台 知识库](https://docs.hpc.hkust-gz.edu.cn/)。尽量使用debug分区，但debug分区最多只能使用2张A40和运行30分钟，如果需要更多的资源，可以使用其它分区但要告诉我。

2. 创建的用于提交slurm作业的.sh脚本放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/slurm_scripts。因为slurm集群运行.ipynb文件有点困难，所以代码尽量创建.py文件，创建的.py文件放在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/codes。

3. 尽量使用名为test的conda环境。如果实在遇到环境冲突问题，可以创建新的conda环境，但要告诉我。

4. 如果需要下载模型，请先在路径/hpc2hdd/home/ckwong627/workdir/models 创建对应文件夹，再用命令

   ```bash
   hf download xxx --local-dir “刚刚创建的模型对应文件夹”
   ```

   去下载模型。下载了什么模型，模型大概占用多少储存空间，运行需要多少资源，也请告诉我。

5. 可以在/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/plan文件夹中的markdown文件，找到需要参考的论文和仓库链接。如果需要下载下来，论文请放这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/paper；仓库请放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references/repository。尽可能写代码前先参考对应论文和仓库（如果你觉得有必要的话。因为我感觉用以前人证实过的代码，成功率更高），如有必要请在我之前说的路径下载下来。如果觉得/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/plan路径下的markdown文件的引用链接有点乱，可以整理一份markdown文件放在/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/plan和/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/references路径下。

6. 结果放在/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/results路径下。日志放在/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/logs路径下。

7. Task 2的结果，请把ground truth和生成出来的图片拼在一起，这样我好看效果。

8. 任务运行完后，请在/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version2/results路径下，将模型得分整理一下，创建两份markdown文件，一份英文版，一份中文版。并且要与其他论文和version 1的模型得分进行对比。

9. 可以参考下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/sample_codes路径下TA给的sample codes。但不确定有没有用。

## 目标

1. 请尽可能提高模型的得分。如果模型得分还不如参考的论文，那就是有问题。
2. 优先完成指定好的任务。之后我会让你在试着能不能再迭代冲分。