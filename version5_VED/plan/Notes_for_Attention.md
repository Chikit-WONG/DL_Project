1. 我是在slurm集群上工作的，所以要使用GPU资源的话，要用计算节点。使用计算节点的方法，可以参考文件/hpc2hdd/home/ckwong627/workdir/models/Qwen3-VL-8B-Instruct/run_qwen3_quick_start.sh和网站[香港科技大学（广州） HPC AI智算平台 知识库 | 香港科技大学（广州） HPC AI智算平台 知识库](https://docs.hpc.hkust-gz.edu.cn/)。尽量使用debug分区，但debug分区最多只能使用2张A40和运行30分钟，如果需要更多的资源，可以使用其它分区但要告诉我。

2. 创建的用于提交slurm作业的.sh脚本放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version5_VED/slurm_scripts。

3. 如果需要下载模型，请先在路径/hpc2hdd/home/ckwong627/workdir/models 创建对应文件夹，再用命令

   ```bash
   hf download xxx --local-dir “刚刚创建的模型对应文件夹”
   ```

   去下载模型。下载了什么模型，模型大概占用多少储存空间，运行需要多少资源，也请告诉我。

4. 按照课程要求的指标来评估模型得分。课程要求在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/Final_Project_Instructions。

5. 课程数据在这个路径下 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data。

6. 如果复现后发现模型得分和论文中的不一样，请分析原因。

7. 因为我感觉我的名为test的conda环境依赖挺多的，你看看可不可以复用。如果会有依赖冲突的话，就新建一个conda环境。

8. 可以参考这里sample code读取数据的方式 /hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/sample_codes

10. 要等待的任务就继续等待，等要完成的任务完成了就继续，不要停，直到完成所有任务。如果一个分区等太久了，就换一个分区。最优先debug分区，如果debug分区资源、时间不够用，再去其他分区。如果分区等太久的话，建议去emergency_gpua40或emergency_gpu分区看看，按照经验，这2个分区应该排队比较快(优先去emergency_gpua40分区，如果一张A40，也就是48 GB显存够用的话，毕竟emergency_gpua40分区比emergency_gpu分区更便宜一些。但如果emergency_gpua40分区要排很久的队，那就去emergency_gpu分区或其他排队更少的分区)，但也不一定就是排队就快的分区，如果不是，你也到时候帮我按照最快排队的分区去运行程序。