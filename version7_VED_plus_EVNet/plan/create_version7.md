# create_version7

## 任务要求

1. 根据/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/temp/yliu674/v5_evnet的代码，复现下。

2. 主要复现这个实验8 blur level + EVNet fixed，然后再试试12 blur level + EVNet fixed。

4. 像之前那样用了将原训练集切分成新训练集和验证集，然后只用了新训练集在测试集测分外，最后还要用原训练集（之前切分的新训练集加上验证集）再在测试集测分。

5. 完成的任务是这个课程要求的task 1，文件路径是/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/Final_Project_Instructions/Project1.pdf。
6. 如果复现的结果和/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/temp/yliu674/README.md不一致，请分析原因。如果可以的话，最好能把遇到的这个问题给解决。



## 注意事项

1. 更多注意实现，请参考这个文件/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/plan/Notes_for_Attention.md。
2. 把用于slurm集群提交作业的.sh脚本放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version7_VED_plus_EVNet/slurm_scripts。
3. 工作区域主要在这个路径/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version7_VED_plus_EVNet，不过你也可能修改其他路径的文件。
4. 把plan mode生成的计划，储存为两份markdown，一份英文，一份中文，放在这个路径下/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/ChiKitWONG/Assignments/Project/DL_Project/version7_VED_plus_EVNet/plan。