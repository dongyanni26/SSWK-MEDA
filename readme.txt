1. Experiment_Houston1NN_SSWK_MEDA、Experiment_Indian1NN_SSWK_MEDA、Experiment_Worldview1NN_SSWK_MEDA三个函数是不同数据集的demo。
2. SSWK_MEDA是我们方法的函数，需要调用到spatial filter函数进行空间滤波、scale_normalization函数进行归一化、GFK_Map函数进行流形嵌入、estimate_mu函数进行动态分布对齐权值的估计、lapgraph函数计算拉普拉斯权值。
3. cvKnn是用于KNN分类的函数。