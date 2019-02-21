# DLForMovieRecommend

2019/1/18  
1. 把NFM修改一下，用来排序，数据集也相应修改，代码能训练模型，但是loss没有优化的痕迹，有时候会出现nan  
深度学习模型参数如果初始化太小，则信号会不断缩小，很难产生作用；如果初始化太大，则信号会不断增大溢出，而起不到作用  
**nan问题解决了，是tf.log（x），x传入0或负值导致的！ 使用tf.clip_by_value()**
http://blog.sina.com.cn/s/blog_6ca0f5eb0102wr4j.html  
   
2019/1/21  
2. loss函数为负数，且不断向负无穷逼近
self.out = tf.sigmoid(self.out_train - self.out_nega)  
self.loss = tf.losses.compute_weighted_loss(tf.log(tf.clip_by_value(self.out, 1e-8, 1.0)))  
  
改为 ->  
self.loss = tf.losses.compute_weighted_loss(self.out)  
或  ->  
self.loss = tf.losses.compute_weighted_loss(-tf.log(tf.clip_by_value(self.out, 1e-8, 1.0)))  
  
3. topN推荐太慢了  
  
1) 每个用户并行处理推荐  
2) 物品提前筛选, 通过简单的算法过滤掉一些不太可能会推荐的电影  
  
  
  
2019/1/22  
4. 回到第一个问题,nan问题是用一个函数:tf.clip_by_value() 解决了,但是呢训练的时候会一直在一个值上下波动,  
这篇文章也提到了这个问题:  
http://blog.sina.com.cn/s/blog_6ca0f5eb0102wr4j.html  
问题的原因可能是:单纯的cut上下界值会影响优化,   
解决的办法是:考虑神经网络每层的值域, 通过激励函数:sigmoid, 初始化的参数也可能要考虑进去.  
进过修改后的神经网络终于朝着loss函数减小的方向前进了  
  
最后的loss函数:  
![NCF_loss](NCF_loss.PNG)  

5.嗯,朝着loss函数增大方向了!  
loss定义:  
self.positive = -tf.log(self.last_pair_out)  
self.negative = -tf.log(1-self.out)  
self.loss = tf.reduce_sum(self.positive + self.negative)  
  
原来的模型输入是这样子的, 如下  
self.train_features = tf.placeholder(tf.int32, shape=[2, None, None], name='featureV')  
self.last_pair_out = tf.placeholder(tf.float32, shape=[None, 1], name='lastPairOut')  
  
然后在训练的时候,先喂正样本的数据  
out = self.sess.run(self.out, feed_dict=feed_dict)
feed_dict[self.last_pair_out] = out  
然后在喂负样本:
这样子训练是会让loss函数一直增大的!  
从书上大致了解到,神经网络在训练的时候是用梯度下降的!,具体是计算从损失函数到每个可变参数(Viariable)的梯度,然后乘以一个学习率,然后相减.
在这里,self.positive, 实际上可以看做是一个常数, 所以损失函数实际只有负样本起作用, self.negative减小的同时, self.positive会增大!,所以总的损失函数增大是有可能的!  
解决办法是: 用placeholder把正负样本都加进去, 产生两个输入流, 最后汇聚在损失函数  
过程中可能覆盖了变量, 相同变量被重复赋值(因为有两个输入流),结果上是朝正确方向的  
```
Epoch 1 [18.0 s]	train=5354.8681 [0.0 s]  
Epoch 2 [17.7 s]	train=5350.6970 [0.0 s]  
Epoch 3 [17.7 s]	train=5350.1972 [0.0 s]  
Epoch 4 [17.5 s]	train=5349.6855 [0.0 s]  
Epoch 5 [17.7 s]	train=5346.7460 [0.0 s]  
Epoch 6 [18.1 s]	train=4520.9783 [0.0 s]  
Epoch 7 [18.8 s]	train=3497.4401 [0.0 s]  
Epoch 8 [18.8 s]	train=3295.0038 [0.0 s]  
Epoch 9 [18.2 s]	train=3173.6831 [0.0 s]  
Epoch 10 [18.1 s]	train=3077.8043 [0.0 s]  
Epoch 11 [18.0 s]	train=2981.6315 [0.0 s]  
Epoch 12 [18.0 s]	train=2901.3917 [0.0 s]  
Epoch 13 [18.0 s]	train=2831.0834 [0.0 s]  
Epoch 14 [18.2 s]	train=2775.1349 [0.0 s]  
Epoch 15 [18.0 s]	train=2729.9486 [0.0 s]  
Epoch 16 [17.7 s]	train=2694.1759 [0.0 s]  
Epoch 17 [18.0 s]	train=2654.7019 [0.0 s]  
Epoch 18 [18.2 s]	train=2632.2117 [0.0 s]  
Epoch 19 [18.0 s]	train=2605.1232 [0.0 s]  
Epoch 20 [18.1 s]	train=2575.8986 [0.0 s]  
Epoch 21 [18.6 s]	train=2556.0981 [0.0 s]  
Epoch 22 [17.6 s]	train=2537.3208 [0.0 s]  
Epoch 23 [18.5 s]	train=2525.1528 [0.0 s]  
Epoch 24 [18.0 s]	train=2512.0661 [0.0 s]  
Epoch 25 [18.0 s]	train=2498.1805 [0.0 s]  
```  
2019/1/23  
6.训练大约200轮后,loss收敛了, 但是HR指标不理想, 每个用户推荐10个电影, 总共只有277次命中  
HR=0.023  
这个结果显然是不理想的!也就比随机好点  
后面应该怎么寻找问题呢?  

2019/1/24  
发现了一个问题, 推荐列表里面有训练集的item, 去掉后,   
HR=0.043, 差不多翻倍了  
如果不训练的话: HR%10=0.0025  
下一步考虑模型的调参吧  
没有神经网络的话loss大概在1950收敛:  
```
hit: 497  
HR%10: 0.04114919688690181  
MAP%10: 0.023411417271102675  
NDCG%10: 0.008022221873267291  
```  
  
两层的话, 大约在1900收敛, HR=0.045  
  
  
感觉这样很奇怪啊, 模型肯定有哪里不对  
HR应该至少0.15才对  
最简单的协同过滤最起码是0.15  
  1/26 ~ 2/16  
  过年咯~
  
2019/2/17    
检查代码的时候发现一个很严重的bug  
userID和movieID同号时会认为是同一个特征  
特征数目也设置为用户数  
这是很严重的bug  
数据处理就出现问题, 后面学习出来的东西肯定一团糟的  
出现bug主要原因是在源代码基础上大幅度修改, 有些代码块含义也不是很明白  
后期打算再看一遍LoadData的代码  
今天开始把数据集换为ml-100K, 训练速度大幅度提升  
数据集修改格式真麻烦, 主要是不想修改源代码处理  
数据集中的movieID特征和movieID真实值起始位的错误  
  
2019/2/18  
找到一个有关的repo.  
https://github.com/zhang122994917/Pairwise-DeepFm  
修改后再python3运行一遍  
中间输出:  
```
Epoch 0 Global_step 100	Train_loss: 0.6430
Epoch 0 Global_step 200	Train_loss: 0.5985
Epoch 0 Global_step 300	Train_loss: 0.5951	Eval_NDCG@10: 0.2554	Eval_HR@10: 267.0000
Epoch 0 Global_step 400	Train_loss: 0.5931
Epoch 0 Global_step 500	Train_loss: 0.5650
Epoch 0 Global_step 600	Train_loss: 0.4820	Eval_NDCG@10: 0.2888	Eval_HR@10: 349.0000
Epoch 0 Global_step 700	Train_loss: 0.5026
Epoch 0 Global_step 800	Train_loss: 0.5451
Epoch 0 Global_step 900	Train_loss: 0.4134	Eval_NDCG@10: 0.3016	Eval_HR@10: 342.0000
Epoch 0 Global_step 1000	Train_loss: 0.5233
Epoch 0 Global_step 1100	Train_loss: 0.5484
Epoch 0 Global_step 1200	Train_loss: 0.5101	Eval_NDCG@10: 0.2999	Eval_HR@10: 337.0000
Epoch 0 Global_step 1300	Train_loss: 0.5972
Epoch 0 Global_step 1400	Train_loss: 0.5243
Epoch 0 Global_step 1500	Train_loss: 0.5049	Eval_NDCG@10: 0.3115	Eval_HR@10: 368.0000
Epoch 0 Global_step 1600	Train_loss: 0.5118
Epoch 0 Global_step 1700	Train_loss: 0.4941
Epoch 0 Global_step 1800	Train_loss: 0.5986	Eval_NDCG@10: 0.3126	Eval_HR@10: 374.0000
Epoch 0 Global_step 1900	Train_loss: 0.3647
Epoch 0 Global_step 2000	Train_loss: 0.5118
Epoch 0 Global_step 2100	Train_loss: 0.5011	Eval_NDCG@10: 0.2907	Eval_HR@10: 337.0000
Epoch 0 Global_step 2200	Train_loss: 0.4066
Epoch 0 Global_step 2300	Train_loss: 0.4717
Epoch 0 Global_step 2400	Train_loss: 0.6106	Eval_NDCG@10: 0.3223	Eval_HR@10: 400.0000
Epoch 1 Global_step 100	Train_loss: 0.4646
Epoch 1 Global_step 200	Train_loss: 0.4552
Epoch 1 Global_step 300	Train_loss: 0.3952	Eval_NDCG@10: 0.3294	Eval_HR@10: 415.0000
Epoch 1 Global_step 400	Train_loss: 0.4910
Epoch 1 Global_step 500	Train_loss: 0.3658
Epoch 1 Global_step 600	Train_loss: 0.3878	Eval_NDCG@10: 0.3456	Eval_HR@10: 425.0000
Epoch 1 Global_step 700	Train_loss: 0.5262
Epoch 1 Global_step 800	Train_loss: 0.4190
Epoch 1 Global_step 900	Train_loss: 0.3945	Eval_NDCG@10: 0.3571	Eval_HR@10: 424.0000
Epoch 1 Global_step 1000	Train_loss: 0.3325
Epoch 1 Global_step 1100	Train_loss: 0.4275
Epoch 1 Global_step 1200	Train_loss: 0.5733	Eval_NDCG@10: 0.3232	Eval_HR@10: 424.0000
Epoch 1 Global_step 1300	Train_loss: 0.4317
Epoch 1 Global_step 1400	Train_loss: 0.3936
Epoch 1 Global_step 1500	Train_loss: 0.4560	Eval_NDCG@10: 0.3420	Eval_HR@10: 428.0000
Epoch 1 Global_step 1600	Train_loss: 0.4720
Epoch 1 Global_step 1700	Train_loss: 0.3584
Epoch 1 Global_step 1800	Train_loss: 0.3436	Eval_NDCG@10: 0.3575	Eval_HR@10: 437.0000
Epoch 1 Global_step 1900	Train_loss: 0.3343
Epoch 1 Global_step 2000	Train_loss: 0.4269
Epoch 1 Global_step 2100	Train_loss: 0.3630	Eval_NDCG@10: 0.3705	Eval_HR@10: 455.0000
Epoch 1 Global_step 2200	Train_loss: 0.3948
Epoch 1 Global_step 2300	Train_loss: 0.4370
Epoch 1 Global_step 2400	Train_loss: 0.3405	Eval_NDCG@10: 0.3392	Eval_HR@10: 445.0000

最后程序大约第5轮NDCG可以到0.4, HR可以到500
```  
发现自己对NDCG理解有问题...  
idealNDCG是当前推荐列表命中情况下的最好排序CG值的和  
比如:  
当前推荐列表命中情况:  
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 
NDCG = 1 / log2(5)  
那么最好排序:  
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  
idealNDCG = 1 / log2(2) = 1    
NDCG = 1 / log2(5) ≈ 0.43  
  
现在试一下换数据集
数据集换好后  
NDCG ≈ 0.39  
HIT ≈ 390  
  
发现了致命bug!  
终于调试好了!  
原来不是模型的锅, 而是自己写的指标函数的...  
```
test_features = []
for i in item_list:
    test_features.append([u, i+user_num])
ranking = self.sess.run(self.out[0], feed_dict={self.train_features: [test_features],
                                                        self.dropout_keep: self.keep_prob})
```  
模型训练好后喂数据给模型, movieID忘加user_num了!!  
改了后HIT: 638!, MAP ≈ 0.153, NDCG还没改好, 不过应该会超过0.4的  

带神经网络的1500次训练结果:  
```
D:\software\anaconda\envs\python3\python.exe C:/Users/goldfish/Desktop/北航毕设/乱胡代码/NFM/NeuralFM.py --dataset ml-100k --hidden_factor 32 --layers [64,32] --keep_prob [0.6,0.6,0.6] --loss_type pairwise_loss --activation relu --pretrain 0 --optimizer AdagradOptimizer --lr 0.01 --batch_norm 1 --verbose 1 --early_stop 1 --epoch 1400 --batch_size 256
# of training: 89561
# of test: 10439
Neural FM: dataset=ml-100k, hidden_factor=32, dropout_keep=[0.6,0.6,0.6], layers=[64,32], loss_type=pairwise_loss, pretrain=0, #epoch=1400, batch=256, lr=0.0100, lambda=0.0000, optimizer=AdagradOptimizer, batch_norm=1, activation=relu, early_stop=1
2019-02-18 21:31:57.169245: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
#params: 90850
Epoch 1 [1.2 s]	train=228.6982 [0.0 s]
Epoch 2 [1.1 s]	train=217.1061 [0.0 s]
Epoch 3 [1.1 s]	train=210.4784 [0.0 s]
Epoch 4 [1.1 s]	train=205.5355 [0.0 s]
Epoch 5 [1.1 s]	train=178.7275 [0.0 s]
Epoch 6 [1.0 s]	train=139.1293 [0.0 s]
Epoch 7 [1.0 s]	train=123.3048 [0.0 s]
Epoch 8 [1.1 s]	train=111.7957 [0.0 s]
Epoch 9 [1.1 s]	train=105.8132 [0.0 s]
Epoch 10 [1.4 s]	train=100.7903 [0.0 s]
Epoch 11 [1.1 s]	train=99.1810 [0.0 s]
Epoch 12 [1.0 s]	train=96.2120 [0.0 s]
Epoch 13 [1.0 s]	train=93.5491 [0.0 s]
Epoch 14 [1.0 s]	train=92.7269 [0.0 s]
Epoch 15 [1.0 s]	train=90.3311 [0.0 s]
Epoch 16 [1.0 s]	train=90.6428 [0.0 s]
Epoch 17 [1.0 s]	train=89.5657 [0.0 s]
Epoch 18 [1.1 s]	train=89.0515 [0.0 s]
Epoch 19 [1.0 s]	train=86.7332 [0.0 s]
Epoch 20 [1.0 s]	train=86.0043 [0.0 s]
Epoch 21 [1.1 s]	train=86.7653 [0.0 s]
Epoch 22 [1.1 s]	train=84.8607 [0.0 s]
Epoch 23 [1.1 s]	train=84.1587 [0.0 s]
Epoch 24 [1.0 s]	train=84.3359 [0.0 s]
Epoch 25 [1.1 s]	train=82.7404 [0.0 s]
Epoch 26 [1.0 s]	train=83.3710 [0.0 s]
Epoch 27 [1.3 s]	train=82.1411 [0.0 s]
Epoch 28 [1.2 s]	train=82.2078 [0.0 s]
Epoch 29 [1.1 s]	train=81.9849 [0.0 s]
Epoch 30 [1.1 s]	train=80.3903 [0.0 s]
Epoch 31 [1.0 s]	train=80.6083 [0.0 s]
Epoch 32 [1.0 s]	train=80.8301 [0.0 s]
Epoch 33 [1.0 s]	train=80.1383 [0.0 s]
Epoch 34 [1.0 s]	train=80.4042 [0.0 s]
Epoch 35 [1.0 s]	train=80.7894 [0.0 s]
Epoch 36 [1.0 s]	train=79.6339 [0.0 s]
Epoch 37 [1.0 s]	train=80.1292 [0.0 s]
Epoch 38 [1.0 s]	train=80.1750 [0.0 s]
Epoch 39 [1.0 s]	train=78.5061 [0.0 s]
Epoch 40 [1.0 s]	train=79.2937 [0.0 s]
Epoch 41 [1.0 s]	train=79.4050 [0.0 s]
Epoch 42 [1.0 s]	train=78.2891 [0.0 s]
Epoch 43 [1.0 s]	train=78.2878 [0.0 s]
Epoch 44 [1.1 s]	train=77.9637 [0.0 s]
Epoch 45 [1.0 s]	train=77.6119 [0.0 s]
Epoch 46 [1.0 s]	train=76.9779 [0.0 s]
Epoch 47 [1.0 s]	train=78.8642 [0.0 s]
Epoch 48 [1.0 s]	train=77.2498 [0.0 s]
Epoch 49 [1.0 s]	train=77.6495 [0.0 s]
Epoch 50 [1.0 s]	train=77.6161 [0.0 s]
Epoch 51 [1.0 s]	train=76.7499 [0.0 s]
Epoch 52 [1.0 s]	train=76.3617 [0.0 s]
Epoch 53 [1.0 s]	train=76.8804 [0.0 s]
Epoch 54 [1.0 s]	train=77.3644 [0.0 s]
Epoch 55 [1.0 s]	train=75.2527 [0.0 s]
Epoch 56 [1.0 s]	train=76.7048 [0.0 s]
Epoch 57 [1.0 s]	train=75.6470 [0.0 s]
Epoch 58 [1.0 s]	train=75.8343 [0.0 s]
Epoch 59 [1.0 s]	train=75.5122 [0.0 s]
Epoch 60 [1.0 s]	train=75.3364 [0.0 s]
Epoch 61 [1.0 s]	train=76.3138 [0.0 s]
Epoch 62 [1.1 s]	train=75.5725 [0.0 s]
Epoch 63 [1.0 s]	train=75.6951 [0.0 s]
Epoch 64 [1.0 s]	train=75.0293 [0.0 s]
Epoch 65 [1.0 s]	train=75.5015 [0.0 s]
Epoch 66 [1.1 s]	train=75.0447 [0.0 s]
Epoch 67 [1.0 s]	train=75.0207 [0.0 s]
Epoch 68 [1.0 s]	train=74.7692 [0.0 s]
Epoch 69 [1.0 s]	train=75.0887 [0.0 s]
Epoch 70 [1.0 s]	train=75.3728 [0.0 s]
Epoch 71 [1.0 s]	train=74.4418 [0.0 s]
Epoch 72 [1.0 s]	train=74.1011 [0.0 s]
Epoch 73 [1.0 s]	train=73.4396 [0.0 s]
Epoch 74 [1.0 s]	train=74.0516 [0.0 s]
Epoch 75 [1.0 s]	train=74.7262 [0.0 s]
Epoch 76 [1.0 s]	train=74.3255 [0.0 s]
Epoch 77 [1.0 s]	train=73.6456 [0.0 s]
Epoch 78 [1.0 s]	train=74.8547 [0.0 s]
Epoch 79 [1.0 s]	train=74.7023 [0.0 s]
Epoch 80 [1.0 s]	train=73.4797 [0.0 s]
Epoch 81 [1.0 s]	train=74.4290 [0.0 s]
Epoch 82 [1.0 s]	train=74.8965 [0.0 s]
Epoch 83 [1.0 s]	train=73.6722 [0.0 s]
Epoch 84 [1.0 s]	train=73.4727 [0.0 s]
Epoch 85 [1.0 s]	train=73.2420 [0.0 s]
Epoch 86 [1.0 s]	train=73.9191 [0.0 s]
Epoch 87 [1.0 s]	train=73.4032 [0.0 s]
Epoch 88 [1.0 s]	train=73.9863 [0.0 s]
Epoch 89 [1.0 s]	train=73.0784 [0.0 s]
Epoch 90 [1.0 s]	train=73.8948 [0.0 s]
Epoch 91 [1.0 s]	train=73.2426 [0.0 s]
Epoch 92 [1.0 s]	train=72.3708 [0.0 s]
Epoch 93 [1.0 s]	train=73.8915 [0.0 s]
Epoch 94 [1.0 s]	train=73.5030 [0.0 s]
Epoch 95 [1.0 s]	train=72.3727 [0.0 s]
Epoch 96 [1.0 s]	train=72.2158 [0.0 s]
Epoch 97 [1.0 s]	train=73.1843 [0.0 s]
Epoch 98 [1.0 s]	train=73.3238 [0.0 s]
Epoch 99 [1.0 s]	train=72.9026 [0.0 s]
Epoch 100 [1.0 s]	train=72.9575 [0.0 s]
recommend 50 user costs 0.5806646347045898s
recommend 50 user costs 0.6126494407653809s
recommend 50 user costs 0.5846478939056396s
recommend 50 user costs 0.582676887512207s
recommend 50 user costs 0.5756747722625732s
recommend 50 user costs 0.6456215381622314s
recommend 50 user costs 0.5836682319641113s
recommend 50 user costs 0.576662540435791s
recommend 50 user costs 0.575430154800415s
recommend 50 user costs 0.5879197120666504s
recommend 50 user costs 0.6478729248046875s
recommend 50 user costs 0.5846507549285889s
recommend 50 user costs 0.5776667594909668s
recommend 50 user costs 0.5876741409301758s
recommend 50 user costs 0.5773346424102783s
recommend 50 user costs 0.5796527862548828s
recommend 50 user costs 0.5876612663269043s
recommend 50 user costs 0.5746815204620361s
hit: 692
HR%10: 0.06628987450905259
MAP%10: 0.1737310915293416
NDCG%10: 0.2475066108198663
Epoch 101 [1.0 s]	train=72.7044 [0.0 s]
Epoch 102 [1.0 s]	train=72.8126 [0.0 s]
Epoch 103 [1.0 s]	train=72.3288 [0.0 s]
Epoch 104 [1.0 s]	train=73.5825 [0.0 s]
Epoch 105 [1.0 s]	train=72.3533 [0.0 s]
......
Epoch 196 [1.0 s]	train=69.5973 [0.0 s]
Epoch 197 [1.0 s]	train=70.4445 [0.0 s]
Epoch 198 [1.0 s]	train=70.4439 [0.0 s]
Epoch 199 [1.0 s]	train=71.9017 [0.0 s]
Epoch 200 [1.0 s]	train=69.9690 [0.0 s]
hit: 742
HR%10: 0.07107960532618067
MAP%10: 0.1865292436050653
NDCG%10: 0.26499316554787167
Epoch 201 [1.0 s]	train=69.9581 [0.0 s]
Epoch 202 [1.0 s]	train=68.9443 [0.0 s]
Epoch 203 [1.0 s]	train=69.4402 [0.0 s]
Epoch 204 [1.0 s]	train=70.1583 [0.0 s]
Epoch 297 [1.0 s]	train=68.7675 [0.0 s]
Epoch 298 [1.0 s]	train=67.5607 [0.0 s]
Epoch 299 [1.0 s]	train=69.0381 [0.0 s]
Epoch 300 [1.0 s]	train=68.8177 [0.0 s]
hit: 721
HR%10: 0.06906791838298687
MAP%10: 0.1800129259315366
NDCG%10: 0.25573062695866483
Epoch 301 [1.0 s]	train=69.0226 [0.0 s]
Epoch 302 [1.0 s]	train=68.3051 [0.0 s]
Epoch 303 [1.0 s]	train=69.3493 [0.0 s]
Epoch 304 [1.0 s]	train=69.1646 [0.0 s]
Epoch 397 [1.0 s]	train=66.8380 [0.0 s]
Epoch 398 [1.0 s]	train=68.5018 [0.0 s]
Epoch 399 [1.0 s]	train=68.0287 [0.0 s]
Epoch 400 [1.0 s]	train=68.0670 [0.0 s]
hit: 758
HR%10: 0.07261231918766166
MAP%10: 0.19637696939857596
NDCG%10: 0.27805293291369454
Epoch 401 [1.0 s]	train=67.1450 [0.0 s]
Epoch 402 [1.0 s]	train=68.9109 [0.0 s]
Epoch 403 [1.0 s]	train=68.1443 [0.0 s]
Epoch 404 [1.0 s]	train=67.6448 [0.0 s]
Epoch 497 [1.0 s]	train=66.4748 [0.0 s]
Epoch 498 [1.0 s]	train=67.2107 [0.0 s]
Epoch 499 [1.0 s]	train=68.0711 [0.0 s]
Epoch 500 [1.0 s]	train=68.6530 [0.0 s]
hit: 761
HR%10: 0.07289970303668934
MAP%10: 0.18834946144122777
NDCG%10: 0.2719184388943225
Epoch 501 [1.0 s]	train=67.4729 [0.0 s]
Epoch 502 [1.0 s]	train=67.4591 [0.0 s]
Epoch 503 [1.0 s]	train=66.8133 [0.0 s]
Epoch 504 [1.0 s]	train=66.7627 [0.0 s]
Epoch 505 [1.0 s]	train=67.7189 [0.0 s]
Epoch 596 [1.0 s]	train=66.5867 [0.0 s]
Epoch 597 [1.0 s]	train=66.6349 [0.0 s]
Epoch 598 [1.0 s]	train=66.4774 [0.0 s]
Epoch 599 [1.0 s]	train=66.9327 [0.0 s]
Epoch 600 [1.0 s]	train=66.0474 [0.0 s]
hit: 764
HR%10: 0.07318708688571703
MAP%10: 0.18821751093509148
NDCG%10: 0.26857222407793463
Epoch 601 [1.0 s]	train=66.2261 [0.0 s]
Epoch 602 [1.0 s]	train=65.6836 [0.0 s]
Epoch 603 [1.0 s]	train=67.2182 [0.0 s]
Epoch 604 [1.0 s]	train=66.8793 [0.0 s]
Epoch 697 [1.0 s]	train=66.3461 [0.0 s]
Epoch 698 [1.0 s]	train=66.1917 [0.0 s]
Epoch 699 [1.0 s]	train=66.7148 [0.0 s]
Epoch 700 [1.0 s]	train=66.1175 [0.0 s]
hit: 751
HR%10: 0.07194175687326372
MAP%10: 0.1806530355750169
NDCG%10: 0.26075683195044486
Epoch 701 [1.0 s]	train=66.3937 [0.0 s]
Epoch 702 [1.0 s]	train=66.4533 [0.0 s]
Epoch 703 [1.0 s]	train=65.8703 [0.0 s]
Epoch 704 [1.0 s]	train=65.6265 [0.0 s]
Epoch 797 [1.0 s]	train=65.3817 [0.0 s]
Epoch 798 [1.0 s]	train=65.1009 [0.0 s]
Epoch 799 [1.0 s]	train=65.6585 [0.0 s]
Epoch 800 [1.0 s]	train=64.8764 [0.0 s]
hit: 754
HR%10: 0.0722291407222914
MAP%10: 0.18460519449914964
NDCG%10: 0.26354504131482626
Epoch 801 [1.1 s]	train=65.3879 [0.0 s]
Epoch 802 [1.0 s]	train=66.2811 [0.0 s]
Epoch 803 [1.0 s]	train=65.3955 [0.0 s]
Epoch 804 [1.0 s]	train=66.0836 [0.0 s]
Epoch 896 [1.0 s]	train=65.6597 [0.0 s]
Epoch 897 [1.0 s]	train=65.0203 [0.0 s]
Epoch 898 [1.0 s]	train=65.3454 [0.0 s]
Epoch 899 [1.0 s]	train=66.7595 [0.0 s]
Epoch 900 [1.0 s]	train=66.1537 [0.0 s]
hit: 775
HR%10: 0.0742408276654852
MAP%10: 0.1919328791372798
NDCG%10: 0.27413604728121943
Epoch 901 [1.0 s]	train=65.2868 [0.0 s]
Epoch 902 [1.0 s]	train=65.5766 [0.0 s]
Epoch 903 [1.0 s]	train=65.3478 [0.0 s]
Epoch 904 [1.0 s]	train=64.7710 [0.0 s]
Epoch 905 [1.0 s]	train=66.2010 [0.0 s]
Epoch 996 [1.0 s]	train=65.5330 [0.0 s]
Epoch 997 [1.0 s]	train=66.4635 [0.0 s]
Epoch 998 [1.0 s]	train=64.6336 [0.0 s]
Epoch 999 [1.0 s]	train=64.8429 [0.0 s]
Epoch 1000 [1.0 s]	train=65.4273 [0.0 s]
hit: 754
HR%10: 0.0722291407222914
MAP%10: 0.1988049580029959
NDCG%10: 0.27666970423693293
Epoch 1001 [1.1 s]	train=65.8312 [0.0 s]
Epoch 1002 [1.0 s]	train=64.6153 [0.0 s]
Epoch 1003 [1.0 s]	train=64.8594 [0.0 s]
Epoch 1004 [1.0 s]	train=65.3158 [0.0 s]
Epoch 1097 [1.0 s]	train=65.2749 [0.0 s]
Epoch 1098 [1.0 s]	train=65.2992 [0.0 s]
Epoch 1099 [1.0 s]	train=65.2479 [0.0 s]
Epoch 1100 [1.0 s]	train=65.7804 [0.0 s]
hit: 794
HR%10: 0.07606092537599386
MAP%10: 0.18455234305929272
NDCG%10: 0.2665840475596497
Epoch 1101 [1.1 s]	train=65.7675 [0.0 s]
Epoch 1102 [1.1 s]	train=65.0565 [0.0 s]
Epoch 1103 [1.1 s]	train=63.9296 [0.0 s]
Epoch 1104 [1.0 s]	train=65.6076 [0.0 s]
Epoch 1105 [1.1 s]	train=65.4226 [0.0 s]
Epoch 1196 [1.0 s]	train=64.0243 [0.0 s]
Epoch 1197 [1.0 s]	train=64.3807 [0.0 s]
Epoch 1198 [1.0 s]	train=63.9787 [0.0 s]
Epoch 1199 [1.0 s]	train=64.8942 [0.0 s]
Epoch 1200 [1.0 s]	train=64.8817 [0.0 s]
hit: 748
HR%10: 0.07165437302423604
MAP%10: 0.18601462300485164
NDCG%10: 0.2650142664880738
Epoch 1201 [1.1 s]	train=63.3466 [0.0 s]
Epoch 1202 [1.0 s]	train=65.3082 [0.0 s]
Epoch 1203 [1.0 s]	train=64.7985 [0.0 s]
Epoch 1204 [1.0 s]	train=65.6078 [0.0 s]
Epoch 1205 [1.0 s]	train=64.4920 [0.0 s]
Epoch 1296 [1.0 s]	train=63.6712 [0.0 s]
Epoch 1297 [1.0 s]	train=64.9056 [0.0 s]
Epoch 1298 [1.0 s]	train=64.9406 [0.0 s]
Epoch 1299 [1.0 s]	train=64.1619 [0.0 s]
Epoch 1300 [1.0 s]	train=64.7689 [0.0 s]
hit: 771
HR%10: 0.07385764920011495
MAP%10: 0.19348164138991264
NDCG%10: 0.2746412536032172
Epoch 1301 [1.0 s]	train=64.9851 [0.0 s]
Epoch 1302 [1.0 s]	train=64.6999 [0.0 s]
Epoch 1303 [1.0 s]	train=64.8810 [0.0 s]
Epoch 1304 [1.0 s]	train=64.0867 [0.0 s]
Epoch 1305 [1.0 s]	train=64.1578 [0.0 s]
Epoch 1397 [1.0 s]	train=64.5530 [0.0 s]
Epoch 1398 [1.0 s]	train=64.3768 [0.0 s]
Epoch 1399 [1.0 s]	train=65.0771 [0.0 s]
Epoch 1400 [1.0 s]	train=63.6035 [0.0 s]
hit: 781
HR%10: 0.07481559536354057
MAP%10: 0.20020807032604465
NDCG%10: 0.2806827094952065
```  
  
2019/2/19  
之前搭的wordpress博客上传图片好像有问题， 然后就换成hexo  
不会nodejs，义无反顾地安装好后发现出现类似的问题    
获取js，css，jpg这些静态文件时候会出现404错误  
事后发现是nginx服务器路由的锅  
这里学习下路由根目录和正则匹配的知识  
结果总算是搭好了：  
http://jinyu.me  
  
2019/2/20  
打算从头开始看：[推荐系统遇上深度学习系列](https://blog.csdn.net/jiangjiang_jian/article/details/80864300)  
加深自己对深度学习应用推荐系统领域知识的理解   
   
1. FM模型优化公式的推导  
  
2. FFM field-aware 拓展了什么  
  
3. DeepFM模型  
  
4. DeepFM 与 NFM 对比  
  
5. Deep&Cross Network 模型中 Cross模型用处？  
DeepFM 是 DNN 与 FM 的 并行组合  
Cross Network 是 DNN 与 Cross Network 的 并行组合  
FM因为效率原因一般只考虑二阶特征组合  
Cross Network因为多层交叉, 是多阶特征组合  
6. Cross Network学习   
![Cross Network](http://xudongyang.coding.me/dcn/dcn.png)  
http://xudongyang.coding.me/dcn/   
  
7. attention机制  
https://blog.csdn.net/jiangjiang_jian/article/details/80674250  
代码  

2019/2/21  
昨天复习先放着, 今天打算做下开题报告的准备  
提纲已列好, 主要写"主要内容" 和 "国内外研究历史与现状"  
  
思考过程有一个很有趣问题:
用深度学习训练好的模型对于冷启动的用户和电影有什么推荐效果吗?  
