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
无法做实验, 因为代码中的特征序号是紧挨着的, 新用户ID == 第一个电影ID, 这样需要修改一下代码  

下午写了快千字的开题报告←.←  
博客改了UI,接触SWIG语法,开启前端大坑...  
  
今天开题报告的思路搞得七七八八了, 下面整理一下通过开题报告发现自己对课题哪方面的不足,需要改进的部分   
为什么觉得每一个部分写的时候都要想很多很久,但是又写不出来...  
国内外研究现状不会写, 没有关注...  
现在平台是用什么技术做推荐系统没有一点了解的  
以后多看博文提高自己姿势吧...  
  
2019/2/22  
补:   
NFM代码有两个问题:  
第一个是模型输出有两个选择, out[0] 和 out[1] 实验结果out[0]更好一些, 但是out[1]结果也没有多差, 大概差距10%  
模型训练的最终结果和模型保存后读取的结果不同, 一般稍微低过训练直接输出的结果,原因?  

今天主要为实验做准备, 在网上搜一下比较好的FM, FFM, DeepFM代码  
开始找的是这个:  
https://github.com/princewen/tensorflow_practice/tree/master/recommendation/recommendation-FM-demo
但是呢, 这代码可读性很差, 用的特征只有用户和电影的ID, 由csr_matrix实现, 因为后期要添加其他特征, 代码拓展性就不如意了  
下午, 找到一个比较好的repo.  
https://github.com/Johnson0722/CTR_Prediction
今天看了FM部分的代码, 之后把他修改下  
btw. 博客想加个很神奇的东西...

2019/02/23
前端跨域获取元素是大坑...  
  
今天跑通了  
https://github.com/Johnson0722/CTR_Prediction  
部分的FM代码, 然后把除了模型主体外的其他部分做了修改, 还差模型, 和调试就完成了  
这是第一次实现多特征的FM模型, 希望能成功  

2019/02/24  
FM跑通了, 新学到sparse_placeholder等稀疏喂数据方式  
现在在跑epoch = 40, 的程序, 跑完估计时间大约要 5 小时 ......  
  
```
start training...
2019-02-24 14:59:27,014 : INFO : Initializing fresh parameters for the my Factorization Machine
2019-02-24 15:00:31,726 : INFO : Iteration 500: with minibatch training loss = 0.4664325714111328
2019-02-24 15:01:35,974 : INFO : Iteration 1000: with minibatch training loss = 0.5151827931404114
2019-02-24 15:02:40,968 : INFO : Iteration 1500: with minibatch training loss = 0.47085273265838623
Epoch 1, Overall loss = 0.498
2019-02-24 15:03:45,425 : INFO : Iteration 2000: with minibatch training loss = 0.44380834698677063
2019-02-24 15:04:49,097 : INFO : Iteration 2500: with minibatch training loss = 0.4268592596054077
2019-02-24 15:05:53,713 : INFO : Iteration 3000: with minibatch training loss = 0.5693184733390808
2019-02-24 15:06:59,046 : INFO : Iteration 3500: with minibatch training loss = 0.43050941824913025
Epoch 2, Overall loss = 0.438
2019-02-24 15:08:02,425 : INFO : Iteration 4000: with minibatch training loss = 0.41970735788345337
2019-02-24 15:09:06,925 : INFO : Iteration 4500: with minibatch training loss = 0.4762305021286011
2019-02-24 15:10:12,421 : INFO : Iteration 5000: with minibatch training loss = 0.44770875573158264
Epoch 3, Overall loss = 0.421
2019-02-24 15:11:18,326 : INFO : Iteration 5500: with minibatch training loss = 0.3069072663784027
2019-02-24 15:12:21,743 : INFO : Iteration 6000: with minibatch training loss = 0.5221328735351562
2019-02-24 15:13:26,043 : INFO : Iteration 6500: with minibatch training loss = 0.35997655987739563
2019-02-24 15:14:31,285 : INFO : Iteration 7000: with minibatch training loss = 0.40101051330566406
Epoch 4, Overall loss = 0.407
2019-02-24 15:15:34,527 : INFO : Iteration 7500: with minibatch training loss = 0.39187681674957275
2019-02-24 15:16:38,505 : INFO : Iteration 8000: with minibatch training loss = 0.506892204284668
2019-02-24 15:17:43,410 : INFO : Iteration 8500: with minibatch training loss = 0.36633235216140747
Epoch 5, Overall loss = 0.402
recommend 200 user costs 28.35388493537903s
recommend 200 user costs 28.738603115081787s
recommend 200 user costs 29.140575170516968s
recommend 200 user costs 29.17214012145996s
recommend 200 user costs 29.234046459197998s
recommend 200 user costs 29.30038094520569s
recommend 200 user costs 29.330805778503418s
recommend 200 user costs 29.493829011917114s
recommend 200 user costs 29.828416109085083s
recommend 200 user costs 29.86491584777832s
recommend 200 user costs 29.73957371711731s
recommend 200 user costs 30.039836645126343s
recommend 200 user costs 29.968694925308228s
recommend 200 user costs 29.966902017593384s
recommend 200 user costs 30.23908233642578s
recommend 200 user costs 30.160735607147217s
recommend 200 user costs 30.21017599105835s
recommend 200 user costs 30.304189443588257s
recommend 200 user costs 30.462279319763184s
recommend 200 user costs 30.344521045684814s
recommend 200 user costs 30.5133056640625s
recommend 200 user costs 30.535704612731934s
recommend 200 user costs 30.590941667556763s
recommend 200 user costs 30.56850814819336s
recommend 200 user costs 30.735422372817993s
recommend 200 user costs 30.985790252685547s
recommend 200 user costs 31.15510368347168s
recommend 200 user costs 32.08350396156311s
recommend 200 user costs 31.2036874294281s
recommend 200 user costs 31.242165327072144s
hit: 3180
HR%10: 0.03094619449391294
MAP%10: 0.12063713815180253
NDCG%10: 0.17106648184213963
2019-02-24 15:34:01,757 : INFO : Iteration 9000: with minibatch training loss = 0.3175331950187683
2019-02-24 15:35:11,842 : INFO : Iteration 9500: with minibatch training loss = 0.40875381231307983
2019-02-24 15:36:23,826 : INFO : Iteration 10000: with minibatch training loss = 0.3472834825515747
2019-02-24 15:37:36,109 : INFO : Iteration 10500: with minibatch training loss = 0.43041545152664185
Epoch 6, Overall loss = 0.394
2019-02-24 15:38:46,261 : INFO : Iteration 11000: with minibatch training loss = 0.4445030689239502
2019-02-24 15:39:56,913 : INFO : Iteration 11500: with minibatch training loss = 0.35770729184150696
2019-02-24 15:41:08,178 : INFO : Iteration 12000: with minibatch training loss = 0.46357208490371704
Epoch 7, Overall loss = 0.391
2019-02-24 15:42:19,143 : INFO : Iteration 12500: with minibatch training loss = 0.37092864513397217
2019-02-24 15:43:29,473 : INFO : Iteration 13000: with minibatch training loss = 0.43272578716278076
2019-02-24 15:44:40,555 : INFO : Iteration 13500: with minibatch training loss = 0.46249210834503174
2019-02-24 15:45:52,328 : INFO : Iteration 14000: with minibatch training loss = 0.5180768370628357
Epoch 8, Overall loss = 0.393
2019-02-24 15:47:02,032 : INFO : Iteration 14500: with minibatch training loss = 0.4181518852710724
2019-02-24 15:48:12,873 : INFO : Iteration 15000: with minibatch training loss = 0.377219557762146
2019-02-24 15:49:24,475 : INFO : Iteration 15500: with minibatch training loss = 0.3139384984970093
Epoch 9, Overall loss = 0.387
2019-02-24 15:50:35,454 : INFO : Iteration 16000: with minibatch training loss = 0.3119763135910034
2019-02-24 15:51:45,790 : INFO : Iteration 16500: with minibatch training loss = 0.40119796991348267
2019-02-24 15:52:57,163 : INFO : Iteration 17000: with minibatch training loss = 0.4552682638168335
2019-02-24 15:54:09,340 : INFO : Iteration 17500: with minibatch training loss = 0.32665103673934937
Epoch 10, Overall loss = 0.384
hit: 3265
HR%10: 0.031773372648624454
MAP%10: 0.12107917329919476
NDCG%10: 0.17235115152113967
2019-02-24 16:10:24,495 : INFO : Iteration 18000: with minibatch training loss = 0.3861101269721985
2019-02-24 16:11:35,084 : INFO : Iteration 18500: with minibatch training loss = 0.33651965856552124
2019-02-24 16:12:46,595 : INFO : Iteration 19000: with minibatch training loss = 0.3118653893470764
Epoch 11, Overall loss = 0.385
2019-02-24 16:13:57,844 : INFO : Iteration 19500: with minibatch training loss = 0.3327292203903198
2019-02-24 16:15:07,880 : INFO : Iteration 20000: with minibatch training loss = 0.3508645296096802
2019-02-24 16:16:18,943 : INFO : Iteration 20500: with minibatch training loss = 0.42334264516830444
2019-02-24 16:17:30,827 : INFO : Iteration 21000: with minibatch training loss = 0.3660416603088379
Epoch 12, Overall loss = 0.384
2019-02-24 16:18:40,974 : INFO : Iteration 21500: with minibatch training loss = 0.3886626362800598
2019-02-24 16:19:52,039 : INFO : Iteration 22000: with minibatch training loss = 0.3742499351501465
2019-02-24 16:21:03,545 : INFO : Iteration 22500: with minibatch training loss = 0.4330451190471649
Epoch 13, Overall loss = 0.385
2019-02-24 16:22:15,417 : INFO : Iteration 23000: with minibatch training loss = 0.40674304962158203
2019-02-24 16:23:25,531 : INFO : Iteration 23500: with minibatch training loss = 0.3865116238594055
2019-02-24 16:24:36,507 : INFO : Iteration 24000: with minibatch training loss = 0.3743622899055481
2019-02-24 16:25:48,232 : INFO : Iteration 24500: with minibatch training loss = 0.34209632873535156
Epoch 14, Overall loss = 0.383
2019-02-24 16:26:57,957 : INFO : Iteration 25000: with minibatch training loss = 0.375931978225708
2019-02-24 16:28:08,599 : INFO : Iteration 25500: with minibatch training loss = 0.427631676197052
2019-02-24 16:29:19,859 : INFO : Iteration 26000: with minibatch training loss = 0.2829141616821289
Epoch 15, Overall loss = 0.38
hit: 3253
HR%10: 0.031656594556194594
MAP%10: 0.12114342917231885
NDCG%10: 0.17158307512477847
2019-02-24 16:45:42,602 : INFO : Iteration 26500: with minibatch training loss = 0.29819256067276
2019-02-24 16:46:52,826 : INFO : Iteration 27000: with minibatch training loss = 0.3379024565219879
2019-02-24 16:48:03,934 : INFO : Iteration 27500: with minibatch training loss = 0.4339085519313812
2019-02-24 16:49:15,692 : INFO : Iteration 28000: with minibatch training loss = 0.46056073904037476
Epoch 16, Overall loss = 0.38
2019-02-24 16:50:25,668 : INFO : Iteration 28500: with minibatch training loss = 0.6689988374710083
2019-02-24 16:51:36,190 : INFO : Iteration 29000: with minibatch training loss = 0.36170095205307007
2019-02-24 16:52:47,672 : INFO : Iteration 29500: with minibatch training loss = 0.4743526875972748
Epoch 17, Overall loss = 0.381
2019-02-24 16:53:58,362 : INFO : Iteration 30000: with minibatch training loss = 0.3340596556663513
2019-02-24 16:55:08,100 : INFO : Iteration 30500: with minibatch training loss = 0.3682502508163452
2019-02-24 16:56:18,736 : INFO : Iteration 31000: with minibatch training loss = 0.3429356813430786
2019-02-24 16:57:29,955 : INFO : Iteration 31500: with minibatch training loss = 0.3414451479911804
Epoch 18, Overall loss = 0.38
2019-02-24 16:58:39,748 : INFO : Iteration 32000: with minibatch training loss = 0.39093440771102905
2019-02-24 16:59:50,018 : INFO : Iteration 32500: with minibatch training loss = 0.30766886472702026
2019-02-24 17:01:00,840 : INFO : Iteration 33000: with minibatch training loss = 0.5031550526618958
Epoch 19, Overall loss = 0.374
2019-02-24 17:02:11,598 : INFO : Iteration 33500: with minibatch training loss = 0.45825809240341187
2019-02-24 17:03:21,245 : INFO : Iteration 34000: with minibatch training loss = 0.3882381319999695
2019-02-24 17:04:32,035 : INFO : Iteration 34500: with minibatch training loss = 0.4089142680168152
2019-02-24 17:05:43,410 : INFO : Iteration 35000: with minibatch training loss = 0.458892822265625
Epoch 20, Overall loss = 0.379
hit: 3332
HR%10: 0.03242538366469117
MAP%10: 0.1232614120151372
NDCG%10: 0.17567580497170382
2019-02-24 17:21:48,960 : INFO : Iteration 35500: with minibatch training loss = 0.4088948667049408
2019-02-24 17:22:59,111 : INFO : Iteration 36000: with minibatch training loss = 0.38906729221343994
2019-02-24 17:24:09,912 : INFO : Iteration 36500: with minibatch training loss = 0.5096989274024963
Epoch 21, Overall loss = 0.375
2019-02-24 17:25:20,515 : INFO : Iteration 37000: with minibatch training loss = 0.3659707009792328
2019-02-24 17:26:30,100 : INFO : Iteration 37500: with minibatch training loss = 0.31192973256111145
2019-02-24 17:27:40,801 : INFO : Iteration 38000: with minibatch training loss = 0.29405611753463745
2019-02-24 17:28:51,921 : INFO : Iteration 38500: with minibatch training loss = 0.33653461933135986
Epoch 22, Overall loss = 0.375
2019-02-24 17:30:01,571 : INFO : Iteration 39000: with minibatch training loss = 0.3889457583427429
2019-02-24 17:31:11,581 : INFO : Iteration 39500: with minibatch training loss = 0.6368078589439392
2019-02-24 17:32:22,659 : INFO : Iteration 40000: with minibatch training loss = 0.3573216497898102
Epoch 23, Overall loss = 0.373
2019-02-24 17:33:33,391 : INFO : Iteration 40500: with minibatch training loss = 0.2628760039806366
2019-02-24 17:34:42,910 : INFO : Iteration 41000: with minibatch training loss = 0.3163571357727051
2019-02-24 17:35:53,531 : INFO : Iteration 41500: with minibatch training loss = 0.3302443027496338
2019-02-24 17:37:04,671 : INFO : Iteration 42000: with minibatch training loss = 0.3239349126815796
Epoch 24, Overall loss = 0.372
2019-02-24 17:38:14,251 : INFO : Iteration 42500: with minibatch training loss = 0.3285844922065735
2019-02-24 17:39:24,159 : INFO : Iteration 43000: with minibatch training loss = 0.3857440948486328
2019-02-24 17:40:35,054 : INFO : Iteration 43500: with minibatch training loss = 0.29615288972854614
Epoch 25, Overall loss = 0.374
hit: 3330
HR%10: 0.03240592064928619
MAP%10: 0.12331134572114358
NDCG%10: 0.1753612089566809
2019-02-24 17:56:44,359 : INFO : Iteration 44000: with minibatch training loss = 0.3132302165031433
2019-02-24 17:57:53,958 : INFO : Iteration 44500: with minibatch training loss = 0.3828069567680359
2019-02-24 17:59:04,245 : INFO : Iteration 45000: with minibatch training loss = 0.3047141432762146
2019-02-24 18:00:15,442 : INFO : Iteration 45500: with minibatch training loss = 0.3696460723876953
Epoch 26, Overall loss = 0.37
2019-02-24 18:01:25,351 : INFO : Iteration 46000: with minibatch training loss = 0.4000228941440582
2019-02-24 18:02:35,308 : INFO : Iteration 46500: with minibatch training loss = 0.3475695848464966
2019-02-24 18:03:46,276 : INFO : Iteration 47000: with minibatch training loss = 0.4330185651779175
Epoch 27, Overall loss = 0.371
2019-02-24 18:04:56,926 : INFO : Iteration 47500: with minibatch training loss = 0.33744680881500244
2019-02-24 18:06:06,457 : INFO : Iteration 48000: with minibatch training loss = 0.31060025095939636
2019-02-24 18:07:16,883 : INFO : Iteration 48500: with minibatch training loss = 0.40795886516571045
2019-02-24 18:08:28,108 : INFO : Iteration 49000: with minibatch training loss = 0.36964988708496094
Epoch 28, Overall loss = 0.374
2019-02-24 18:09:37,727 : INFO : Iteration 49500: with minibatch training loss = 0.27569347620010376
2019-02-24 18:10:47,624 : INFO : Iteration 50000: with minibatch training loss = 0.3997970223426819
2019-02-24 18:11:58,413 : INFO : Iteration 50500: with minibatch training loss = 0.3598373532295227
Epoch 29, Overall loss = 0.372
2019-02-24 18:13:09,142 : INFO : Iteration 51000: with minibatch training loss = 0.4646188020706177
2019-02-24 18:14:18,677 : INFO : Iteration 51500: with minibatch training loss = 0.32982712984085083
2019-02-24 18:15:30,877 : INFO : Iteration 52000: with minibatch training loss = 0.37617555260658264
2019-02-24 18:16:42,540 : INFO : Iteration 52500: with minibatch training loss = 0.333675354719162
Epoch 30, Overall loss = 0.367
hit: 3373
HR%10: 0.03282437548049319
MAP%10: 0.12622995386221714
NDCG%10: 0.17820010775981837
2019-02-24 18:33:33,157 : INFO : Iteration 53000: with minibatch training loss = 0.2850351333618164
2019-02-24 18:34:43,864 : INFO : Iteration 53500: with minibatch training loss = 0.3750549554824829
2019-02-24 18:35:55,359 : INFO : Iteration 54000: with minibatch training loss = 0.4258989095687866
Epoch 31, Overall loss = 0.37
2019-02-24 18:37:06,745 : INFO : Iteration 54500: with minibatch training loss = 0.2623193860054016
2019-02-24 18:38:16,952 : INFO : Iteration 55000: with minibatch training loss = 0.3485430181026459
2019-02-24 18:39:27,835 : INFO : Iteration 55500: with minibatch training loss = 0.3285815119743347
2019-02-24 18:40:39,770 : INFO : Iteration 56000: with minibatch training loss = 0.3204154372215271
Epoch 32, Overall loss = 0.367
2019-02-24 18:41:50,036 : INFO : Iteration 56500: with minibatch training loss = 0.4627700448036194
2019-02-24 18:43:00,500 : INFO : Iteration 57000: with minibatch training loss = 0.32525816559791565
2019-02-24 18:44:11,937 : INFO : Iteration 57500: with minibatch training loss = 0.46915268898010254
Epoch 33, Overall loss = 0.37
2019-02-24 18:45:23,192 : INFO : Iteration 58000: with minibatch training loss = 0.3108735978603363
2019-02-24 18:46:33,121 : INFO : Iteration 58500: with minibatch training loss = 0.3383723795413971
2019-02-24 18:47:44,315 : INFO : Iteration 59000: with minibatch training loss = 0.40042805671691895
2019-02-24 18:48:56,077 : INFO : Iteration 59500: with minibatch training loss = 0.7021112442016602
Epoch 34, Overall loss = 0.366
2019-02-24 18:50:06,482 : INFO : Iteration 60000: with minibatch training loss = 0.33356156945228577
2019-02-24 18:51:17,039 : INFO : Iteration 60500: with minibatch training loss = 0.35709935426712036
2019-02-24 18:52:28,381 : INFO : Iteration 61000: with minibatch training loss = 0.33909231424331665
Epoch 35, Overall loss = 0.365
hit: 3333
HR%10: 0.03243511517239366
MAP%10: 0.12571778750131427
NDCG%10: 0.1777935088393452
2019-02-24 19:08:50,453 : INFO : Iteration 61500: with minibatch training loss = 0.37129735946655273
2019-02-24 19:10:00,522 : INFO : Iteration 62000: with minibatch training loss = 0.3170716166496277
2019-02-24 19:11:11,401 : INFO : Iteration 62500: with minibatch training loss = 0.3373425602912903
2019-02-24 19:12:23,385 : INFO : Iteration 63000: with minibatch training loss = 0.33820345997810364
Epoch 36, Overall loss = 0.366
2019-02-24 19:13:33,747 : INFO : Iteration 63500: with minibatch training loss = 0.3544052839279175
2019-02-24 19:14:44,256 : INFO : Iteration 64000: with minibatch training loss = 0.5235470533370972
2019-02-24 19:15:55,465 : INFO : Iteration 64500: with minibatch training loss = 0.3025522828102112
Epoch 37, Overall loss = 0.366
2019-02-24 19:17:06,843 : INFO : Iteration 65000: with minibatch training loss = 0.2671307325363159
2019-02-24 19:18:16,776 : INFO : Iteration 65500: with minibatch training loss = 0.3583720624446869
2019-02-24 19:19:27,914 : INFO : Iteration 66000: with minibatch training loss = 0.2816469669342041
2019-02-24 19:20:39,637 : INFO : Iteration 66500: with minibatch training loss = 0.30226874351501465
Epoch 38, Overall loss = 0.366
2019-02-24 19:21:49,952 : INFO : Iteration 67000: with minibatch training loss = 0.4812304377555847
2019-02-24 19:23:00,349 : INFO : Iteration 67500: with minibatch training loss = 0.5040313005447388
2019-02-24 19:24:11,763 : INFO : Iteration 68000: with minibatch training loss = 0.40066590905189514
Epoch 39, Overall loss = 0.365
2019-02-24 19:25:23,251 : INFO : Iteration 68500: with minibatch training loss = 0.37205639481544495
2019-02-24 19:26:33,061 : INFO : Iteration 69000: with minibatch training loss = 0.38341331481933594
2019-02-24 19:27:43,947 : INFO : Iteration 69500: with minibatch training loss = 0.4241176247596741
2019-02-24 19:28:55,669 : INFO : Iteration 70000: with minibatch training loss = 0.330785870552063
Epoch 40, Overall loss = 0.365
hit: 3385
HR%10: 0.03294115357292305
MAP%10: 0.12498649263789403
NDCG%10: 0.17838147552801428
```
即使添加多特征, 比NFM单特征差了不少  
  
2019/02/25  
今天放假  
看阿丽塔, 玩疯兔~  
  
2019/02/26  
开始学习django  
django框架内容真多, 规范复杂, 用的时候估计要大量查阅API文档

model:  
user: id, gender, ......  
movie: id, name, year, ......  
rating: rating_score, timestamp  
  
page:  
index  
login  
userinfo  
movieinfo -> score  
recommendpage  

最开始工作: 就把静态数据好好展示出来吧  

2019/02/27  
django自带的数据库sqlite3添加数据慢的不行...  
7000多条数据, 1分钟还没处理完...  
决定换mysql了   
  
数据载入完成  
现在在做views视图层和模板层内容  
  
视图层之后:  
连接各模板, 适当添加元素, 排版  
动态设置:1. 登录, 2. 添加新电影  
模型动态适应   
提高训练和输出结果的速度  
  
想不到做recommendpage时候遇到那么多麻烦  
1. 官方说有tensorflow_serving 部署工具使用, 但是文档很复杂, API接口不是很清晰, 不会用  
2. 之后使用simple_save代替, 调试成功了, 生成了文件, 还没有开始使用  
3. 想要模型提供一个借助user_id获取recommend_list的函数, 但是呢, 耦合很强, 不调整的话有很多重复代码片段, 耦合性非常差  
4. python import 的 绝对路径和相对路径在pycharm用的很迷, 不清楚  
我是用一个文件夹的Python文件调用另一个文件夹的python类, 这个类里有很多相对路径的open文件操作, 会失效, 不知道怎么操作...  
现在添加了很多硬解码, 不是很好的代码风格  
  
可以成功运行了, 但是模型被改的不成样了  
然而, 二次运行失败, 貌似重复调用函数会出问题...  
为什么每次推荐结果会变啊!  

2019/02/28  
二次运行失败问题解决了, 原因是重复定义tensor, 实际上应该直接调用保存的模型里的tensor 这里调用时出现点问题, 是输入X的sparseplaceholder, 对应有三个张量, 是输出所有的张量才找到的.  
问题:
1. tf.globall_variables_initializer()函数在创建会话都都会调用一次, 有什么确切的作用呢? 问什么我在调用保存的模型后在调用这个函数, 模型的参数会初始化, 输出结果是未训练的的结果呢?    
答: tf.globall_variables_initializer()  初始化所有的变量(tf.Variable值), 运行一次会把原来训练的参数全部擦掉.  
至于每次开始使用模型都调用一次, 也许是惯例吧?  
2. 我在测试模型时, dropout率仍设置为0.7, 此时输出结果会不确定.  

自己建了一个首页, 把所有模板通过超链接连接起来了, 然后开始把一个repo.里的index借鉴一下布局,元素等信息  
https://github.com/JaniceWuo/MovieRecommend  
过程中有很多坑没填,   
1. 比如一些django寻找文件路径的逻辑 views里render那个  
2. 比如静态文件一系列未知问题, index首页挪动很顺利, 甚至有点怀疑  
3. div布局问题, 一直想把登录信息和横栏合并成一行 但是没有成功  
4. bootstrap一大堆的class含义了, 这个很头疼  
  
我现在前端做的一头莽撞, 完全不看前面的路了, 我觉得有必要静下来好好思考下接下来应该怎么去做了  

2019/03/03  
最近在做关于毕设有关的主题阅读, 工程方面暂时搁置, 之后可能会开始FM模型的改进  , 直到AFM, 突然发现Attention机制并不难理解, 就像FFM一样, 多了个权重层  

2019/03/05  
夭寿啦， 加了神经网络跟没加一个loss  
训练集每一轮重排列时用  
sampler = np.random.permutation()  
***dataFrame.take(sampler)***  
下面这个函数有返回值， 返回重排列的dataFrame对象  
***dataFrame = dataFrame.take(sampler)***  
embedding变量竟然有两声明了两次。。。  
修改后：  
HR%10：0.036  
MAP：0.14  
NDCG：0.196  
看来代码还得改， 有些地方不够优化  
25轮后：  
MAP：0.152  
NDCG：0.213  

pick a bug：  in get random_block_from_data(data, batch_size):
~~item -= user~~  
之后每轮mean_loss变得稳定下降了  
但是指标并没有好转  
发现一个很神奇的现象  
在推荐时，用户用户越靠后， 用时越长， 这可能是因为rank比较高的电影用户都看过了  
修改一下placeholder（把sparse_placeholder舍去, 因为计算indices太耗时了）， 然后把IO放到全局里， 
500次batch_size=500的训练耗时从68秒 -> 49秒，  
训练128轮：  
HR%10:0.044  
MAP：0.168  
NDCG：0.235  
