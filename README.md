# DLForMovieRecommend

2019/1/18
1. 把NFM修改一下，用来排序，数据集也相应修改，代码能训练模型，但是loss没有优化的痕迹，有时候会出现nan
深度学习模型参数如果初始化太小，则信号会不断缩小，很难产生作用；如果初始化太大，则信号会不断增大溢出，而起不到作用
## nan问题解决了，是tf.log（x），x传入0或负值导致的！ 使用tf.clip_by_value()
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
Epoch 106 [1.0 s]	train=73.0699 [0.0 s]
Epoch 107 [1.0 s]	train=72.7068 [0.0 s]
Epoch 108 [1.0 s]	train=72.9609 [0.0 s]
Epoch 109 [1.0 s]	train=72.2712 [0.0 s]
Epoch 110 [1.0 s]	train=72.3913 [0.0 s]
Epoch 111 [1.1 s]	train=73.2733 [0.0 s]
Epoch 112 [1.0 s]	train=71.7241 [0.0 s]
Epoch 113 [1.0 s]	train=72.6757 [0.0 s]
Epoch 114 [1.0 s]	train=72.2408 [0.0 s]
Epoch 115 [1.0 s]	train=71.9488 [0.0 s]
Epoch 116 [1.0 s]	train=73.0491 [0.0 s]
Epoch 117 [1.0 s]	train=71.8514 [0.0 s]
Epoch 118 [1.0 s]	train=71.8056 [0.0 s]
Epoch 119 [1.0 s]	train=72.6396 [0.0 s]
Epoch 120 [1.0 s]	train=72.6693 [0.0 s]
Epoch 121 [1.0 s]	train=72.2280 [0.0 s]
Epoch 122 [1.0 s]	train=72.3941 [0.0 s]
Epoch 123 [1.0 s]	train=71.9410 [0.0 s]
Epoch 124 [1.0 s]	train=71.9701 [0.0 s]
Epoch 125 [1.0 s]	train=72.2148 [0.0 s]
Epoch 126 [1.0 s]	train=72.5405 [0.0 s]
Epoch 127 [1.0 s]	train=70.9800 [0.0 s]
Epoch 128 [1.0 s]	train=72.2523 [0.0 s]
Epoch 129 [1.0 s]	train=72.1033 [0.0 s]
Epoch 130 [1.0 s]	train=70.8989 [0.0 s]
Epoch 131 [1.0 s]	train=71.9899 [0.0 s]
Epoch 132 [1.0 s]	train=71.2936 [0.0 s]
Epoch 133 [1.0 s]	train=71.4436 [0.0 s]
Epoch 134 [1.0 s]	train=70.8989 [0.0 s]
Epoch 135 [1.0 s]	train=71.9852 [0.0 s]
Epoch 136 [1.0 s]	train=72.0954 [0.0 s]
Epoch 137 [1.0 s]	train=71.6978 [0.0 s]
Epoch 138 [1.0 s]	train=71.7354 [0.0 s]
Epoch 139 [1.0 s]	train=70.9941 [0.0 s]
Epoch 140 [1.0 s]	train=71.1497 [0.0 s]
Epoch 141 [1.0 s]	train=71.3723 [0.0 s]
Epoch 142 [1.0 s]	train=71.0961 [0.0 s]
Epoch 143 [1.0 s]	train=71.6775 [0.0 s]
Epoch 144 [1.0 s]	train=71.4976 [0.0 s]
Epoch 145 [1.0 s]	train=72.3618 [0.0 s]
Epoch 146 [1.0 s]	train=71.2665 [0.0 s]
Epoch 147 [1.0 s]	train=70.8665 [0.0 s]
Epoch 148 [1.0 s]	train=70.8661 [0.0 s]
Epoch 149 [1.0 s]	train=70.8430 [0.0 s]
Epoch 150 [1.0 s]	train=70.5474 [0.0 s]
Epoch 151 [1.0 s]	train=71.8018 [0.0 s]
Epoch 152 [1.0 s]	train=71.3872 [0.0 s]
Epoch 153 [1.0 s]	train=70.6838 [0.0 s]
Epoch 154 [1.0 s]	train=70.9943 [0.0 s]
Epoch 155 [1.0 s]	train=70.6662 [0.0 s]
Epoch 156 [1.0 s]	train=70.8126 [0.0 s]
Epoch 157 [1.0 s]	train=70.1630 [0.0 s]
Epoch 158 [1.0 s]	train=71.2039 [0.0 s]
Epoch 159 [1.0 s]	train=71.3449 [0.0 s]
Epoch 160 [1.0 s]	train=71.8078 [0.0 s]
Epoch 161 [1.0 s]	train=70.4316 [0.0 s]
Epoch 162 [1.0 s]	train=70.2784 [0.0 s]
Epoch 163 [1.0 s]	train=70.5749 [0.0 s]
Epoch 164 [1.0 s]	train=71.2819 [0.0 s]
Epoch 165 [1.0 s]	train=70.4606 [0.0 s]
Epoch 166 [1.0 s]	train=71.2941 [0.0 s]
Epoch 167 [1.0 s]	train=71.0975 [0.0 s]
Epoch 168 [1.0 s]	train=71.2107 [0.0 s]
Epoch 169 [1.0 s]	train=70.3371 [0.0 s]
Epoch 170 [1.0 s]	train=71.5218 [0.0 s]
Epoch 171 [1.0 s]	train=71.6593 [0.0 s]
Epoch 172 [1.0 s]	train=70.5695 [0.0 s]
Epoch 173 [1.0 s]	train=71.4510 [0.0 s]
Epoch 174 [1.0 s]	train=71.2299 [0.0 s]
Epoch 175 [1.0 s]	train=70.9030 [0.0 s]
Epoch 176 [1.0 s]	train=70.7032 [0.0 s]
Epoch 177 [1.0 s]	train=71.0982 [0.0 s]
Epoch 178 [1.0 s]	train=70.3946 [0.0 s]
Epoch 179 [1.0 s]	train=71.0427 [0.0 s]
Epoch 180 [1.0 s]	train=70.7150 [0.0 s]
Epoch 181 [1.0 s]	train=71.3239 [0.0 s]
Epoch 182 [1.0 s]	train=70.6611 [0.0 s]
Epoch 183 [1.0 s]	train=69.5403 [0.0 s]
Epoch 184 [1.0 s]	train=69.4318 [0.0 s]
Epoch 185 [1.0 s]	train=71.3700 [0.0 s]
Epoch 186 [1.0 s]	train=69.7830 [0.0 s]
Epoch 187 [1.0 s]	train=71.0100 [0.0 s]
Epoch 188 [1.0 s]	train=69.6277 [0.0 s]
Epoch 189 [1.0 s]	train=69.9747 [0.0 s]
Epoch 190 [1.0 s]	train=70.7797 [0.0 s]
Epoch 191 [1.0 s]	train=69.9362 [0.0 s]
Epoch 192 [1.0 s]	train=69.7065 [0.0 s]
Epoch 193 [1.0 s]	train=71.7374 [0.0 s]
Epoch 194 [1.0 s]	train=70.4705 [0.0 s]
Epoch 195 [1.0 s]	train=70.6360 [0.0 s]
Epoch 196 [1.0 s]	train=69.5973 [0.0 s]
Epoch 197 [1.0 s]	train=70.4445 [0.0 s]
Epoch 198 [1.0 s]	train=70.4439 [0.0 s]
Epoch 199 [1.0 s]	train=71.9017 [0.0 s]
Epoch 200 [1.0 s]	train=69.9690 [0.0 s]
recommend 50 user costs 0.6341965198516846s
recommend 50 user costs 0.6236233711242676s
recommend 50 user costs 0.5716707706451416s
recommend 50 user costs 0.5836808681488037s
recommend 50 user costs 0.5751357078552246s
recommend 50 user costs 0.576648473739624s
recommend 50 user costs 0.5876789093017578s
recommend 50 user costs 0.5756680965423584s
recommend 50 user costs 0.5776622295379639s
recommend 50 user costs 0.5766675472259521s
recommend 50 user costs 0.5826685428619385s
recommend 50 user costs 0.6485064029693604s
recommend 50 user costs 0.5796661376953125s
recommend 50 user costs 0.5955765247344971s
recommend 50 user costs 0.578648567199707s
recommend 50 user costs 0.5786776542663574s
recommend 50 user costs 0.5916478633880615s
recommend 50 user costs 0.6497936248779297s
hit: 742
HR%10: 0.07107960532618067
MAP%10: 0.1865292436050653
NDCG%10: 0.26499316554787167
Epoch 201 [1.0 s]	train=69.9581 [0.0 s]
Epoch 202 [1.0 s]	train=68.9443 [0.0 s]
Epoch 203 [1.0 s]	train=69.4402 [0.0 s]
Epoch 204 [1.0 s]	train=70.1583 [0.0 s]
Epoch 205 [1.0 s]	train=69.9216 [0.0 s]
Epoch 206 [1.0 s]	train=69.7558 [0.0 s]
Epoch 207 [1.0 s]	train=69.5330 [0.0 s]
Epoch 208 [1.0 s]	train=69.5150 [0.0 s]
Epoch 209 [1.0 s]	train=69.2961 [0.0 s]
Epoch 210 [1.0 s]	train=70.5595 [0.0 s]
Epoch 211 [1.0 s]	train=70.1214 [0.0 s]
Epoch 212 [1.0 s]	train=70.0925 [0.0 s]
Epoch 213 [1.0 s]	train=69.5396 [0.0 s]
Epoch 214 [1.0 s]	train=69.9033 [0.0 s]
Epoch 215 [1.0 s]	train=69.7709 [0.0 s]
Epoch 216 [1.0 s]	train=70.1000 [0.0 s]
Epoch 217 [1.0 s]	train=69.4150 [0.0 s]
Epoch 218 [1.0 s]	train=70.1594 [0.0 s]
Epoch 219 [1.0 s]	train=69.7355 [0.0 s]
Epoch 220 [1.0 s]	train=69.4916 [0.0 s]
Epoch 221 [1.0 s]	train=69.7096 [0.0 s]
Epoch 222 [1.0 s]	train=70.1569 [0.0 s]
Epoch 223 [1.0 s]	train=70.1141 [0.0 s]
Epoch 224 [1.0 s]	train=70.2845 [0.0 s]
Epoch 225 [1.0 s]	train=69.9781 [0.0 s]
Epoch 226 [1.0 s]	train=69.7148 [0.0 s]
Epoch 227 [1.0 s]	train=69.5460 [0.0 s]
Epoch 228 [1.0 s]	train=69.3121 [0.0 s]
Epoch 229 [1.0 s]	train=69.1182 [0.0 s]
Epoch 230 [1.0 s]	train=69.2191 [0.0 s]
Epoch 231 [1.0 s]	train=69.6768 [0.0 s]
Epoch 232 [1.0 s]	train=69.4167 [0.0 s]
Epoch 233 [1.0 s]	train=69.9554 [0.0 s]
Epoch 234 [1.0 s]	train=69.1501 [0.0 s]
Epoch 235 [1.0 s]	train=69.4346 [0.0 s]
Epoch 236 [1.0 s]	train=70.2248 [0.0 s]
Epoch 237 [1.0 s]	train=69.0426 [0.0 s]
Epoch 238 [1.0 s]	train=69.7101 [0.0 s]
Epoch 239 [1.0 s]	train=68.7910 [0.0 s]
Epoch 240 [1.0 s]	train=69.6189 [0.0 s]
Epoch 241 [1.0 s]	train=69.1627 [0.0 s]
Epoch 242 [1.0 s]	train=69.8809 [0.0 s]
Epoch 243 [1.0 s]	train=69.4972 [0.0 s]
Epoch 244 [1.0 s]	train=69.4487 [0.0 s]
Epoch 245 [1.0 s]	train=69.4335 [0.0 s]
Epoch 246 [1.0 s]	train=69.3334 [0.0 s]
Epoch 247 [1.0 s]	train=68.9366 [0.0 s]
Epoch 248 [1.0 s]	train=69.4682 [0.0 s]
Epoch 249 [1.0 s]	train=69.5974 [0.0 s]
Epoch 250 [1.0 s]	train=69.8262 [0.0 s]
Epoch 251 [1.0 s]	train=68.7995 [0.0 s]
Epoch 252 [1.0 s]	train=70.1170 [0.0 s]
Epoch 253 [1.0 s]	train=69.0956 [0.0 s]
Epoch 254 [1.0 s]	train=69.4273 [0.0 s]
Epoch 255 [1.0 s]	train=69.1638 [0.0 s]
Epoch 256 [1.0 s]	train=69.5309 [0.0 s]
Epoch 257 [1.0 s]	train=68.7178 [0.0 s]
Epoch 258 [1.0 s]	train=69.1224 [0.0 s]
Epoch 259 [1.0 s]	train=70.2858 [0.0 s]
Epoch 260 [1.0 s]	train=69.4111 [0.0 s]
Epoch 261 [1.0 s]	train=68.8103 [0.0 s]
Epoch 262 [1.0 s]	train=69.9627 [0.0 s]
Epoch 263 [1.0 s]	train=69.1161 [0.0 s]
Epoch 264 [1.0 s]	train=69.5878 [0.0 s]
Epoch 265 [1.0 s]	train=68.0306 [0.0 s]
Epoch 266 [1.0 s]	train=67.6265 [0.0 s]
Epoch 267 [1.0 s]	train=68.8559 [0.0 s]
Epoch 268 [1.1 s]	train=69.8901 [0.0 s]
Epoch 269 [1.0 s]	train=69.5293 [0.0 s]
Epoch 270 [1.0 s]	train=68.4605 [0.0 s]
Epoch 271 [1.0 s]	train=69.0558 [0.0 s]
Epoch 272 [1.0 s]	train=69.6331 [0.0 s]
Epoch 273 [1.0 s]	train=69.3375 [0.0 s]
Epoch 274 [1.0 s]	train=69.5498 [0.0 s]
Epoch 275 [1.0 s]	train=69.4821 [0.0 s]
Epoch 276 [1.0 s]	train=69.0454 [0.0 s]
Epoch 277 [1.0 s]	train=68.0312 [0.0 s]
Epoch 278 [1.0 s]	train=69.7748 [0.0 s]
Epoch 279 [1.0 s]	train=68.0108 [0.0 s]
Epoch 280 [1.0 s]	train=69.4114 [0.0 s]
Epoch 281 [1.0 s]	train=69.2626 [0.0 s]
Epoch 282 [1.0 s]	train=69.0558 [0.0 s]
Epoch 283 [1.0 s]	train=68.4948 [0.0 s]
Epoch 284 [1.0 s]	train=68.6202 [0.0 s]
Epoch 285 [1.0 s]	train=69.4580 [0.0 s]
Epoch 286 [1.0 s]	train=69.2813 [0.0 s]
Epoch 287 [1.0 s]	train=69.2163 [0.0 s]
Epoch 288 [1.0 s]	train=68.9029 [0.0 s]
Epoch 289 [1.0 s]	train=68.8898 [0.0 s]
Epoch 290 [1.0 s]	train=69.4434 [0.0 s]
Epoch 291 [1.0 s]	train=68.8832 [0.0 s]
Epoch 292 [1.0 s]	train=68.7993 [0.0 s]
Epoch 293 [1.0 s]	train=69.4002 [0.0 s]
Epoch 294 [1.0 s]	train=70.0417 [0.0 s]
Epoch 295 [1.0 s]	train=69.6589 [0.0 s]
Epoch 296 [1.0 s]	train=69.7626 [0.0 s]
Epoch 297 [1.0 s]	train=68.7675 [0.0 s]
Epoch 298 [1.0 s]	train=67.5607 [0.0 s]
Epoch 299 [1.0 s]	train=69.0381 [0.0 s]
Epoch 300 [1.0 s]	train=68.8177 [0.0 s]
recommend 50 user costs 0.5796782970428467s
recommend 50 user costs 0.6146326065063477s
recommend 50 user costs 0.5816645622253418s
recommend 50 user costs 0.6568758487701416s
recommend 50 user costs 0.5726718902587891s
recommend 50 user costs 0.5766739845275879s
recommend 50 user costs 0.5876603126525879s
recommend 50 user costs 0.5776667594909668s
recommend 50 user costs 0.5760111808776855s
recommend 50 user costs 0.5746862888336182s
recommend 50 user costs 0.5946571826934814s
recommend 50 user costs 0.5776622295379639s
recommend 50 user costs 0.5784854888916016s
recommend 50 user costs 0.5896470546722412s
recommend 50 user costs 0.6489748954772949s
recommend 50 user costs 0.5786840915679932s
recommend 50 user costs 0.5826592445373535s
recommend 50 user costs 0.5935661792755127s
hit: 721
HR%10: 0.06906791838298687
MAP%10: 0.1800129259315366
NDCG%10: 0.25573062695866483
Epoch 301 [1.0 s]	train=69.0226 [0.0 s]
Epoch 302 [1.0 s]	train=68.3051 [0.0 s]
Epoch 303 [1.0 s]	train=69.3493 [0.0 s]
Epoch 304 [1.0 s]	train=69.1646 [0.0 s]
Epoch 305 [1.0 s]	train=68.7869 [0.0 s]
Epoch 306 [1.0 s]	train=69.4619 [0.0 s]
Epoch 307 [1.0 s]	train=69.4099 [0.0 s]
Epoch 308 [1.0 s]	train=68.9632 [0.0 s]
Epoch 309 [1.0 s]	train=68.9443 [0.0 s]
Epoch 310 [1.0 s]	train=68.7740 [0.0 s]
Epoch 311 [1.0 s]	train=68.1582 [0.0 s]
Epoch 312 [1.0 s]	train=69.1493 [0.0 s]
Epoch 313 [1.0 s]	train=68.0452 [0.0 s]
Epoch 314 [1.1 s]	train=68.1608 [0.0 s]
Epoch 315 [1.1 s]	train=68.6419 [0.0 s]
Epoch 316 [1.0 s]	train=68.4641 [0.0 s]
Epoch 317 [1.1 s]	train=68.2468 [0.0 s]
Epoch 318 [1.0 s]	train=68.6771 [0.0 s]
Epoch 319 [1.0 s]	train=68.8769 [0.0 s]
Epoch 320 [1.0 s]	train=69.1228 [0.0 s]
Epoch 321 [1.0 s]	train=68.4725 [0.0 s]
Epoch 322 [1.0 s]	train=68.3249 [0.0 s]
Epoch 323 [1.0 s]	train=69.1886 [0.0 s]
Epoch 324 [1.0 s]	train=69.3269 [0.0 s]
Epoch 325 [1.0 s]	train=68.9090 [0.0 s]
Epoch 326 [1.0 s]	train=70.2720 [0.0 s]
Epoch 327 [1.0 s]	train=67.9508 [0.0 s]
Epoch 328 [1.0 s]	train=68.6271 [0.0 s]
Epoch 329 [1.0 s]	train=69.0238 [0.0 s]
Epoch 330 [1.0 s]	train=69.5472 [0.0 s]
Epoch 331 [1.0 s]	train=68.6012 [0.0 s]
Epoch 332 [1.0 s]	train=68.3538 [0.0 s]
Epoch 333 [1.0 s]	train=67.9159 [0.0 s]
Epoch 334 [1.0 s]	train=68.6549 [0.0 s]
Epoch 335 [1.0 s]	train=69.1390 [0.0 s]
Epoch 336 [1.0 s]	train=67.8460 [0.0 s]
Epoch 337 [1.0 s]	train=68.6636 [0.0 s]
Epoch 338 [1.1 s]	train=67.9738 [0.0 s]
Epoch 339 [1.0 s]	train=68.1446 [0.0 s]
Epoch 340 [1.0 s]	train=68.3421 [0.0 s]
Epoch 341 [1.0 s]	train=68.2246 [0.0 s]
Epoch 342 [1.0 s]	train=69.3941 [0.0 s]
Epoch 343 [1.0 s]	train=68.7223 [0.0 s]
Epoch 344 [1.0 s]	train=69.6110 [0.0 s]
Epoch 345 [1.0 s]	train=68.8247 [0.0 s]
Epoch 346 [1.0 s]	train=67.9654 [0.0 s]
Epoch 347 [1.0 s]	train=68.5626 [0.0 s]
Epoch 348 [1.0 s]	train=67.8179 [0.0 s]
Epoch 349 [1.0 s]	train=67.5439 [0.0 s]
Epoch 350 [1.0 s]	train=68.6263 [0.0 s]
Epoch 351 [1.0 s]	train=68.6871 [0.0 s]
Epoch 352 [1.0 s]	train=68.5480 [0.0 s]
Epoch 353 [1.0 s]	train=68.2045 [0.0 s]
Epoch 354 [1.0 s]	train=67.5007 [0.0 s]
Epoch 355 [1.0 s]	train=67.8108 [0.0 s]
Epoch 356 [1.0 s]	train=69.0296 [0.0 s]
Epoch 357 [1.0 s]	train=67.9252 [0.0 s]
Epoch 358 [1.0 s]	train=68.2275 [0.0 s]
Epoch 359 [1.0 s]	train=68.6775 [0.0 s]
Epoch 360 [1.0 s]	train=68.5002 [0.0 s]
Epoch 361 [1.0 s]	train=67.4821 [0.0 s]
Epoch 362 [1.0 s]	train=68.5450 [0.0 s]
Epoch 363 [1.0 s]	train=67.5105 [0.0 s]
Epoch 364 [1.0 s]	train=68.7225 [0.0 s]
Epoch 365 [1.0 s]	train=68.6487 [0.0 s]
Epoch 366 [1.0 s]	train=68.4107 [0.0 s]
Epoch 367 [1.0 s]	train=67.7225 [0.0 s]
Epoch 368 [1.0 s]	train=68.7284 [0.0 s]
Epoch 369 [1.0 s]	train=68.6195 [0.0 s]
Epoch 370 [1.0 s]	train=68.1605 [0.0 s]
Epoch 371 [1.0 s]	train=67.8707 [0.0 s]
Epoch 372 [1.0 s]	train=68.4450 [0.0 s]
Epoch 373 [1.0 s]	train=67.9781 [0.0 s]
Epoch 374 [1.0 s]	train=68.1257 [0.0 s]
Epoch 375 [1.0 s]	train=69.0223 [0.0 s]
Epoch 376 [1.0 s]	train=68.6486 [0.0 s]
Epoch 377 [1.0 s]	train=67.6079 [0.0 s]
Epoch 378 [1.0 s]	train=67.2728 [0.0 s]
Epoch 379 [1.0 s]	train=67.0925 [0.0 s]
Epoch 380 [1.0 s]	train=68.2510 [0.0 s]
Epoch 381 [1.0 s]	train=68.2666 [0.0 s]
Epoch 382 [1.0 s]	train=68.3818 [0.0 s]
Epoch 383 [1.0 s]	train=68.7502 [0.0 s]
Epoch 384 [1.0 s]	train=68.7237 [0.0 s]
Epoch 385 [1.0 s]	train=67.1327 [0.0 s]
Epoch 386 [1.0 s]	train=68.3056 [0.0 s]
Epoch 387 [1.0 s]	train=67.5871 [0.0 s]
Epoch 388 [1.0 s]	train=68.2830 [0.0 s]
Epoch 389 [1.0 s]	train=68.1835 [0.0 s]
Epoch 390 [1.0 s]	train=68.7481 [0.0 s]
Epoch 391 [1.0 s]	train=68.1882 [0.0 s]
Epoch 392 [1.0 s]	train=68.8949 [0.0 s]
Epoch 393 [1.0 s]	train=68.5711 [0.0 s]
Epoch 394 [1.0 s]	train=68.3909 [0.0 s]
Epoch 395 [1.0 s]	train=69.1485 [0.0 s]
Epoch 396 [1.0 s]	train=68.6386 [0.0 s]
Epoch 397 [1.0 s]	train=66.8380 [0.0 s]
Epoch 398 [1.0 s]	train=68.5018 [0.0 s]
Epoch 399 [1.0 s]	train=68.0287 [0.0 s]
Epoch 400 [1.0 s]	train=68.0670 [0.0 s]
recommend 50 user costs 0.6519765853881836s
recommend 50 user costs 0.6126465797424316s
recommend 50 user costs 0.5996544361114502s
recommend 50 user costs 0.5755410194396973s
recommend 50 user costs 0.5766727924346924s
recommend 50 user costs 0.5786490440368652s
recommend 50 user costs 0.6494452953338623s
recommend 50 user costs 0.5805966854095459s
recommend 50 user costs 0.5756680965423584s
recommend 50 user costs 0.5856575965881348s
recommend 50 user costs 0.5796785354614258s
recommend 50 user costs 0.5806524753570557s
recommend 50 user costs 0.5876739025115967s
recommend 50 user costs 0.5746641159057617s
recommend 50 user costs 0.5803711414337158s
recommend 50 user costs 0.5776798725128174s
recommend 50 user costs 0.5936577320098877s
recommend 50 user costs 0.6556098461151123s
hit: 758
HR%10: 0.07261231918766166
MAP%10: 0.19637696939857596
NDCG%10: 0.27805293291369454
Epoch 401 [1.0 s]	train=67.1450 [0.0 s]
Epoch 402 [1.0 s]	train=68.9109 [0.0 s]
Epoch 403 [1.0 s]	train=68.1443 [0.0 s]
Epoch 404 [1.0 s]	train=67.6448 [0.0 s]
Epoch 405 [1.0 s]	train=67.1263 [0.0 s]
Epoch 406 [1.0 s]	train=67.8707 [0.0 s]
Epoch 407 [1.0 s]	train=68.4885 [0.0 s]
Epoch 408 [1.0 s]	train=67.8397 [0.0 s]
Epoch 409 [1.0 s]	train=67.6314 [0.0 s]
Epoch 410 [1.1 s]	train=67.9311 [0.0 s]
Epoch 411 [1.0 s]	train=67.3799 [0.0 s]
Epoch 412 [1.0 s]	train=67.9180 [0.0 s]
Epoch 413 [1.0 s]	train=67.6723 [0.0 s]
Epoch 414 [1.0 s]	train=68.3638 [0.0 s]
Epoch 415 [1.0 s]	train=68.1621 [0.0 s]
Epoch 416 [1.0 s]	train=67.0866 [0.0 s]
Epoch 417 [1.0 s]	train=68.4767 [0.0 s]
Epoch 418 [1.0 s]	train=67.9578 [0.0 s]
Epoch 419 [1.0 s]	train=67.9925 [0.0 s]
Epoch 420 [1.0 s]	train=66.6397 [0.0 s]
Epoch 421 [1.0 s]	train=68.0160 [0.0 s]
Epoch 422 [1.0 s]	train=67.2118 [0.0 s]
Epoch 423 [1.0 s]	train=68.7342 [0.0 s]
Epoch 424 [1.0 s]	train=66.9170 [0.0 s]
Epoch 425 [1.0 s]	train=67.5101 [0.0 s]
Epoch 426 [1.0 s]	train=67.5023 [0.0 s]
Epoch 427 [1.0 s]	train=67.9598 [0.0 s]
Epoch 428 [1.0 s]	train=69.1171 [0.0 s]
Epoch 429 [1.0 s]	train=67.4805 [0.0 s]
Epoch 430 [1.0 s]	train=66.9624 [0.0 s]
Epoch 431 [1.0 s]	train=67.6102 [0.0 s]
Epoch 432 [1.0 s]	train=67.6925 [0.0 s]
Epoch 433 [1.0 s]	train=67.4087 [0.0 s]
Epoch 434 [1.0 s]	train=67.4518 [0.0 s]
Epoch 435 [1.0 s]	train=67.2134 [0.0 s]
Epoch 436 [1.0 s]	train=68.8512 [0.0 s]
Epoch 437 [1.0 s]	train=67.6621 [0.0 s]
Epoch 438 [1.0 s]	train=67.5693 [0.0 s]
Epoch 439 [1.0 s]	train=67.3264 [0.0 s]
Epoch 440 [1.0 s]	train=67.9843 [0.0 s]
Epoch 441 [1.0 s]	train=67.9812 [0.0 s]
Epoch 442 [1.0 s]	train=66.7357 [0.0 s]
Epoch 443 [1.0 s]	train=67.4946 [0.0 s]
Epoch 444 [1.0 s]	train=67.1122 [0.0 s]
Epoch 445 [1.0 s]	train=67.4417 [0.0 s]
Epoch 446 [1.0 s]	train=66.9158 [0.0 s]
Epoch 447 [1.0 s]	train=67.2220 [0.0 s]
Epoch 448 [1.0 s]	train=67.9887 [0.0 s]
Epoch 449 [1.0 s]	train=66.8887 [0.0 s]
Epoch 450 [1.0 s]	train=67.3949 [0.0 s]
Epoch 451 [1.0 s]	train=67.3745 [0.0 s]
Epoch 452 [1.0 s]	train=67.4084 [0.0 s]
Epoch 453 [1.0 s]	train=67.5922 [0.0 s]
Epoch 454 [1.0 s]	train=68.3295 [0.0 s]
Epoch 455 [1.0 s]	train=67.8693 [0.0 s]
Epoch 456 [1.0 s]	train=67.0246 [0.0 s]
Epoch 457 [1.0 s]	train=67.7587 [0.0 s]
Epoch 458 [1.0 s]	train=68.5911 [0.0 s]
Epoch 459 [1.0 s]	train=67.1795 [0.0 s]
Epoch 460 [1.0 s]	train=67.6139 [0.0 s]
Epoch 461 [1.0 s]	train=68.0370 [0.0 s]
Epoch 462 [1.0 s]	train=67.9230 [0.0 s]
Epoch 463 [1.0 s]	train=66.7959 [0.0 s]
Epoch 464 [1.0 s]	train=67.1217 [0.0 s]
Epoch 465 [1.0 s]	train=66.5069 [0.0 s]
Epoch 466 [1.0 s]	train=67.2166 [0.0 s]
Epoch 467 [1.0 s]	train=67.4110 [0.0 s]
Epoch 468 [1.0 s]	train=68.1065 [0.0 s]
Epoch 469 [1.0 s]	train=66.8079 [0.0 s]
Epoch 470 [1.0 s]	train=67.3981 [0.0 s]
Epoch 471 [1.0 s]	train=67.4822 [0.0 s]
Epoch 472 [1.0 s]	train=67.7990 [0.0 s]
Epoch 473 [1.0 s]	train=66.6892 [0.0 s]
Epoch 474 [1.0 s]	train=67.7000 [0.0 s]
Epoch 475 [1.0 s]	train=67.5990 [0.0 s]
Epoch 476 [1.0 s]	train=68.0336 [0.0 s]
Epoch 477 [1.0 s]	train=66.4869 [0.0 s]
Epoch 478 [1.0 s]	train=67.8283 [0.0 s]
Epoch 479 [1.0 s]	train=66.9230 [0.0 s]
Epoch 480 [1.0 s]	train=67.5011 [0.0 s]
Epoch 481 [1.0 s]	train=68.7760 [0.0 s]
Epoch 482 [1.0 s]	train=67.2152 [0.0 s]
Epoch 483 [1.0 s]	train=67.5274 [0.0 s]
Epoch 484 [1.0 s]	train=67.1409 [0.0 s]
Epoch 485 [1.0 s]	train=67.6648 [0.0 s]
Epoch 486 [1.0 s]	train=68.1928 [0.0 s]
Epoch 487 [1.0 s]	train=67.6308 [0.0 s]
Epoch 488 [1.0 s]	train=66.3560 [0.0 s]
Epoch 489 [1.0 s]	train=67.1652 [0.0 s]
Epoch 490 [1.0 s]	train=67.2179 [0.0 s]
Epoch 491 [1.0 s]	train=66.8418 [0.0 s]
Epoch 492 [1.0 s]	train=67.2019 [0.0 s]
Epoch 493 [1.0 s]	train=67.3217 [0.0 s]
Epoch 494 [1.0 s]	train=67.1455 [0.0 s]
Epoch 495 [1.0 s]	train=67.9477 [0.0 s]
Epoch 496 [1.0 s]	train=67.3458 [0.0 s]
Epoch 497 [1.0 s]	train=66.4748 [0.0 s]
Epoch 498 [1.0 s]	train=67.2107 [0.0 s]
Epoch 499 [1.0 s]	train=68.0711 [0.0 s]
Epoch 500 [1.0 s]	train=68.6530 [0.0 s]
recommend 50 user costs 0.5873386859893799s
recommend 50 user costs 0.6173834800720215s
recommend 50 user costs 0.5848650932312012s
recommend 50 user costs 0.5756747722625732s
recommend 50 user costs 0.579648494720459s
recommend 50 user costs 0.5772531032562256s
recommend 50 user costs 0.6500864028930664s
recommend 50 user costs 0.5803072452545166s
recommend 50 user costs 0.5906767845153809s
recommend 50 user costs 0.5796613693237305s
recommend 50 user costs 0.5816519260406494s
recommend 50 user costs 0.6576337814331055s
recommend 50 user costs 0.5786411762237549s
recommend 50 user costs 0.5796830654144287s
recommend 50 user costs 0.5806610584259033s
recommend 50 user costs 0.5902712345123291s
recommend 50 user costs 0.5809328556060791s
recommend 50 user costs 0.5778892040252686s
hit: 761
HR%10: 0.07289970303668934
MAP%10: 0.18834946144122777
NDCG%10: 0.2719184388943225
Epoch 501 [1.0 s]	train=67.4729 [0.0 s]
Epoch 502 [1.0 s]	train=67.4591 [0.0 s]
Epoch 503 [1.0 s]	train=66.8133 [0.0 s]
Epoch 504 [1.0 s]	train=66.7627 [0.0 s]
Epoch 505 [1.0 s]	train=67.7189 [0.0 s]
Epoch 506 [1.0 s]	train=66.9886 [0.0 s]
Epoch 507 [1.0 s]	train=67.9674 [0.0 s]
Epoch 508 [1.0 s]	train=67.4598 [0.0 s]
Epoch 509 [1.0 s]	train=67.9109 [0.0 s]
Epoch 510 [1.0 s]	train=68.2991 [0.0 s]
Epoch 511 [1.0 s]	train=67.6796 [0.0 s]
Epoch 512 [1.0 s]	train=68.5452 [0.0 s]
Epoch 513 [1.0 s]	train=67.1788 [0.0 s]
Epoch 514 [1.0 s]	train=66.9768 [0.0 s]
Epoch 515 [1.0 s]	train=66.4570 [0.0 s]
Epoch 516 [1.0 s]	train=68.3783 [0.0 s]
Epoch 517 [1.0 s]	train=67.6031 [0.0 s]
Epoch 518 [1.0 s]	train=66.6456 [0.0 s]
Epoch 519 [1.0 s]	train=67.3284 [0.0 s]
Epoch 520 [1.0 s]	train=67.0513 [0.0 s]
Epoch 521 [1.0 s]	train=67.2679 [0.0 s]
Epoch 522 [1.0 s]	train=66.4284 [0.0 s]
Epoch 523 [1.0 s]	train=66.6860 [0.0 s]
Epoch 524 [1.0 s]	train=67.3892 [0.0 s]
Epoch 525 [1.0 s]	train=67.3543 [0.0 s]
Epoch 526 [1.0 s]	train=67.1883 [0.0 s]
Epoch 527 [1.0 s]	train=67.1094 [0.0 s]
Epoch 528 [1.0 s]	train=66.8018 [0.0 s]
Epoch 529 [1.0 s]	train=67.4452 [0.0 s]
Epoch 530 [1.0 s]	train=67.1139 [0.0 s]
Epoch 531 [1.0 s]	train=67.3930 [0.0 s]
Epoch 532 [1.0 s]	train=68.0037 [0.0 s]
Epoch 533 [1.0 s]	train=67.2379 [0.0 s]
Epoch 534 [1.0 s]	train=67.1939 [0.0 s]
Epoch 535 [1.0 s]	train=67.1968 [0.0 s]
Epoch 536 [1.0 s]	train=67.4237 [0.0 s]
Epoch 537 [1.0 s]	train=67.5120 [0.0 s]
Epoch 538 [1.0 s]	train=67.2097 [0.0 s]
Epoch 539 [1.0 s]	train=66.6620 [0.0 s]
Epoch 540 [1.0 s]	train=66.1614 [0.0 s]
Epoch 541 [1.0 s]	train=67.2107 [0.0 s]
Epoch 542 [1.0 s]	train=67.1497 [0.0 s]
Epoch 543 [1.0 s]	train=67.0051 [0.0 s]
Epoch 544 [1.0 s]	train=68.2370 [0.0 s]
Epoch 545 [1.0 s]	train=66.7520 [0.0 s]
Epoch 546 [1.0 s]	train=66.4319 [0.0 s]
Epoch 547 [1.0 s]	train=66.7797 [0.0 s]
Epoch 548 [1.0 s]	train=67.4773 [0.0 s]
Epoch 549 [1.0 s]	train=67.0581 [0.0 s]
Epoch 550 [1.0 s]	train=66.3832 [0.0 s]
Epoch 551 [1.0 s]	train=66.6209 [0.0 s]
Epoch 552 [1.0 s]	train=67.2985 [0.0 s]
Epoch 553 [1.0 s]	train=66.7494 [0.0 s]
Epoch 554 [1.0 s]	train=67.1704 [0.0 s]
Epoch 555 [1.0 s]	train=66.6614 [0.0 s]
Epoch 556 [1.0 s]	train=66.0814 [0.0 s]
Epoch 557 [1.0 s]	train=67.4763 [0.0 s]
Epoch 558 [1.0 s]	train=66.9541 [0.0 s]
Epoch 559 [1.0 s]	train=67.6010 [0.0 s]
Epoch 560 [1.0 s]	train=66.7533 [0.0 s]
Epoch 561 [1.0 s]	train=66.3448 [0.0 s]
Epoch 562 [1.0 s]	train=66.8378 [0.0 s]
Epoch 563 [1.0 s]	train=67.5210 [0.0 s]
Epoch 564 [1.0 s]	train=66.3537 [0.0 s]
Epoch 565 [1.0 s]	train=66.1961 [0.0 s]
Epoch 566 [1.0 s]	train=67.1991 [0.0 s]
Epoch 567 [1.0 s]	train=67.1398 [0.0 s]
Epoch 568 [1.0 s]	train=66.9666 [0.0 s]
Epoch 569 [1.0 s]	train=66.5726 [0.0 s]
Epoch 570 [1.0 s]	train=67.2833 [0.0 s]
Epoch 571 [1.0 s]	train=67.2129 [0.0 s]
Epoch 572 [1.0 s]	train=65.9848 [0.0 s]
Epoch 573 [1.0 s]	train=66.5655 [0.0 s]
Epoch 574 [1.0 s]	train=67.0949 [0.0 s]
Epoch 575 [1.0 s]	train=66.5090 [0.0 s]
Epoch 576 [1.0 s]	train=66.7546 [0.0 s]
Epoch 577 [1.0 s]	train=66.0620 [0.0 s]
Epoch 578 [1.0 s]	train=66.8492 [0.0 s]
Epoch 579 [1.0 s]	train=67.9357 [0.0 s]
Epoch 580 [1.0 s]	train=66.1977 [0.0 s]
Epoch 581 [1.0 s]	train=66.6481 [0.0 s]
Epoch 582 [1.0 s]	train=66.7615 [0.0 s]
Epoch 583 [1.0 s]	train=67.0685 [0.0 s]
Epoch 584 [1.0 s]	train=66.5322 [0.0 s]
Epoch 585 [1.0 s]	train=66.4882 [0.0 s]
Epoch 586 [1.0 s]	train=68.0157 [0.0 s]
Epoch 587 [1.0 s]	train=67.4405 [0.0 s]
Epoch 588 [1.0 s]	train=66.1625 [0.0 s]
Epoch 589 [1.0 s]	train=66.6072 [0.0 s]
Epoch 590 [1.0 s]	train=66.9149 [0.0 s]
Epoch 591 [1.0 s]	train=66.0852 [0.0 s]
Epoch 592 [1.0 s]	train=67.2816 [0.0 s]
Epoch 593 [1.0 s]	train=66.7237 [0.0 s]
Epoch 594 [1.0 s]	train=66.7860 [0.0 s]
Epoch 595 [1.0 s]	train=67.1368 [0.0 s]
Epoch 596 [1.0 s]	train=66.5867 [0.0 s]
Epoch 597 [1.0 s]	train=66.6349 [0.0 s]
Epoch 598 [1.0 s]	train=66.4774 [0.0 s]
Epoch 599 [1.0 s]	train=66.9327 [0.0 s]
Epoch 600 [1.0 s]	train=66.0474 [0.0 s]
recommend 50 user costs 0.5664350986480713s
recommend 50 user costs 0.697596549987793s
recommend 50 user costs 0.5856616497039795s
recommend 50 user costs 0.5768461227416992s
recommend 50 user costs 0.5838897228240967s
recommend 50 user costs 0.5783853530883789s
recommend 50 user costs 0.5756556987762451s
recommend 50 user costs 0.575681209564209s
recommend 50 user costs 0.5846624374389648s
recommend 50 user costs 0.5806660652160645s
recommend 50 user costs 0.5786707401275635s
recommend 50 user costs 0.6611793041229248s
recommend 50 user costs 0.5726730823516846s
recommend 50 user costs 0.5734145641326904s
recommend 50 user costs 0.5793750286102295s
recommend 50 user costs 0.5796477794647217s
recommend 50 user costs 0.5796656608581543s
recommend 50 user costs 0.6411094665527344s
hit: 764
HR%10: 0.07318708688571703
MAP%10: 0.18821751093509148
NDCG%10: 0.26857222407793463
Epoch 601 [1.0 s]	train=66.2261 [0.0 s]
Epoch 602 [1.0 s]	train=65.6836 [0.0 s]
Epoch 603 [1.0 s]	train=67.2182 [0.0 s]
Epoch 604 [1.0 s]	train=66.8793 [0.0 s]
Epoch 605 [1.0 s]	train=66.8953 [0.0 s]
Epoch 606 [1.0 s]	train=66.3975 [0.0 s]
Epoch 607 [1.0 s]	train=67.3584 [0.0 s]
Epoch 608 [1.0 s]	train=66.2898 [0.0 s]
Epoch 609 [1.0 s]	train=67.0064 [0.0 s]
Epoch 610 [1.0 s]	train=66.6960 [0.0 s]
Epoch 611 [1.0 s]	train=65.4434 [0.0 s]
Epoch 612 [1.0 s]	train=66.2128 [0.0 s]
Epoch 613 [1.0 s]	train=67.7780 [0.0 s]
Epoch 614 [1.0 s]	train=66.4351 [0.0 s]
Epoch 615 [1.0 s]	train=66.1865 [0.0 s]
Epoch 616 [1.0 s]	train=66.8915 [0.0 s]
Epoch 617 [1.0 s]	train=67.3212 [0.0 s]
Epoch 618 [1.0 s]	train=66.9656 [0.0 s]
Epoch 619 [1.0 s]	train=66.7066 [0.0 s]
Epoch 620 [1.0 s]	train=66.2535 [0.0 s]
Epoch 621 [1.0 s]	train=66.4104 [0.0 s]
Epoch 622 [1.0 s]	train=67.4852 [0.0 s]
Epoch 623 [1.0 s]	train=66.1843 [0.0 s]
Epoch 624 [1.0 s]	train=65.8380 [0.0 s]
Epoch 625 [1.0 s]	train=66.9198 [0.0 s]
Epoch 626 [1.0 s]	train=66.1455 [0.0 s]
Epoch 627 [1.0 s]	train=66.3360 [0.0 s]
Epoch 628 [1.0 s]	train=66.8439 [0.0 s]
Epoch 629 [1.0 s]	train=67.9409 [0.0 s]
Epoch 630 [1.0 s]	train=66.6397 [0.0 s]
Epoch 631 [1.0 s]	train=66.6363 [0.0 s]
Epoch 632 [1.0 s]	train=65.8952 [0.0 s]
Epoch 633 [1.0 s]	train=66.2341 [0.0 s]
Epoch 634 [1.0 s]	train=65.5118 [0.0 s]
Epoch 635 [1.0 s]	train=66.9806 [0.0 s]
Epoch 636 [1.0 s]	train=67.1812 [0.0 s]
Epoch 637 [1.0 s]	train=66.5528 [0.0 s]
Epoch 638 [1.0 s]	train=66.5778 [0.0 s]
Epoch 639 [1.0 s]	train=66.4542 [0.0 s]
Epoch 640 [1.0 s]	train=65.9991 [0.0 s]
Epoch 641 [1.0 s]	train=67.3088 [0.0 s]
Epoch 642 [1.0 s]	train=66.2933 [0.0 s]
Epoch 643 [1.0 s]	train=66.4729 [0.0 s]
Epoch 644 [1.0 s]	train=66.3939 [0.0 s]
Epoch 645 [1.0 s]	train=67.4683 [0.0 s]
Epoch 646 [1.0 s]	train=66.3412 [0.0 s]
Epoch 647 [1.0 s]	train=67.1035 [0.0 s]
Epoch 648 [1.0 s]	train=67.1534 [0.0 s]
Epoch 649 [1.0 s]	train=66.5910 [0.0 s]
Epoch 650 [1.0 s]	train=66.0154 [0.0 s]
Epoch 651 [1.0 s]	train=65.9443 [0.0 s]
Epoch 652 [1.0 s]	train=65.7870 [0.0 s]
Epoch 653 [1.0 s]	train=65.7473 [0.0 s]
Epoch 654 [1.0 s]	train=65.7286 [0.0 s]
Epoch 655 [1.0 s]	train=66.6764 [0.0 s]
Epoch 656 [1.0 s]	train=65.5500 [0.0 s]
Epoch 657 [1.0 s]	train=66.9364 [0.0 s]
Epoch 658 [1.0 s]	train=67.1331 [0.0 s]
Epoch 659 [1.0 s]	train=67.3523 [0.0 s]
Epoch 660 [1.0 s]	train=66.7924 [0.0 s]
Epoch 661 [1.0 s]	train=66.2641 [0.0 s]
Epoch 662 [1.0 s]	train=67.1917 [0.0 s]
Epoch 663 [1.0 s]	train=66.2449 [0.0 s]
Epoch 664 [1.0 s]	train=65.7288 [0.0 s]
Epoch 665 [1.0 s]	train=67.3768 [0.0 s]
Epoch 666 [1.0 s]	train=66.8664 [0.0 s]
Epoch 667 [1.0 s]	train=66.9174 [0.0 s]
Epoch 668 [1.0 s]	train=66.3220 [0.0 s]
Epoch 669 [1.0 s]	train=66.1866 [0.0 s]
Epoch 670 [1.0 s]	train=66.1132 [0.0 s]
Epoch 671 [1.0 s]	train=66.3464 [0.0 s]
Epoch 672 [1.0 s]	train=66.8507 [0.0 s]
Epoch 673 [1.0 s]	train=66.1176 [0.0 s]
Epoch 674 [1.0 s]	train=66.9205 [0.0 s]
Epoch 675 [1.0 s]	train=67.0912 [0.0 s]
Epoch 676 [1.0 s]	train=65.8922 [0.0 s]
Epoch 677 [1.0 s]	train=66.5613 [0.0 s]
Epoch 678 [1.0 s]	train=66.6971 [0.0 s]
Epoch 679 [1.0 s]	train=66.5936 [0.0 s]
Epoch 680 [1.0 s]	train=66.0952 [0.0 s]
Epoch 681 [1.0 s]	train=65.7850 [0.0 s]
Epoch 682 [1.0 s]	train=66.6609 [0.0 s]
Epoch 683 [1.0 s]	train=66.4797 [0.0 s]
Epoch 684 [1.0 s]	train=66.7573 [0.0 s]
Epoch 685 [1.0 s]	train=66.3366 [0.0 s]
Epoch 686 [1.0 s]	train=65.9462 [0.0 s]
Epoch 687 [1.0 s]	train=65.6697 [0.0 s]
Epoch 688 [1.0 s]	train=67.0354 [0.0 s]
Epoch 689 [1.0 s]	train=66.3153 [0.0 s]
Epoch 690 [1.0 s]	train=66.6586 [0.0 s]
Epoch 691 [1.0 s]	train=66.7175 [0.0 s]
Epoch 692 [1.0 s]	train=66.4489 [0.0 s]
Epoch 693 [1.0 s]	train=66.3611 [0.0 s]
Epoch 694 [1.0 s]	train=67.1851 [0.0 s]
Epoch 695 [1.0 s]	train=66.0273 [0.0 s]
Epoch 696 [1.0 s]	train=65.5081 [0.0 s]
Epoch 697 [1.0 s]	train=66.3461 [0.0 s]
Epoch 698 [1.0 s]	train=66.1917 [0.0 s]
Epoch 699 [1.0 s]	train=66.7148 [0.0 s]
Epoch 700 [1.0 s]	train=66.1175 [0.0 s]
recommend 50 user costs 0.567842960357666s
recommend 50 user costs 0.6236414909362793s
recommend 50 user costs 0.5826640129089355s
recommend 50 user costs 0.576671838760376s
recommend 50 user costs 0.6554620265960693s
recommend 50 user costs 0.5736515522003174s
recommend 50 user costs 0.5736691951751709s
recommend 50 user costs 0.5749306678771973s
recommend 50 user costs 0.5836634635925293s
recommend 50 user costs 0.5746819972991943s
recommend 50 user costs 0.5736746788024902s
recommend 50 user costs 0.5998721122741699s
recommend 50 user costs 0.5782556533813477s
recommend 50 user costs 0.5750737190246582s
recommend 50 user costs 0.6500961780548096s
recommend 50 user costs 0.5856485366821289s
recommend 50 user costs 0.5778985023498535s
recommend 50 user costs 0.5776667594909668s
hit: 751
HR%10: 0.07194175687326372
MAP%10: 0.1806530355750169
NDCG%10: 0.26075683195044486
Epoch 701 [1.0 s]	train=66.3937 [0.0 s]
Epoch 702 [1.0 s]	train=66.4533 [0.0 s]
Epoch 703 [1.0 s]	train=65.8703 [0.0 s]
Epoch 704 [1.0 s]	train=65.6265 [0.0 s]
Epoch 705 [1.0 s]	train=66.1889 [0.0 s]
Epoch 706 [1.0 s]	train=66.2079 [0.0 s]
Epoch 707 [1.0 s]	train=66.5736 [0.0 s]
Epoch 708 [1.0 s]	train=65.8329 [0.0 s]
Epoch 709 [1.0 s]	train=67.5252 [0.0 s]
Epoch 710 [1.1 s]	train=66.1042 [0.0 s]
Epoch 711 [1.0 s]	train=66.3757 [0.0 s]
Epoch 712 [1.0 s]	train=65.9897 [0.0 s]
Epoch 713 [1.1 s]	train=66.5872 [0.0 s]
Epoch 714 [1.1 s]	train=65.6354 [0.0 s]
Epoch 715 [1.0 s]	train=66.3597 [0.0 s]
Epoch 716 [1.0 s]	train=65.5434 [0.0 s]
Epoch 717 [1.0 s]	train=66.7949 [0.0 s]
Epoch 718 [1.0 s]	train=66.4844 [0.0 s]
Epoch 719 [1.0 s]	train=65.3071 [0.0 s]
Epoch 720 [1.0 s]	train=67.1739 [0.0 s]
Epoch 721 [1.0 s]	train=66.6943 [0.0 s]
Epoch 722 [1.0 s]	train=66.0613 [0.0 s]
Epoch 723 [1.0 s]	train=65.2189 [0.0 s]
Epoch 724 [1.0 s]	train=66.7915 [0.0 s]
Epoch 725 [1.0 s]	train=66.2365 [0.0 s]
Epoch 726 [1.0 s]	train=65.6711 [0.0 s]
Epoch 727 [1.0 s]	train=66.8601 [0.0 s]
Epoch 728 [1.0 s]	train=66.5727 [0.0 s]
Epoch 729 [1.0 s]	train=66.3985 [0.0 s]
Epoch 730 [1.0 s]	train=66.3685 [0.0 s]
Epoch 731 [1.0 s]	train=65.1996 [0.0 s]
Epoch 732 [1.0 s]	train=65.8510 [0.0 s]
Epoch 733 [1.0 s]	train=66.3322 [0.0 s]
Epoch 734 [1.1 s]	train=66.0407 [0.0 s]
Epoch 735 [1.0 s]	train=65.6705 [0.0 s]
Epoch 736 [1.0 s]	train=66.8563 [0.0 s]
Epoch 737 [1.0 s]	train=66.2682 [0.0 s]
Epoch 738 [1.0 s]	train=65.1211 [0.0 s]
Epoch 739 [1.0 s]	train=66.8544 [0.0 s]
Epoch 740 [1.0 s]	train=64.8876 [0.0 s]
Epoch 741 [1.0 s]	train=67.3435 [0.0 s]
Epoch 742 [1.0 s]	train=66.7689 [0.0 s]
Epoch 743 [1.0 s]	train=66.0016 [0.0 s]
Epoch 744 [1.0 s]	train=65.8220 [0.0 s]
Epoch 745 [1.0 s]	train=66.7743 [0.0 s]
Epoch 746 [1.0 s]	train=66.1518 [0.0 s]
Epoch 747 [1.0 s]	train=65.3714 [0.0 s]
Epoch 748 [1.0 s]	train=65.9222 [0.0 s]
Epoch 749 [1.0 s]	train=66.4685 [0.0 s]
Epoch 750 [1.0 s]	train=66.0665 [0.0 s]
Epoch 751 [1.0 s]	train=66.4441 [0.0 s]
Epoch 752 [1.0 s]	train=66.0270 [0.0 s]
Epoch 753 [1.0 s]	train=66.8201 [0.0 s]
Epoch 754 [1.0 s]	train=65.7463 [0.0 s]
Epoch 755 [1.0 s]	train=66.2376 [0.0 s]
Epoch 756 [1.0 s]	train=65.6545 [0.0 s]
Epoch 757 [1.0 s]	train=65.7301 [0.0 s]
Epoch 758 [1.0 s]	train=66.7641 [0.0 s]
Epoch 759 [1.0 s]	train=67.0460 [0.0 s]
Epoch 760 [1.0 s]	train=66.0106 [0.0 s]
Epoch 761 [1.0 s]	train=66.2466 [0.0 s]
Epoch 762 [1.0 s]	train=65.0474 [0.0 s]
Epoch 763 [1.0 s]	train=66.0204 [0.0 s]
Epoch 764 [1.0 s]	train=65.6381 [0.0 s]
Epoch 765 [1.1 s]	train=65.9050 [0.0 s]
Epoch 766 [1.0 s]	train=66.3604 [0.0 s]
Epoch 767 [1.0 s]	train=65.7330 [0.0 s]
Epoch 768 [1.0 s]	train=66.6540 [0.0 s]
Epoch 769 [1.0 s]	train=66.7721 [0.0 s]
Epoch 770 [1.0 s]	train=65.9469 [0.0 s]
Epoch 771 [1.0 s]	train=65.7414 [0.0 s]
Epoch 772 [1.0 s]	train=66.2783 [0.0 s]
Epoch 773 [1.0 s]	train=66.5720 [0.0 s]
Epoch 774 [1.0 s]	train=67.2181 [0.0 s]
Epoch 775 [1.0 s]	train=66.2786 [0.0 s]
Epoch 776 [1.0 s]	train=66.8198 [0.0 s]
Epoch 777 [1.0 s]	train=66.2342 [0.0 s]
Epoch 778 [1.0 s]	train=65.9362 [0.0 s]
Epoch 779 [1.0 s]	train=65.7999 [0.0 s]
Epoch 780 [1.0 s]	train=65.6572 [0.0 s]
Epoch 781 [1.0 s]	train=66.7244 [0.0 s]
Epoch 782 [1.0 s]	train=65.8758 [0.0 s]
Epoch 783 [1.0 s]	train=64.9113 [0.0 s]
Epoch 784 [1.0 s]	train=65.6334 [0.0 s]
Epoch 785 [1.0 s]	train=63.9200 [0.0 s]
Epoch 786 [1.0 s]	train=66.2039 [0.0 s]
Epoch 787 [1.0 s]	train=66.5448 [0.0 s]
Epoch 788 [1.0 s]	train=66.4326 [0.0 s]
Epoch 789 [1.0 s]	train=65.6587 [0.0 s]
Epoch 790 [1.0 s]	train=66.0430 [0.0 s]
Epoch 791 [1.0 s]	train=66.3385 [0.0 s]
Epoch 792 [1.0 s]	train=65.3582 [0.0 s]
Epoch 793 [1.0 s]	train=66.0642 [0.0 s]
Epoch 794 [1.0 s]	train=65.3913 [0.0 s]
Epoch 795 [1.0 s]	train=65.9192 [0.0 s]
Epoch 796 [1.0 s]	train=65.8197 [0.0 s]
Epoch 797 [1.0 s]	train=65.3817 [0.0 s]
Epoch 798 [1.0 s]	train=65.1009 [0.0 s]
Epoch 799 [1.0 s]	train=65.6585 [0.0 s]
Epoch 800 [1.0 s]	train=64.8764 [0.0 s]
recommend 50 user costs 0.5770249366760254s
recommend 50 user costs 0.7045817375183105s
recommend 50 user costs 0.573322057723999s
recommend 50 user costs 0.5814294815063477s
recommend 50 user costs 0.5775179862976074s
recommend 50 user costs 0.5750794410705566s
recommend 50 user costs 0.5826692581176758s
recommend 50 user costs 0.639951229095459s
recommend 50 user costs 0.5786666870117188s
recommend 50 user costs 0.5746562480926514s
recommend 50 user costs 0.5867114067077637s
recommend 50 user costs 0.5756509304046631s
recommend 50 user costs 0.5766856670379639s
recommend 50 user costs 0.5876433849334717s
recommend 50 user costs 0.5786669254302979s
recommend 50 user costs 0.5766680240631104s
recommend 50 user costs 0.5826647281646729s
recommend 50 user costs 0.6486268043518066s
hit: 754
HR%10: 0.0722291407222914
MAP%10: 0.18460519449914964
NDCG%10: 0.26354504131482626
Epoch 801 [1.1 s]	train=65.3879 [0.0 s]
Epoch 802 [1.0 s]	train=66.2811 [0.0 s]
Epoch 803 [1.0 s]	train=65.3955 [0.0 s]
Epoch 804 [1.0 s]	train=66.0836 [0.0 s]
Epoch 805 [1.0 s]	train=66.2381 [0.0 s]
Epoch 806 [1.0 s]	train=65.7786 [0.0 s]
Epoch 807 [1.1 s]	train=65.2229 [0.0 s]
Epoch 808 [1.0 s]	train=65.8820 [0.0 s]
Epoch 809 [1.0 s]	train=65.3550 [0.0 s]
Epoch 810 [1.0 s]	train=66.2219 [0.0 s]
Epoch 811 [1.0 s]	train=65.7416 [0.0 s]
Epoch 812 [1.0 s]	train=66.0082 [0.0 s]
Epoch 813 [1.0 s]	train=66.7891 [0.0 s]
Epoch 814 [1.0 s]	train=65.4308 [0.0 s]
Epoch 815 [1.0 s]	train=65.7829 [0.0 s]
Epoch 816 [1.0 s]	train=65.4556 [0.0 s]
Epoch 817 [1.0 s]	train=65.0604 [0.0 s]
Epoch 818 [1.0 s]	train=66.2112 [0.0 s]
Epoch 819 [1.0 s]	train=64.6785 [0.0 s]
Epoch 820 [1.0 s]	train=66.8070 [0.0 s]
Epoch 821 [1.0 s]	train=64.7577 [0.0 s]
Epoch 822 [1.0 s]	train=66.1851 [0.0 s]
Epoch 823 [1.0 s]	train=66.5055 [0.0 s]
Epoch 824 [1.0 s]	train=65.8521 [0.0 s]
Epoch 825 [1.0 s]	train=65.5700 [0.0 s]
Epoch 826 [1.1 s]	train=65.6004 [0.0 s]
Epoch 827 [1.0 s]	train=65.8536 [0.0 s]
Epoch 828 [1.0 s]	train=65.7652 [0.0 s]
Epoch 829 [1.0 s]	train=65.2422 [0.0 s]
Epoch 830 [1.0 s]	train=66.6757 [0.0 s]
Epoch 831 [1.0 s]	train=66.2177 [0.0 s]
Epoch 832 [1.0 s]	train=66.3192 [0.0 s]
Epoch 833 [1.0 s]	train=66.6213 [0.0 s]
Epoch 834 [1.0 s]	train=66.0302 [0.0 s]
Epoch 835 [1.0 s]	train=65.5114 [0.0 s]
Epoch 836 [1.0 s]	train=66.4584 [0.0 s]
Epoch 837 [1.0 s]	train=66.4057 [0.0 s]
Epoch 838 [1.1 s]	train=66.2357 [0.0 s]
Epoch 839 [1.0 s]	train=65.5359 [0.0 s]
Epoch 840 [1.0 s]	train=66.1977 [0.0 s]
Epoch 841 [1.0 s]	train=65.9563 [0.0 s]
Epoch 842 [1.1 s]	train=65.6347 [0.0 s]
Epoch 843 [1.1 s]	train=65.9331 [0.0 s]
Epoch 844 [1.0 s]	train=65.3925 [0.0 s]
Epoch 845 [1.1 s]	train=66.1447 [0.0 s]
Epoch 846 [1.0 s]	train=65.2704 [0.0 s]
Epoch 847 [1.0 s]	train=66.1532 [0.0 s]
Epoch 848 [1.0 s]	train=65.5589 [0.0 s]
Epoch 849 [1.0 s]	train=65.5761 [0.0 s]
Epoch 850 [1.0 s]	train=65.6193 [0.0 s]
Epoch 851 [1.0 s]	train=66.4286 [0.0 s]
Epoch 852 [1.0 s]	train=66.1825 [0.0 s]
Epoch 853 [1.0 s]	train=64.9876 [0.0 s]
Epoch 854 [1.0 s]	train=65.8883 [0.0 s]
Epoch 855 [1.1 s]	train=65.6932 [0.0 s]
Epoch 856 [1.0 s]	train=65.2409 [0.0 s]
Epoch 857 [1.0 s]	train=65.6476 [0.0 s]
Epoch 858 [1.0 s]	train=65.3600 [0.0 s]
Epoch 859 [1.0 s]	train=65.0672 [0.0 s]
Epoch 860 [1.0 s]	train=64.8403 [0.0 s]
Epoch 861 [1.1 s]	train=65.5910 [0.0 s]
Epoch 862 [1.0 s]	train=65.5390 [0.0 s]
Epoch 863 [1.0 s]	train=65.4047 [0.0 s]
Epoch 864 [1.0 s]	train=65.3737 [0.0 s]
Epoch 865 [1.0 s]	train=65.4886 [0.0 s]
Epoch 866 [1.0 s]	train=66.4244 [0.0 s]
Epoch 867 [1.0 s]	train=65.3476 [0.0 s]
Epoch 868 [1.1 s]	train=65.8210 [0.0 s]
Epoch 869 [1.0 s]	train=65.4525 [0.0 s]
Epoch 870 [1.1 s]	train=66.1262 [0.0 s]
Epoch 871 [1.0 s]	train=65.2332 [0.0 s]
Epoch 872 [1.0 s]	train=65.6747 [0.0 s]
Epoch 873 [1.0 s]	train=64.5589 [0.0 s]
Epoch 874 [1.0 s]	train=65.2079 [0.0 s]
Epoch 875 [1.0 s]	train=64.9975 [0.0 s]
Epoch 876 [1.0 s]	train=66.7064 [0.0 s]
Epoch 877 [1.0 s]	train=66.2877 [0.0 s]
Epoch 878 [1.0 s]	train=64.9443 [0.0 s]
Epoch 879 [1.0 s]	train=65.9634 [0.0 s]
Epoch 880 [1.0 s]	train=64.9995 [0.0 s]
Epoch 881 [1.0 s]	train=65.1493 [0.0 s]
Epoch 882 [1.0 s]	train=66.1769 [0.0 s]
Epoch 883 [1.0 s]	train=66.0917 [0.0 s]
Epoch 884 [1.0 s]	train=65.7937 [0.0 s]
Epoch 885 [1.0 s]	train=66.3698 [0.0 s]
Epoch 886 [1.0 s]	train=66.1537 [0.0 s]
Epoch 887 [1.0 s]	train=65.7289 [0.0 s]
Epoch 888 [1.0 s]	train=65.9069 [0.0 s]
Epoch 889 [1.0 s]	train=65.4435 [0.0 s]
Epoch 890 [1.0 s]	train=65.9063 [0.0 s]
Epoch 891 [1.0 s]	train=66.0568 [0.0 s]
Epoch 892 [1.0 s]	train=65.4004 [0.0 s]
Epoch 893 [1.0 s]	train=66.8288 [0.0 s]
Epoch 894 [1.0 s]	train=65.3305 [0.0 s]
Epoch 895 [1.0 s]	train=65.4784 [0.0 s]
Epoch 896 [1.0 s]	train=65.6597 [0.0 s]
Epoch 897 [1.0 s]	train=65.0203 [0.0 s]
Epoch 898 [1.0 s]	train=65.3454 [0.0 s]
Epoch 899 [1.0 s]	train=66.7595 [0.0 s]
Epoch 900 [1.0 s]	train=66.1537 [0.0 s]
recommend 50 user costs 0.5766677856445312s
recommend 50 user costs 0.6206095218658447s
recommend 50 user costs 0.5972442626953125s
recommend 50 user costs 0.5897722244262695s
recommend 50 user costs 0.5799386501312256s
recommend 50 user costs 0.5743017196655273s
recommend 50 user costs 0.5736238956451416s
recommend 50 user costs 0.6556220054626465s
recommend 50 user costs 0.5782296657562256s
recommend 50 user costs 0.5816614627838135s
recommend 50 user costs 0.5912742614746094s
recommend 50 user costs 0.575678825378418s
recommend 50 user costs 0.6497988700866699s
recommend 50 user costs 0.5866439342498779s
recommend 50 user costs 0.5786793231964111s
recommend 50 user costs 0.5796542167663574s
recommend 50 user costs 0.581681489944458s
recommend 50 user costs 0.588660717010498s
hit: 775
HR%10: 0.0742408276654852
MAP%10: 0.1919328791372798
NDCG%10: 0.27413604728121943
Epoch 901 [1.0 s]	train=65.2868 [0.0 s]
Epoch 902 [1.0 s]	train=65.5766 [0.0 s]
Epoch 903 [1.0 s]	train=65.3478 [0.0 s]
Epoch 904 [1.0 s]	train=64.7710 [0.0 s]
Epoch 905 [1.0 s]	train=66.2010 [0.0 s]
Epoch 906 [1.0 s]	train=65.5515 [0.0 s]
Epoch 907 [1.0 s]	train=65.3880 [0.0 s]
Epoch 908 [1.0 s]	train=66.2573 [0.0 s]
Epoch 909 [1.0 s]	train=65.7405 [0.0 s]
Epoch 910 [1.0 s]	train=64.9893 [0.0 s]
Epoch 911 [1.0 s]	train=64.9829 [0.0 s]
Epoch 912 [1.0 s]	train=65.6702 [0.0 s]
Epoch 913 [1.0 s]	train=66.0959 [0.0 s]
Epoch 914 [1.0 s]	train=65.6472 [0.0 s]
Epoch 915 [1.0 s]	train=64.8859 [0.0 s]
Epoch 916 [1.0 s]	train=65.5300 [0.0 s]
Epoch 917 [1.0 s]	train=66.2412 [0.0 s]
Epoch 918 [1.0 s]	train=65.5296 [0.0 s]
Epoch 919 [1.0 s]	train=65.9183 [0.0 s]
Epoch 920 [1.0 s]	train=66.6168 [0.0 s]
Epoch 921 [1.0 s]	train=65.1701 [0.0 s]
Epoch 922 [1.0 s]	train=66.2659 [0.0 s]
Epoch 923 [1.0 s]	train=65.7923 [0.0 s]
Epoch 924 [1.0 s]	train=64.9744 [0.0 s]
Epoch 925 [1.0 s]	train=65.9765 [0.0 s]
Epoch 926 [1.0 s]	train=65.3052 [0.0 s]
Epoch 927 [1.0 s]	train=65.3996 [0.0 s]
Epoch 928 [1.0 s]	train=65.3730 [0.0 s]
Epoch 929 [1.0 s]	train=66.3293 [0.0 s]
Epoch 930 [1.1 s]	train=65.8165 [0.0 s]
Epoch 931 [1.0 s]	train=66.2496 [0.0 s]
Epoch 932 [1.0 s]	train=64.8378 [0.0 s]
Epoch 933 [1.0 s]	train=65.1027 [0.0 s]
Epoch 934 [1.0 s]	train=66.7534 [0.0 s]
Epoch 935 [1.0 s]	train=65.4051 [0.0 s]
Epoch 936 [1.0 s]	train=65.2555 [0.0 s]
Epoch 937 [1.0 s]	train=64.9215 [0.0 s]
Epoch 938 [1.0 s]	train=64.8455 [0.0 s]
Epoch 939 [1.0 s]	train=66.6099 [0.0 s]
Epoch 940 [1.0 s]	train=64.8603 [0.0 s]
Epoch 941 [1.0 s]	train=64.8000 [0.0 s]
Epoch 942 [1.0 s]	train=64.6250 [0.0 s]
Epoch 943 [1.0 s]	train=65.2784 [0.0 s]
Epoch 944 [1.0 s]	train=65.7249 [0.0 s]
Epoch 945 [1.0 s]	train=65.6198 [0.0 s]
Epoch 946 [1.1 s]	train=65.3465 [0.0 s]
Epoch 947 [1.0 s]	train=65.4202 [0.0 s]
Epoch 948 [1.0 s]	train=65.9580 [0.0 s]
Epoch 949 [1.0 s]	train=65.4650 [0.0 s]
Epoch 950 [1.0 s]	train=65.8824 [0.0 s]
Epoch 951 [1.0 s]	train=65.8412 [0.0 s]
Epoch 952 [1.0 s]	train=65.7021 [0.0 s]
Epoch 953 [1.0 s]	train=65.6605 [0.0 s]
Epoch 954 [1.0 s]	train=64.9510 [0.0 s]
Epoch 955 [1.0 s]	train=65.3822 [0.0 s]
Epoch 956 [1.0 s]	train=65.7877 [0.0 s]
Epoch 957 [1.0 s]	train=65.7186 [0.0 s]
Epoch 958 [1.0 s]	train=65.8862 [0.0 s]
Epoch 959 [1.0 s]	train=65.6241 [0.0 s]
Epoch 960 [1.0 s]	train=65.4959 [0.0 s]
Epoch 961 [1.0 s]	train=65.4053 [0.0 s]
Epoch 962 [1.0 s]	train=65.8337 [0.0 s]
Epoch 963 [1.0 s]	train=65.7479 [0.0 s]
Epoch 964 [1.0 s]	train=65.0415 [0.0 s]
Epoch 965 [1.0 s]	train=65.9824 [0.0 s]
Epoch 966 [1.0 s]	train=65.6194 [0.0 s]
Epoch 967 [1.0 s]	train=65.9363 [0.0 s]
Epoch 968 [1.0 s]	train=65.6530 [0.0 s]
Epoch 969 [1.0 s]	train=64.3245 [0.0 s]
Epoch 970 [1.0 s]	train=64.7961 [0.0 s]
Epoch 971 [1.0 s]	train=64.9406 [0.0 s]
Epoch 972 [1.0 s]	train=66.3019 [0.0 s]
Epoch 973 [1.0 s]	train=64.6300 [0.0 s]
Epoch 974 [1.0 s]	train=65.7217 [0.0 s]
Epoch 975 [1.0 s]	train=64.8486 [0.0 s]
Epoch 976 [1.0 s]	train=65.4361 [0.0 s]
Epoch 977 [1.0 s]	train=65.5724 [0.0 s]
Epoch 978 [1.0 s]	train=65.3013 [0.0 s]
Epoch 979 [1.0 s]	train=65.8802 [0.0 s]
Epoch 980 [1.0 s]	train=65.1679 [0.0 s]
Epoch 981 [1.0 s]	train=64.4310 [0.0 s]
Epoch 982 [1.0 s]	train=65.3148 [0.0 s]
Epoch 983 [1.0 s]	train=64.7404 [0.0 s]
Epoch 984 [1.0 s]	train=66.0577 [0.0 s]
Epoch 985 [1.0 s]	train=65.4226 [0.0 s]
Epoch 986 [1.0 s]	train=65.1704 [0.0 s]
Epoch 987 [1.0 s]	train=65.4619 [0.0 s]
Epoch 988 [1.0 s]	train=65.6127 [0.0 s]
Epoch 989 [1.0 s]	train=65.2902 [0.0 s]
Epoch 990 [1.0 s]	train=66.1619 [0.0 s]
Epoch 991 [1.0 s]	train=65.4599 [0.0 s]
Epoch 992 [1.0 s]	train=64.9804 [0.0 s]
Epoch 993 [1.0 s]	train=64.8865 [0.0 s]
Epoch 994 [1.0 s]	train=65.2012 [0.0 s]
Epoch 995 [1.0 s]	train=66.1876 [0.0 s]
Epoch 996 [1.0 s]	train=65.5330 [0.0 s]
Epoch 997 [1.0 s]	train=66.4635 [0.0 s]
Epoch 998 [1.0 s]	train=64.6336 [0.0 s]
Epoch 999 [1.0 s]	train=64.8429 [0.0 s]
Epoch 1000 [1.0 s]	train=65.4273 [0.0 s]
recommend 50 user costs 0.5736546516418457s
recommend 50 user costs 0.7025957107543945s
recommend 50 user costs 0.572669506072998s
recommend 50 user costs 0.5866742134094238s
recommend 50 user costs 0.5716578960418701s
recommend 50 user costs 0.5746686458587646s
recommend 50 user costs 0.5876622200012207s
recommend 50 user costs 0.5786659717559814s
recommend 50 user costs 0.5815820693969727s
recommend 50 user costs 0.5796072483062744s
recommend 50 user costs 0.5946569442749023s
recommend 50 user costs 0.5796525478363037s
recommend 50 user costs 0.6636176109313965s
recommend 50 user costs 0.5906732082366943s
recommend 50 user costs 0.580664873123169s
recommend 50 user costs 0.5776426792144775s
recommend 50 user costs 0.590662956237793s
recommend 50 user costs 0.5876617431640625s
hit: 754
HR%10: 0.0722291407222914
MAP%10: 0.1988049580029959
NDCG%10: 0.27666970423693293
Epoch 1001 [1.1 s]	train=65.8312 [0.0 s]
Epoch 1002 [1.0 s]	train=64.6153 [0.0 s]
Epoch 1003 [1.0 s]	train=64.8594 [0.0 s]
Epoch 1004 [1.0 s]	train=65.3158 [0.0 s]
Epoch 1005 [1.0 s]	train=66.0427 [0.0 s]
Epoch 1006 [1.0 s]	train=65.2432 [0.0 s]
Epoch 1007 [1.1 s]	train=65.5413 [0.0 s]
Epoch 1008 [1.0 s]	train=65.1275 [0.0 s]
Epoch 1009 [1.0 s]	train=65.3134 [0.0 s]
Epoch 1010 [1.0 s]	train=65.4792 [0.0 s]
Epoch 1011 [1.1 s]	train=65.3402 [0.0 s]
Epoch 1012 [1.0 s]	train=64.8010 [0.0 s]
Epoch 1013 [1.0 s]	train=65.4042 [0.0 s]
Epoch 1014 [1.1 s]	train=65.2884 [0.0 s]
Epoch 1015 [1.0 s]	train=65.5441 [0.0 s]
Epoch 1016 [1.1 s]	train=65.6811 [0.0 s]
Epoch 1017 [1.0 s]	train=64.9978 [0.0 s]
Epoch 1018 [1.1 s]	train=66.3875 [0.0 s]
Epoch 1019 [1.0 s]	train=65.2507 [0.0 s]
Epoch 1020 [1.1 s]	train=65.0663 [0.0 s]
Epoch 1021 [1.0 s]	train=66.5570 [0.0 s]
Epoch 1022 [1.1 s]	train=65.2945 [0.0 s]
Epoch 1023 [1.0 s]	train=65.3693 [0.0 s]
Epoch 1024 [1.0 s]	train=65.0498 [0.0 s]
Epoch 1025 [1.0 s]	train=65.6433 [0.0 s]
Epoch 1026 [1.0 s]	train=65.2564 [0.0 s]
Epoch 1027 [1.0 s]	train=66.4002 [0.0 s]
Epoch 1028 [1.1 s]	train=65.7902 [0.0 s]
Epoch 1029 [1.0 s]	train=65.3678 [0.0 s]
Epoch 1030 [1.0 s]	train=66.1150 [0.0 s]
Epoch 1031 [1.0 s]	train=64.8520 [0.0 s]
Epoch 1032 [1.0 s]	train=65.6282 [0.0 s]
Epoch 1033 [1.0 s]	train=65.1034 [0.0 s]
Epoch 1034 [1.0 s]	train=65.4613 [0.0 s]
Epoch 1035 [1.0 s]	train=65.1020 [0.0 s]
Epoch 1036 [1.0 s]	train=65.2332 [0.0 s]
Epoch 1037 [1.0 s]	train=65.2965 [0.0 s]
Epoch 1038 [1.0 s]	train=65.1599 [0.0 s]
Epoch 1039 [1.1 s]	train=65.4217 [0.0 s]
Epoch 1040 [1.0 s]	train=65.2757 [0.0 s]
Epoch 1041 [1.0 s]	train=65.7185 [0.0 s]
Epoch 1042 [1.0 s]	train=65.7515 [0.0 s]
Epoch 1043 [1.1 s]	train=64.6610 [0.0 s]
Epoch 1044 [1.0 s]	train=65.5938 [0.0 s]
Epoch 1045 [1.1 s]	train=65.1255 [0.0 s]
Epoch 1046 [1.0 s]	train=65.2526 [0.0 s]
Epoch 1047 [1.0 s]	train=64.8918 [0.0 s]
Epoch 1048 [1.1 s]	train=64.8248 [0.0 s]
Epoch 1049 [1.1 s]	train=64.5999 [0.0 s]
Epoch 1050 [1.0 s]	train=65.0047 [0.0 s]
Epoch 1051 [1.1 s]	train=65.1763 [0.0 s]
Epoch 1052 [1.0 s]	train=66.0001 [0.0 s]
Epoch 1053 [1.0 s]	train=64.3601 [0.0 s]
Epoch 1054 [1.1 s]	train=64.7426 [0.0 s]
Epoch 1055 [1.0 s]	train=66.0387 [0.0 s]
Epoch 1056 [1.0 s]	train=64.8365 [0.0 s]
Epoch 1057 [1.0 s]	train=65.3815 [0.0 s]
Epoch 1058 [1.0 s]	train=65.5498 [0.0 s]
Epoch 1059 [1.0 s]	train=65.8446 [0.0 s]
Epoch 1060 [1.0 s]	train=64.7041 [0.0 s]
Epoch 1061 [1.0 s]	train=65.6009 [0.0 s]
Epoch 1062 [1.0 s]	train=65.6129 [0.0 s]
Epoch 1063 [1.0 s]	train=64.6644 [0.0 s]
Epoch 1064 [1.1 s]	train=65.2222 [0.0 s]
Epoch 1065 [1.0 s]	train=65.2858 [0.0 s]
Epoch 1066 [1.1 s]	train=66.0342 [0.0 s]
Epoch 1067 [1.0 s]	train=64.7071 [0.0 s]
Epoch 1068 [1.0 s]	train=64.3775 [0.0 s]
Epoch 1069 [1.0 s]	train=65.0627 [0.0 s]
Epoch 1070 [1.0 s]	train=65.1793 [0.0 s]
Epoch 1071 [1.0 s]	train=65.3806 [0.0 s]
Epoch 1072 [1.0 s]	train=65.3110 [0.0 s]
Epoch 1073 [1.0 s]	train=65.5688 [0.0 s]
Epoch 1074 [1.1 s]	train=64.4575 [0.0 s]
Epoch 1075 [1.0 s]	train=64.5947 [0.0 s]
Epoch 1076 [1.1 s]	train=64.9357 [0.0 s]
Epoch 1077 [1.0 s]	train=64.5599 [0.0 s]
Epoch 1078 [1.0 s]	train=66.2089 [0.0 s]
Epoch 1079 [1.0 s]	train=65.2134 [0.0 s]
Epoch 1080 [1.0 s]	train=65.5525 [0.0 s]
Epoch 1081 [1.0 s]	train=65.4936 [0.0 s]
Epoch 1082 [1.0 s]	train=64.5114 [0.0 s]
Epoch 1083 [1.1 s]	train=66.2394 [0.0 s]
Epoch 1084 [1.0 s]	train=64.7241 [0.0 s]
Epoch 1085 [1.0 s]	train=64.5963 [0.0 s]
Epoch 1086 [1.0 s]	train=64.6161 [0.0 s]
Epoch 1087 [1.0 s]	train=64.8659 [0.0 s]
Epoch 1088 [1.0 s]	train=64.2756 [0.0 s]
Epoch 1089 [1.1 s]	train=64.3321 [0.0 s]
Epoch 1090 [1.0 s]	train=65.5713 [0.0 s]
Epoch 1091 [1.0 s]	train=65.8525 [0.0 s]
Epoch 1092 [1.0 s]	train=65.5276 [0.0 s]
Epoch 1093 [1.1 s]	train=64.8333 [0.0 s]
Epoch 1094 [1.0 s]	train=64.5485 [0.0 s]
Epoch 1095 [1.0 s]	train=64.9438 [0.0 s]
Epoch 1096 [1.0 s]	train=64.5037 [0.0 s]
Epoch 1097 [1.0 s]	train=65.2749 [0.0 s]
Epoch 1098 [1.0 s]	train=65.2992 [0.0 s]
Epoch 1099 [1.0 s]	train=65.2479 [0.0 s]
Epoch 1100 [1.0 s]	train=65.7804 [0.0 s]
recommend 50 user costs 0.5706605911254883s
recommend 50 user costs 0.6136465072631836s
recommend 50 user costs 0.5928242206573486s
recommend 50 user costs 0.5796003341674805s
recommend 50 user costs 0.6448445320129395s
recommend 50 user costs 0.5716779232025146s
recommend 50 user costs 0.5846288204193115s
recommend 50 user costs 0.5756638050079346s
recommend 50 user costs 0.5776715278625488s
recommend 50 user costs 0.588660717010498s
recommend 50 user costs 0.5796661376953125s
recommend 50 user costs 0.5766630172729492s
recommend 50 user costs 0.5796701908111572s
recommend 50 user costs 0.5936405658721924s
recommend 50 user costs 0.5794694423675537s
recommend 50 user costs 0.6456277370452881s
recommend 50 user costs 0.6046514511108398s
recommend 50 user costs 0.6146459579467773s
hit: 794
HR%10: 0.07606092537599386
MAP%10: 0.18455234305929272
NDCG%10: 0.2665840475596497
Epoch 1101 [1.1 s]	train=65.7675 [0.0 s]
Epoch 1102 [1.1 s]	train=65.0565 [0.0 s]
Epoch 1103 [1.1 s]	train=63.9296 [0.0 s]
Epoch 1104 [1.0 s]	train=65.6076 [0.0 s]
Epoch 1105 [1.1 s]	train=65.4226 [0.0 s]
Epoch 1106 [1.0 s]	train=64.8375 [0.0 s]
Epoch 1107 [1.0 s]	train=64.4050 [0.0 s]
Epoch 1108 [1.0 s]	train=65.0709 [0.0 s]
Epoch 1109 [1.1 s]	train=65.3199 [0.0 s]
Epoch 1110 [1.1 s]	train=65.0461 [0.0 s]
Epoch 1111 [1.0 s]	train=64.5641 [0.0 s]
Epoch 1112 [1.0 s]	train=64.4385 [0.0 s]
Epoch 1113 [1.0 s]	train=64.7670 [0.0 s]
Epoch 1114 [1.0 s]	train=64.8306 [0.0 s]
Epoch 1115 [1.0 s]	train=64.6620 [0.0 s]
Epoch 1116 [1.1 s]	train=65.0469 [0.0 s]
Epoch 1117 [1.0 s]	train=64.1395 [0.0 s]
Epoch 1118 [1.0 s]	train=64.7117 [0.0 s]
Epoch 1119 [1.0 s]	train=65.8510 [0.0 s]
Epoch 1120 [1.1 s]	train=65.9650 [0.0 s]
Epoch 1121 [1.0 s]	train=65.8404 [0.0 s]
Epoch 1122 [1.1 s]	train=64.7960 [0.0 s]
Epoch 1123 [1.1 s]	train=64.7007 [0.0 s]
Epoch 1124 [1.1 s]	train=64.9038 [0.0 s]
Epoch 1125 [1.0 s]	train=65.8205 [0.0 s]
Epoch 1126 [1.1 s]	train=65.3065 [0.0 s]
Epoch 1127 [1.0 s]	train=65.0767 [0.0 s]
Epoch 1128 [1.1 s]	train=65.3314 [0.0 s]
Epoch 1129 [1.0 s]	train=65.0589 [0.0 s]
Epoch 1130 [1.0 s]	train=64.5766 [0.0 s]
Epoch 1131 [1.0 s]	train=64.5687 [0.0 s]
Epoch 1132 [1.0 s]	train=64.3718 [0.0 s]
Epoch 1133 [1.1 s]	train=64.4037 [0.0 s]
Epoch 1134 [1.0 s]	train=64.9937 [0.0 s]
Epoch 1135 [1.0 s]	train=64.8243 [0.0 s]
Epoch 1136 [1.0 s]	train=64.7423 [0.0 s]
Epoch 1137 [1.1 s]	train=64.3396 [0.0 s]
Epoch 1138 [1.0 s]	train=64.4315 [0.0 s]
Epoch 1139 [1.1 s]	train=64.6506 [0.0 s]
Epoch 1140 [1.0 s]	train=64.0280 [0.0 s]
Epoch 1141 [1.0 s]	train=65.8482 [0.0 s]
Epoch 1142 [1.0 s]	train=65.7131 [0.0 s]
Epoch 1143 [1.1 s]	train=64.6149 [0.0 s]
Epoch 1144 [1.0 s]	train=65.0419 [0.0 s]
Epoch 1145 [1.1 s]	train=65.1045 [0.0 s]
Epoch 1146 [1.0 s]	train=65.2523 [0.0 s]
Epoch 1147 [1.0 s]	train=64.9091 [0.0 s]
Epoch 1148 [1.0 s]	train=65.1067 [0.0 s]
Epoch 1149 [1.0 s]	train=65.0279 [0.0 s]
Epoch 1150 [1.0 s]	train=64.8983 [0.0 s]
Epoch 1151 [1.1 s]	train=65.6382 [0.0 s]
Epoch 1152 [1.0 s]	train=64.6879 [0.0 s]
Epoch 1153 [1.0 s]	train=64.2799 [0.0 s]
Epoch 1154 [1.0 s]	train=64.9784 [0.0 s]
Epoch 1155 [1.0 s]	train=64.6638 [0.0 s]
Epoch 1156 [1.0 s]	train=64.5910 [0.0 s]
Epoch 1157 [1.0 s]	train=64.0808 [0.0 s]
Epoch 1158 [1.1 s]	train=64.0891 [0.0 s]
Epoch 1159 [1.0 s]	train=65.9890 [0.0 s]
Epoch 1160 [1.1 s]	train=65.3268 [0.0 s]
Epoch 1161 [1.0 s]	train=65.9397 [0.0 s]
Epoch 1162 [1.0 s]	train=64.8778 [0.0 s]
Epoch 1163 [1.0 s]	train=64.8742 [0.0 s]
Epoch 1164 [1.0 s]	train=65.2118 [0.0 s]
Epoch 1165 [1.0 s]	train=64.1810 [0.0 s]
Epoch 1166 [1.0 s]	train=65.6166 [0.0 s]
Epoch 1167 [1.0 s]	train=64.4330 [0.0 s]
Epoch 1168 [1.0 s]	train=64.8795 [0.0 s]
Epoch 1169 [1.0 s]	train=66.5187 [0.0 s]
Epoch 1170 [1.0 s]	train=63.8519 [0.0 s]
Epoch 1171 [1.0 s]	train=64.3302 [0.0 s]
Epoch 1172 [1.0 s]	train=63.7421 [0.0 s]
Epoch 1173 [1.0 s]	train=64.2917 [0.0 s]
Epoch 1174 [1.1 s]	train=65.0854 [0.0 s]
Epoch 1175 [1.1 s]	train=64.7185 [0.0 s]
Epoch 1176 [1.1 s]	train=64.2645 [0.0 s]
Epoch 1177 [1.0 s]	train=64.0330 [0.0 s]
Epoch 1178 [1.1 s]	train=64.2824 [0.0 s]
Epoch 1179 [1.1 s]	train=64.3858 [0.0 s]
Epoch 1180 [1.0 s]	train=64.4360 [0.0 s]
Epoch 1181 [1.1 s]	train=65.7204 [0.0 s]
Epoch 1182 [1.1 s]	train=64.4943 [0.0 s]
Epoch 1183 [1.1 s]	train=64.8987 [0.0 s]
Epoch 1184 [1.0 s]	train=64.6360 [0.0 s]
Epoch 1185 [1.1 s]	train=65.2563 [0.0 s]
Epoch 1186 [1.1 s]	train=64.9412 [0.0 s]
Epoch 1187 [1.1 s]	train=64.5989 [0.0 s]
Epoch 1188 [1.0 s]	train=64.6378 [0.0 s]
Epoch 1189 [1.0 s]	train=64.4714 [0.0 s]
Epoch 1190 [1.0 s]	train=64.6443 [0.0 s]
Epoch 1191 [1.0 s]	train=64.3748 [0.0 s]
Epoch 1192 [1.0 s]	train=64.8820 [0.0 s]
Epoch 1193 [1.0 s]	train=64.9880 [0.0 s]
Epoch 1194 [1.0 s]	train=65.3940 [0.0 s]
Epoch 1195 [1.0 s]	train=64.7250 [0.0 s]
Epoch 1196 [1.0 s]	train=64.0243 [0.0 s]
Epoch 1197 [1.0 s]	train=64.3807 [0.0 s]
Epoch 1198 [1.0 s]	train=63.9787 [0.0 s]
Epoch 1199 [1.0 s]	train=64.8942 [0.0 s]
Epoch 1200 [1.0 s]	train=64.8817 [0.0 s]
recommend 50 user costs 0.5596725940704346s
recommend 50 user costs 0.7115960121154785s
recommend 50 user costs 0.582181453704834s
recommend 50 user costs 0.5787937641143799s
recommend 50 user costs 0.5766737461090088s
recommend 50 user costs 0.581646203994751s
recommend 50 user costs 0.5846810340881348s
recommend 50 user costs 0.6404902935028076s
recommend 50 user costs 0.578679084777832s
recommend 50 user costs 0.5826692581176758s
recommend 50 user costs 0.5746636390686035s
recommend 50 user costs 0.5716750621795654s
recommend 50 user costs 0.584916353225708s
recommend 50 user costs 0.5775260925292969s
recommend 50 user costs 0.5779075622558594s
recommend 50 user costs 0.5826771259307861s
recommend 50 user costs 0.588665246963501s
recommend 50 user costs 0.5766501426696777s
hit: 748
HR%10: 0.07165437302423604
MAP%10: 0.18601462300485164
NDCG%10: 0.2650142664880738
Epoch 1201 [1.1 s]	train=63.3466 [0.0 s]
Epoch 1202 [1.0 s]	train=65.3082 [0.0 s]
Epoch 1203 [1.0 s]	train=64.7985 [0.0 s]
Epoch 1204 [1.0 s]	train=65.6078 [0.0 s]
Epoch 1205 [1.0 s]	train=64.4920 [0.0 s]
Epoch 1206 [1.0 s]	train=64.7153 [0.0 s]
Epoch 1207 [1.0 s]	train=65.9061 [0.0 s]
Epoch 1208 [1.0 s]	train=64.9331 [0.0 s]
Epoch 1209 [1.0 s]	train=64.8905 [0.0 s]
Epoch 1210 [1.0 s]	train=65.1762 [0.0 s]
Epoch 1211 [1.0 s]	train=64.8383 [0.0 s]
Epoch 1212 [1.0 s]	train=65.3750 [0.0 s]
Epoch 1213 [1.0 s]	train=64.7871 [0.0 s]
Epoch 1214 [1.0 s]	train=64.1518 [0.0 s]
Epoch 1215 [1.0 s]	train=63.4314 [0.0 s]
Epoch 1216 [1.0 s]	train=65.4265 [0.0 s]
Epoch 1217 [1.0 s]	train=65.4433 [0.0 s]
Epoch 1218 [1.0 s]	train=64.1316 [0.0 s]
Epoch 1219 [1.0 s]	train=65.3894 [0.0 s]
Epoch 1220 [1.0 s]	train=64.3464 [0.0 s]
Epoch 1221 [1.0 s]	train=65.0577 [0.0 s]
Epoch 1222 [1.0 s]	train=64.3174 [0.0 s]
Epoch 1223 [1.0 s]	train=65.0548 [0.0 s]
Epoch 1224 [1.0 s]	train=64.2753 [0.0 s]
Epoch 1225 [1.0 s]	train=63.9116 [0.0 s]
Epoch 1226 [1.0 s]	train=64.0864 [0.0 s]
Epoch 1227 [1.0 s]	train=64.7189 [0.0 s]
Epoch 1228 [1.0 s]	train=65.0954 [0.0 s]
Epoch 1229 [1.0 s]	train=64.7210 [0.0 s]
Epoch 1230 [1.0 s]	train=64.9539 [0.0 s]
Epoch 1231 [1.0 s]	train=65.4293 [0.0 s]
Epoch 1232 [1.0 s]	train=64.1005 [0.0 s]
Epoch 1233 [1.0 s]	train=63.9548 [0.0 s]
Epoch 1234 [1.0 s]	train=64.5315 [0.0 s]
Epoch 1235 [1.0 s]	train=64.5651 [0.0 s]
Epoch 1236 [1.0 s]	train=65.6723 [0.0 s]
Epoch 1237 [1.0 s]	train=63.2155 [0.0 s]
Epoch 1238 [1.0 s]	train=64.2440 [0.0 s]
Epoch 1239 [1.0 s]	train=64.9530 [0.0 s]
Epoch 1240 [1.0 s]	train=65.1636 [0.0 s]
Epoch 1241 [1.0 s]	train=65.0041 [0.0 s]
Epoch 1242 [1.0 s]	train=64.1255 [0.0 s]
Epoch 1243 [1.0 s]	train=65.2524 [0.0 s]
Epoch 1244 [1.0 s]	train=63.3747 [0.0 s]
Epoch 1245 [1.0 s]	train=64.9194 [0.0 s]
Epoch 1246 [1.0 s]	train=64.2873 [0.0 s]
Epoch 1247 [1.0 s]	train=64.5697 [0.0 s]
Epoch 1248 [1.0 s]	train=65.3310 [0.0 s]
Epoch 1249 [1.0 s]	train=65.1282 [0.0 s]
Epoch 1250 [1.0 s]	train=63.9645 [0.0 s]
Epoch 1251 [1.0 s]	train=65.4871 [0.0 s]
Epoch 1252 [1.0 s]	train=64.7027 [0.0 s]
Epoch 1253 [1.0 s]	train=64.9815 [0.0 s]
Epoch 1254 [1.0 s]	train=64.7264 [0.0 s]
Epoch 1255 [1.0 s]	train=65.3181 [0.0 s]
Epoch 1256 [1.0 s]	train=64.9455 [0.0 s]
Epoch 1257 [1.0 s]	train=64.5631 [0.0 s]
Epoch 1258 [1.0 s]	train=64.5538 [0.0 s]
Epoch 1259 [1.0 s]	train=64.7316 [0.0 s]
Epoch 1260 [1.0 s]	train=63.9219 [0.0 s]
Epoch 1261 [1.0 s]	train=64.4393 [0.0 s]
Epoch 1262 [1.0 s]	train=64.8923 [0.0 s]
Epoch 1263 [1.0 s]	train=64.5021 [0.0 s]
Epoch 1264 [1.0 s]	train=65.4462 [0.0 s]
Epoch 1265 [1.0 s]	train=64.8996 [0.0 s]
Epoch 1266 [1.0 s]	train=65.3255 [0.0 s]
Epoch 1267 [1.0 s]	train=64.6960 [0.0 s]
Epoch 1268 [1.0 s]	train=65.1688 [0.0 s]
Epoch 1269 [1.0 s]	train=64.9398 [0.0 s]
Epoch 1270 [1.1 s]	train=64.6212 [0.0 s]
Epoch 1271 [1.0 s]	train=64.6295 [0.0 s]
Epoch 1272 [1.1 s]	train=64.8442 [0.0 s]
Epoch 1273 [1.0 s]	train=64.6137 [0.0 s]
Epoch 1274 [1.1 s]	train=65.4366 [0.0 s]
Epoch 1275 [1.0 s]	train=65.0749 [0.0 s]
Epoch 1276 [1.0 s]	train=63.8481 [0.0 s]
Epoch 1277 [1.0 s]	train=63.7646 [0.0 s]
Epoch 1278 [1.0 s]	train=64.1835 [0.0 s]
Epoch 1279 [1.0 s]	train=65.8232 [0.0 s]
Epoch 1280 [1.0 s]	train=64.9542 [0.0 s]
Epoch 1281 [1.0 s]	train=64.2691 [0.0 s]
Epoch 1282 [1.1 s]	train=65.2875 [0.0 s]
Epoch 1283 [1.0 s]	train=65.6239 [0.0 s]
Epoch 1284 [1.0 s]	train=64.6635 [0.0 s]
Epoch 1285 [1.0 s]	train=65.0001 [0.0 s]
Epoch 1286 [1.0 s]	train=63.8104 [0.0 s]
Epoch 1287 [1.0 s]	train=65.0938 [0.0 s]
Epoch 1288 [1.0 s]	train=64.3775 [0.0 s]
Epoch 1289 [1.0 s]	train=63.9083 [0.0 s]
Epoch 1290 [1.0 s]	train=64.3553 [0.0 s]
Epoch 1291 [1.0 s]	train=64.3965 [0.0 s]
Epoch 1292 [1.0 s]	train=64.3697 [0.0 s]
Epoch 1293 [1.0 s]	train=64.5192 [0.0 s]
Epoch 1294 [1.0 s]	train=64.3148 [0.0 s]
Epoch 1295 [1.0 s]	train=64.8578 [0.0 s]
Epoch 1296 [1.0 s]	train=63.6712 [0.0 s]
Epoch 1297 [1.0 s]	train=64.9056 [0.0 s]
Epoch 1298 [1.0 s]	train=64.9406 [0.0 s]
Epoch 1299 [1.0 s]	train=64.1619 [0.0 s]
Epoch 1300 [1.0 s]	train=64.7689 [0.0 s]
recommend 50 user costs 0.5676829814910889s
recommend 50 user costs 0.6226482391357422s
recommend 50 user costs 0.5859005451202393s
recommend 50 user costs 0.5776844024658203s
recommend 50 user costs 0.5816652774810791s
recommend 50 user costs 0.5726525783538818s
recommend 50 user costs 0.5802018642425537s
recommend 50 user costs 0.5906431674957275s
recommend 50 user costs 0.6555497646331787s
recommend 50 user costs 0.5776677131652832s
recommend 50 user costs 0.5798478126525879s
recommend 50 user costs 0.5901377201080322s
recommend 50 user costs 0.5796592235565186s
recommend 50 user costs 0.6446337699890137s
recommend 50 user costs 0.5870590209960938s
recommend 50 user costs 0.584061861038208s
recommend 50 user costs 0.5796658992767334s
recommend 50 user costs 0.5942609310150146s
hit: 771
HR%10: 0.07385764920011495
MAP%10: 0.19348164138991264
NDCG%10: 0.2746412536032172
Epoch 1301 [1.0 s]	train=64.9851 [0.0 s]
Epoch 1302 [1.0 s]	train=64.6999 [0.0 s]
Epoch 1303 [1.0 s]	train=64.8810 [0.0 s]
Epoch 1304 [1.0 s]	train=64.0867 [0.0 s]
Epoch 1305 [1.0 s]	train=64.1578 [0.0 s]
Epoch 1306 [1.0 s]	train=64.5829 [0.0 s]
Epoch 1307 [1.0 s]	train=64.7716 [0.0 s]
Epoch 1308 [1.0 s]	train=64.2923 [0.0 s]
Epoch 1309 [1.0 s]	train=64.6949 [0.0 s]
Epoch 1310 [1.0 s]	train=63.9698 [0.0 s]
Epoch 1311 [1.0 s]	train=64.9301 [0.0 s]
Epoch 1312 [1.0 s]	train=64.9724 [0.0 s]
Epoch 1313 [1.0 s]	train=64.2562 [0.0 s]
Epoch 1314 [1.0 s]	train=64.6128 [0.0 s]
Epoch 1315 [1.0 s]	train=64.8458 [0.0 s]
Epoch 1316 [1.0 s]	train=64.1360 [0.0 s]
Epoch 1317 [1.0 s]	train=64.4070 [0.0 s]
Epoch 1318 [1.0 s]	train=65.4097 [0.0 s]
Epoch 1319 [1.0 s]	train=64.3584 [0.0 s]
Epoch 1320 [1.0 s]	train=64.0753 [0.0 s]
Epoch 1321 [1.0 s]	train=64.8791 [0.0 s]
Epoch 1322 [1.0 s]	train=63.1409 [0.0 s]
Epoch 1323 [1.0 s]	train=64.6243 [0.0 s]
Epoch 1324 [1.0 s]	train=66.0834 [0.0 s]
Epoch 1325 [1.0 s]	train=65.0374 [0.0 s]
Epoch 1326 [1.0 s]	train=63.3547 [0.0 s]
Epoch 1327 [1.0 s]	train=64.9716 [0.0 s]
Epoch 1328 [1.0 s]	train=64.5578 [0.0 s]
Epoch 1329 [1.0 s]	train=64.0865 [0.0 s]
Epoch 1330 [1.0 s]	train=64.5248 [0.0 s]
Epoch 1331 [1.0 s]	train=64.3513 [0.0 s]
Epoch 1332 [1.0 s]	train=63.8121 [0.0 s]
Epoch 1333 [1.0 s]	train=64.6178 [0.0 s]
Epoch 1334 [1.0 s]	train=65.0682 [0.0 s]
Epoch 1335 [1.0 s]	train=64.9939 [0.0 s]
Epoch 1336 [1.0 s]	train=64.8316 [0.0 s]
Epoch 1337 [1.0 s]	train=63.9109 [0.0 s]
Epoch 1338 [1.0 s]	train=64.3943 [0.0 s]
Epoch 1339 [1.0 s]	train=65.3574 [0.0 s]
Epoch 1340 [1.0 s]	train=65.0064 [0.0 s]
Epoch 1341 [1.0 s]	train=64.2241 [0.0 s]
Epoch 1342 [1.0 s]	train=64.4333 [0.0 s]
Epoch 1343 [1.0 s]	train=64.2581 [0.0 s]
Epoch 1344 [1.0 s]	train=65.8116 [0.0 s]
Epoch 1345 [1.0 s]	train=65.0527 [0.0 s]
Epoch 1346 [1.0 s]	train=64.2796 [0.0 s]
Epoch 1347 [1.0 s]	train=65.1382 [0.0 s]
Epoch 1348 [1.0 s]	train=64.7021 [0.0 s]
Epoch 1349 [1.0 s]	train=65.1230 [0.0 s]
Epoch 1350 [1.0 s]	train=64.3429 [0.0 s]
Epoch 1351 [1.0 s]	train=64.4738 [0.0 s]
Epoch 1352 [1.0 s]	train=64.1577 [0.0 s]
Epoch 1353 [1.0 s]	train=65.0042 [0.0 s]
Epoch 1354 [1.0 s]	train=64.1258 [0.0 s]
Epoch 1355 [1.0 s]	train=64.9228 [0.0 s]
Epoch 1356 [1.0 s]	train=65.5311 [0.0 s]
Epoch 1357 [1.0 s]	train=64.1863 [0.0 s]
Epoch 1358 [1.0 s]	train=64.5981 [0.0 s]
Epoch 1359 [1.0 s]	train=64.9337 [0.0 s]
Epoch 1360 [1.0 s]	train=64.3437 [0.0 s]
Epoch 1361 [1.0 s]	train=63.6145 [0.0 s]
Epoch 1362 [1.1 s]	train=65.0232 [0.0 s]
Epoch 1363 [1.0 s]	train=65.5459 [0.0 s]
Epoch 1364 [1.0 s]	train=63.3660 [0.0 s]
Epoch 1365 [1.0 s]	train=64.4319 [0.0 s]
Epoch 1366 [1.1 s]	train=65.1579 [0.0 s]
Epoch 1367 [1.0 s]	train=64.3384 [0.0 s]
Epoch 1368 [1.0 s]	train=64.7883 [0.0 s]
Epoch 1369 [1.0 s]	train=65.7714 [0.0 s]
Epoch 1370 [1.0 s]	train=65.1269 [0.0 s]
Epoch 1371 [1.0 s]	train=65.2612 [0.0 s]
Epoch 1372 [1.0 s]	train=64.8856 [0.0 s]
Epoch 1373 [1.0 s]	train=64.9135 [0.0 s]
Epoch 1374 [1.0 s]	train=65.3692 [0.0 s]
Epoch 1375 [1.0 s]	train=63.3774 [0.0 s]
Epoch 1376 [1.0 s]	train=64.6414 [0.0 s]
Epoch 1377 [1.0 s]	train=64.5807 [0.0 s]
Epoch 1378 [1.0 s]	train=64.5373 [0.0 s]
Epoch 1379 [1.0 s]	train=63.5510 [0.0 s]
Epoch 1380 [1.0 s]	train=64.9238 [0.0 s]
Epoch 1381 [1.0 s]	train=64.4622 [0.0 s]
Epoch 1382 [1.1 s]	train=64.2565 [0.0 s]
Epoch 1383 [1.0 s]	train=63.8785 [0.0 s]
Epoch 1384 [1.0 s]	train=64.0338 [0.0 s]
Epoch 1385 [1.0 s]	train=63.5151 [0.0 s]
Epoch 1386 [1.0 s]	train=64.8774 [0.0 s]
Epoch 1387 [1.0 s]	train=63.7444 [0.0 s]
Epoch 1388 [1.0 s]	train=64.9196 [0.0 s]
Epoch 1389 [1.0 s]	train=63.9585 [0.0 s]
Epoch 1390 [1.0 s]	train=65.4072 [0.0 s]
Epoch 1391 [1.0 s]	train=63.7641 [0.0 s]
Epoch 1392 [1.0 s]	train=64.6931 [0.0 s]
Epoch 1393 [1.0 s]	train=64.1233 [0.0 s]
Epoch 1394 [1.0 s]	train=64.1156 [0.0 s]
Epoch 1395 [1.0 s]	train=65.0603 [0.0 s]
Epoch 1396 [1.0 s]	train=63.2335 [0.0 s]
Epoch 1397 [1.0 s]	train=64.5530 [0.0 s]
Epoch 1398 [1.0 s]	train=64.3768 [0.0 s]
Epoch 1399 [1.0 s]	train=65.0771 [0.0 s]
Epoch 1400 [1.0 s]	train=63.6035 [0.0 s]
recommend 50 user costs 0.5756826400756836s
recommend 50 user costs 0.624643087387085s
recommend 50 user costs 0.6756105422973633s
recommend 50 user costs 0.5756680965423584s
recommend 50 user costs 0.5846588611602783s
recommend 50 user costs 0.5729842185974121s
recommend 50 user costs 0.5726566314697266s
recommend 50 user costs 0.5856626033782959s
recommend 50 user costs 0.5766808986663818s
recommend 50 user costs 0.5786533355712891s
recommend 50 user costs 0.583899736404419s
recommend 50 user costs 0.5836629867553711s
recommend 50 user costs 0.6536381244659424s
recommend 50 user costs 0.5816669464111328s
recommend 50 user costs 0.5877776145935059s
recommend 50 user costs 0.5822482109069824s
recommend 50 user costs 0.580665111541748s
recommend 50 user costs 0.5936448574066162s
hit: 781
HR%10: 0.07481559536354057
MAP%10: 0.20020807032604465
NDCG%10: 0.2806827094952065
```  
  
