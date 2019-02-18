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
