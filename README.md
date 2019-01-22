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
![NCF_loss](NCF_loss.png)
