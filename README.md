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
