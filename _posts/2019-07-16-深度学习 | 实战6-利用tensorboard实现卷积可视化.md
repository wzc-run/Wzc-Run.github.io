---
layout:     post
title:      深度学习 | 实战6-利用TensorBoard实现卷积可视化
subtitle:
date:       2019-07-16
author:     JoselynZhao
header-img: img/post-bg-os-metro.jpg
catalog: true
tags:
    - Deep Learning
    - Python
    - TensorFlow
    - TensorBoard

---

[Github源码](https://github.com/zhaojing1995/DeepLearning.Advanceing/tree/master/DL_6/work)
## 要求
卷积可视化：


在Lenet中，分别使用ReLU及sigmoid激活函数，观察不同情况下，Lenet学习MNIST分类时，参数的变化。

并在最终训练好Lenet的情况下，观察分类操作前的最后一个全连接层fc2的84位特征向量，比较不同类型样本的fc2特征图。



要求：提交代码，文档。文档包括可视化截图。

（1）tensorboard可视化包括：loss， acc， w、b参数的分布， 卷积核、全连接矩阵的参数可视化，至少比较2种情况：[ReLU, sigmoid] 。


有兴趣可以尝试分别使用GradientDescentOptimizer 和AdamOptimizer的效果。

（2）fc2特征图不用tensorboard显示，plot绘出即可：

在模型训练好的情况下：绘制10个数字类型的fc2特征图，

每个类型一张fc2图，共10张fc2图：

每一张fc2图由同类型的100个不同样本的fc2特征按行拼接组成。

例如数字3的fc2特征图，由100个不同的数字3样本的fc2特征按行拼接组成。

故一张fc2图的大小为：100（行）*84（列）。

## 实验与结果
### 运行截图
图 1 程序运行截图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184747156.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
图1 右上角为分类操作前的最后一个全连接层fc2的84位特征向量，其样本数字为3.

### 全连接层fc2的84位特征图的实现
图 2 get_sample100()的实现
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184751606.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
通过get_sample100(label)获取100个同一数字的不同样本。参数lable取0-9，分别表示数字0-9。

图 3 get_sample100()的调用

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184757471.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
### 不同激活函数的控制实现
图 4 _activation_way()的实现
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184802440.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
在封装的Lenet类中，实现了一个关于选用哪一个激活函数的方法。这个_activation_way()在_build_network_graph()执行过程中凡是需要用到激活函数的地方都将被调用。

图 5 _build_network_graph()的改进实现

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184807437.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
### [Relu]Tensorboard可视化loss， acc， w、b参数的分布
图 6  [Relu] w和b 的可视化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184813549.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
图 7 [Relu] loss和accuracy的可视化

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184817983.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
### [Relu]卷积核、全连接矩阵的参数可视化
图 8 [Relu] 第一层卷积核5*5, 共六个
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184822705.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
图 9 [Relu] 第二层卷积核 5*5，共16个
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184827129.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
图 10 [Relu] 三个全连接层的参数可视化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184832151.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

### [sigmoid]Tensorboard可视化loss， acc， w、b参数的分布
图 11 [sigmoid] w和b的可视化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184837919.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
图 12 [sigmoid] loss和accuracy的可视化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184843496.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

### [sigmoid]卷积核、全连接矩阵的参数可视化
图 13 [sigmoid] 第一层卷积核5*5, 共六个
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184848225.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

图 14 [sigmoid] 第二层卷积核 5*5，共16个

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184854861.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
图 15 [sigmoid] 三个全连接层的参数可视化

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184859956.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
### [Relu]全连接层fc2的84位特征图
图 16 ：[Relu]100个不同的数字3 对应全连接层fc2的84位特征图
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019071718490654.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
图 17 :[Relu]100个不同的数字4 对应全连接层fc2的84位特征图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184909856.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
### [sigmoid]全连接层fc2的84位特征图
图 18 [sigmoid]100个不同的数字3 对应全连接层fc2的84位特征图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184916353.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
图 19 [sigmoid]100个不同的数字4 对应全连接层fc2的84位特征图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717184921656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## 源码展示
### LENET

```py
class Lenet():
    def __init__(self,mu,sigma,lr=0.02,act = 'relu'):
        self.mu = mu
        self.sigma = sigma
        self.lr = lr
        self.activation = act  # 默认是relu
        self._build_graph()

    def _build_graph(self,network_name = "Lenet"):
        self._setup_placeholders_graph()
        self._build_network_graph(network_name)
        self._compute_loss_graph()
        self._compute_acc_graph()
        self._create_train_op_graph()

    def _setup_placeholders_graph(self):
        self.x  = tf.placeholder("float",shape=[None,32,32,1],name='x')
        self.y_ = tf.placeholder("float",shape = [None,10],name ="y_")

    def _cnn_layer(self,scope_name,W_name,b_name,x,filter_shape,conv_stride,padding_tag="VALID",reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            conv_W = tf.Variable(tf.truncated_normal(shape=filter_shape, mean=self.mu, stddev=self.sigma), name=W_name)
            conv_b = tf.Variable(tf.zeros(filter_shape[3]),name=b_name)
            conv = tf.nn.conv2d(x, conv_W, strides=conv_stride, padding=padding_tag) + conv_b
            tf.summary.histogram("weights",conv_W)
            tf.summary.histogram("biases",conv_b)
            self._conv_visual(conv,conv_W,filter_shape)  #可视化
            return conv

    def _pooling_layer(self,scope_name,x,pool_ksize,pool_strides,padding_tag="VALID",reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            pool = tf.nn.max_pool(x, ksize=pool_ksize, strides=pool_strides, padding=padding_tag)
            return pool

    def _fully_connected_layer(self,scope_name,W_name,b_name,x,W_shape,reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            fc_W = tf.Variable(tf.truncated_normal(shape=W_shape, mean=self.mu, stddev=self.sigma),name=W_name)
            fc_b = tf.Variable(tf.zeros(W_shape[1]),name=b_name)
            fc = tf.matmul(x, fc_W) + fc_b
            tf.summary.histogram("weights",fc_W)
            tf.summary.histogram("biases",fc_b)
            self._full_visual(fc_W,W_shape)  #可视化
            return fc

    def _build_network_graph(self,scope_name="Lenet"):
        with tf.variable_scope(scope_name):
            conv1 =self._cnn_layer("conv1","w1","b1",self.x,[5,5,1,6],[1, 1, 1, 1])
            self.conv1 = self._activation_way(conv1)
            self.pool1 = self._pooling_layer("pool1",self.conv1,[1, 2, 2, 1],[1, 2, 2, 1])

            conv2 = self._cnn_layer("conv2","w2","b2",self.pool1,[5,5,6,16],[1, 1, 1, 1])
            self.conv2 = self._activation_way(conv2)
            self.pool2 = self._pooling_layer("pool2",self.conv2,[1, 2, 2, 1],[1, 2, 2, 1])

            self.fc0 = self._flatten(self.pool2)

            fc1 = self._fully_connected_layer("fc1","wfc1","bfc1",self.fc0,[400,120])
            self.fc1 = self._activation_way(fc1)

            fc2 = self._fully_connected_layer("fc2","wfc2","bfc2",self.fc1,[120,84])
            self.fc2 = self._activation_way(fc2)

            self.y = self._fully_connected_layer("fc3","wfc3","bfc3",self.fc2,[84,10])
            tf.summary.histogram("ypredict",self.y)

    def _activation_way(self,layer):
        if (self.activation == "relu"):
            layer = tf.nn.relu(layer)
        elif (self.activation =="sigmoid"):
            layer = tf.nn.sigmoid(layer)
        return layer #返回激活后的层


    def _flatten(self,conv):
        conv1 = tf.reshape(conv, [-1, 400])
        return conv1

    def _compute_loss_graph(self):
        with tf.name_scope("loss_function"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.y_,logits = self.y)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar("loss",self.loss)

    def _compute_acc_graph(self):
        with tf.name_scope("acc_function"):
            correct_prediction = tf.equal(tf.argmax(self.y,1),tf.argmax(self.y_,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

    def _create_train_op_graph(self):
        with tf.name_scope("train_function"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y,labels=self.y_))
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
            tf.summary.scalar("cross_entropy",self.cross_entropy)

    def _conv_visual(self,conv,conv_W, filter_shape):
        with tf.name_scope('visual'):
            x_min = tf.reduce_min(conv_W)  #不清楚这个是什么意思
            x_max = tf.reduce_max(conv_W)
            kernel_0_to_1 = (conv_W - x_min) / (x_max - x_min)

            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 2, 0, 1])
            conv_W_img = tf.reshape(kernel_transposed, [-1, filter_shape[0], filter_shape[1], 1])
            tf.summary.image('conv_w', conv_W_img, max_outputs=filter_shape[3])
            feature_img = conv[0:1, :, :, 0:filter_shape[3]]
            feature_img = tf.transpose(feature_img, perm=[3, 1, 2, 0])
            tf.summary.image('feature_conv', feature_img, max_outputs=filter_shape[3])

    def _full_visual(self,fc_W, W_shape):
        with tf.name_scope('visual'):
            x_min = tf.reduce_min(fc_W)
            x_max = tf.reduce_max(fc_W)
            kernel_0_to_1 = (fc_W - x_min) / (x_max - x_min)

            fc_W_img = tf.reshape(kernel_0_to_1, [-1, W_shape[0], W_shape[1], 1])
            tf.summary.image('fc_w', fc_W_img, max_outputs=1)

```

### main

```py
mnist = input_data.read_data_sets('../../../data/mnist', one_hot=True)
x_test = np.reshape(mnist.test.images, [-1, 28, 28, 1])
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),
                'constant')  # print("Updated Image Shape: {}".format(X_train[0].shape))
tf.logging.set_verbosity(old_v)


iteratons = 2000
batch_size = 64
ma = 0
sigma = 0.1
lr = 0.01


def get_sample100(label):
    sample100_x=[]
    sample100_y=[]
    count = 0
    for i in range(len(mnist.test.images)):
        if mnist.test.labels[i][label]==1:
            count+=1
            sample100_y.append(mnist.test.labels[i])
            sample100_x.append(mnist.test.images[i])
        if count>=100:
            break
    return sample100_x,sample100_y


def train_lenet(lenet):
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())

        tf.summary.image("input",lenet.x,3)
        merged_summary = tf.summary.merge_all()

        writer = tf.summary.FileWriter("LOGDIR/4/",sess.graph)  # 保存到不同的路径下
        # writer.add_graph(sess.graph)
        for ii in range(iteratons):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.reshape(batch_xs,[-1,28,28,1])
            batch_xs = np.pad(batch_xs,((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
            sess.run(lenet.train_step,feed_dict ={lenet.x:batch_xs,lenet.y_:batch_ys})
            if ii % 50 == 1:
                acc,s = sess.run([lenet.accuracy,merged_summary],feed_dict ={lenet.x:x_test,lenet.y_:mnist.test.labels})
                writer.add_summary(s,ii)
                print("%5d: accuracy is: %4f" % (ii, acc))
        sample100_x,sample100_y = get_sample100(4) #随便选了一个label 输入0-9的值
        sample100_x = np.reshape(sample100_x,[-1,28,28,1])
        sample100_x = np.pad(sample100_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        x_min = tf.reduce_min(lenet.fc2)
        x_max = tf.reduce_max(lenet.fc2)
        fc2 = (lenet.fc2 - x_min) / (x_max - x_min)
        fc2 = sess.run(fc2,feed_dict={lenet.x:sample100_x,lenet.y_:sample100_y})
        plt.imshow(fc2)
        plt.show()

        print('[accuracy,loss]:', sess.run([lenet.accuracy], feed_dict={lenet.x:x_test,lenet.y_:mnist.test.labels}))


if __name__ =="__main__":
    act = "sigmoid" #或者act =“relu”
    lenet = Lenet(ma,sigma,lr,act)
    train_lenet(lenet)
```

