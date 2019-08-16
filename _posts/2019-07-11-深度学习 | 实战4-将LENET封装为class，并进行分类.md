---
layout:     post
title:      深度学习 | 实战4-将LENET封装为class，并用此封装好的lenet对minist进行分类
subtitle:
date:       2019-07-11
author:     JoselynZhao
header-img: img/post-bg-os-metro.jpg
catalog: true
tags:
    - Deep Learning
    - Python
    - TensorFlow

---
[Github源码](https://github.com/zhaojing1995/DeepLearning.Advanceing/tree/master/DL_4/work)
## 要求
将LENET封装为class，并用此封装好的lenet对minist进行分类。

有关lenet定义请参考卷积网络课件最后2页；class封装的内容，请参考class封装课件

### 1. lenet 结构如附件描述。注意：
（1）lenet 输入为32x32，而minist为28x28，故需要先对数据进行填充，例如：

```py
import numpy as np

#Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
```

（2）lenet 输出 10位的 one-hot形式的输出 logits, 故minist的标签读取需采用one-hot的形式。

采用softmax 交叉熵作为损失函数。用softmax进行分类。


### 2. 在init函数中传入初始化变量所需的mu， sigma参数，以及其他所需定制化参数。

例如：

```py
def __init__(self,mu):

    self.mu=mu

```

设计需要的输入输出接，例如，如果想把对外数据的交互也封装在class里：

```py
self.raw_input_image = tf.placeholder(tf.float32, [None, 784]) 或者需要的进一步变换，例如

self.input_x = tf.reshape(self.raw_input_image, [-1, 28, 28, 1])

```

或者把外部交互的事情交给外部去做，class只是想实现一个纯净的net计算通路：

```py
self.input_x=input (input是你外部给的输入引用)
```


### 3. 对lenet中常见的conv层，fc层，pooling层定义统一的定制化功能层graph绘图函数. 为层次化组织网络，给每个层定义一个不同的名字空间，例如：

```py
def conv(w_shape, scope_name, .......):

    with tf.variable_scope(scope_name) as scope:

        xxxx.....

```

4. 绘制整个网络计算图的函数，_build_graph(). 这里要求调用_build_graph()的过程放在 _init_函数里，这样外部每调用并生成一个class的实例，实际上就自动绘制了一次lenet。


_build_graph()绘制整个lenet的时候，调用之前你定义的各个功能层，并逐层搭建出整个网络。期望网络对外的输出tensor引用都用self记录，例如：



```py
def __init__(self, config):
   self.config = config
   self._build_graph() 

    .....
def _build_graph(self, network_name='Lenet'):
    self._setup_placeholders_graph()
    self._build_network_graph(network_name)
    self._compute_loss_graph()
    self._compute_acc_graph()
    ....
```

      
5. 在外部调用该模块并通过实例化实现对lenet的绘制，例如：



```py
......

from lenet import Lenet （lenet.py 里定义的 class Lenet）

.......

lenet_part = Lenet() 

```

这样调用一下已经完成了lenet的绘制了，你需要引用的lenet中间的tensor都保存在lenet_part里


例如：

```py
sess.run(train_op,feed_dict={lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]})
```



要求：用class封装好的lenet对minist进行分类，训练和模型定义分开成两个文件train.py, lenet.py，打印训练和测试截图，测试分类准确率ACC。


## 实验与结果
运行截图
图 1
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717183206257.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

图 2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717183212634.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

参数设置
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717183218503.png)


## 源码展示
### LENET

```py


class Lenet():
    def __init__(self,mu,sigma,lr=0.02):
        self.mu = mu
        self.sigma = sigma
        self.lr = lr
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

    def _cnn_layer(self,scope_name,W_name,b_name,x,filter_shape,conv_stride,padding_tag="VALID"):
        with tf.variable_scope(scope_name):
            conv_W = tf.Variable(tf.truncated_normal(shape=filter_shape, mean=self.mu, stddev=self.sigma), name=W_name)
            conv_b = tf.Variable(tf.zeros(filter_shape[3]),name=b_name)
            conv = tf.nn.conv2d(x, conv_W, strides=conv_stride, padding=padding_tag) + conv_b
            return conv

    def _pooling_layer(self,scope_name,x,pool_ksize,pool_strides,padding_tag="VALID"):
        with tf.variable_scope(scope_name):
            pool = tf.nn.max_pool(x, ksize=pool_ksize, strides=pool_strides, padding=padding_tag)
            return pool
    def _fully_connected_layer(self,scope_name,W_name,b_name,x,W_shape):
        with tf.variable_scope(scope_name):
            fc_W = tf.Variable(tf.truncated_normal(shape=W_shape, mean=self.mu, stddev=self.sigma),name=W_name)
            fc_b = tf.Variable(tf.zeros(W_shape[1]),name=b_name)
            fc = tf.matmul(x, fc_W) + fc_b
            return fc

    def _build_network_graph(self,scope_name):
        with tf.variable_scope(scope_name):
            conv1 =self._cnn_layer("conv1","w1","b1",self.x,[5,5,1,6],[1, 1, 1, 1])
            self.conv1 = tf.nn.relu(conv1)
            self.pool1 = self._pooling_layer("pool1",self.conv1,[1, 2, 2, 1],[1, 2, 2, 1])
            conv2 = self._cnn_layer("conv2","w2","b2",self.pool1,[5,5,6,16],[1, 1, 1, 1])
            self.conv2 = tf.nn.relu(conv2)
            self.pool2 = self._pooling_layer("pool2",self.conv2,[1, 2, 2, 1],[1, 2, 2, 1])
            self.fc0 = self._flatten(self.pool2)
            fc1 = self._fully_connected_layer("fc1","wfc1","bfc1",self.fc0,[400,120])
            self.fc1 = tf.nn.relu(fc1)
            fc2 = self._fully_connected_layer("fc2","wfc2","bfc2",self.fc1,[120,84])
            self.fc2 = tf.nn.relu(fc2)
            self.y = self._fully_connected_layer("fc3","wfc3","bfc3",self.fc2,[84,10])

    def _flatten(self,conv):
        conv1 = tf.reshape(conv, [-1, 400])
        return conv1

    def _compute_loss_graph(self):
        with tf.name_scope("loss_function"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.y_,logits = self.y)
            self.loss = tf.reduce_mean(loss)

    def _compute_acc_graph(self):
        with tf.name_scope("acc_function"):
            correct_prediction = tf.equal(tf.argmax(self.y,1),tf.argmax(self.y_,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    def _create_train_op_graph(self):
        with tf.name_scope("train_function"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y,labels=self.y_))
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

```

### train

```py
from lenet import  *

if __name__ =="__main__":
    mnist = input_data.read_data_sets('../../../data/mnist', one_hot=True)
    x_test = np.reshape(mnist.test.images,[-1,28,28,1])
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')    # print("Updated Image Shape: {}".format(X_train[0].shape))
    tf.logging.set_verbosity(old_v)

    iteratons = 30000
    batch_size = 64
    ma = 0
    sigma = 0.1
    lr = 0.01
    mylenet = Lenet(ma,sigma,lr)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ii in range(iteratons):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.reshape(batch_xs,[-1,28,28,1])
            batch_xs = np.pad(batch_xs,((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

            sess.run(mylenet.train_step,feed_dict ={mylenet.x:batch_xs,mylenet.y_:batch_ys})
            if ii % 500 == 1:
                acc = sess.run(mylenet.accuracy,feed_dict ={mylenet.x:x_test,mylenet.y_:mnist.test.labels})
                print("%5d: accuracy is: %4f" % (ii, acc))

        print('[accuracy,loss]:', sess.run([mylenet.accuracy], feed_dict={mylenet.x:x_test,mylenet.y_:mnist.test.labels}))


```


