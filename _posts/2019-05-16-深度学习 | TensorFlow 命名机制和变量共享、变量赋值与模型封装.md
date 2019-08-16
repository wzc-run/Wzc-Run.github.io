---
layout:     post
title:      深度学习 | TensorFlow 命名、变量、封装
subtitle:   TensorFlow 命名机制和变量共享、变量赋值和更新、模型封装
date:       2019-05-16
author:     JoselynZhao
header-img: img/post-bg-os-metro.jpg
catalog: true
tags:
    - Deep Learning
    - Python
    - TensorFlow
---

# 命名机制与变量共享
Variable 变量 (一般表达参数)、Tensor(操 作输出)、操作 Operation、Placeholder 输 入都有名字

当模型复杂的时候，需要有效的命名机制: 方便、清晰

## TF 中的命名机制
有效的命名机制:
1. 起名方便 (不用从 w1 起到 w1024...)
2. 甚至不显式写名，也能自动给个好找的名字 3. 同时避免重名带来的名字冲突

**TF 采用 scope, 一种层级的命名管理机制**:例:vgg16/conv1/w:0 
用一级级的 scope 名字空间来管理变量
可以不用显示起名，TF 会按规则自动起名

两种 scope:tf.name_scope, tf.variable_scope 用于给名字一级一级 指定管理空间，例如:

```py
with tf.name_scope('name_sp1') as scp1:
	with tf.variable_scope('var_scp2') as scp2: 
		with tf.name_scope('name_scp3') as scp3:
			a = tf.Variable(1, name='a') #name_sp1/var_scp2/name_scp3/a:0
```

tf.name_scope, tf.variable_scope 主要对 tf.get_variable 创建的变
 量管理有区别，主要涉及变量重用

## tf.name_scope, tf.variable_scope 管理方式的异同
对 Tensor 或 tf.Variable 创建的变量，tf.name_scope, tf.variable_scope 管理基本相同，一层层加前缀;无视 python 自带 name_scope 限制; 无视 variable_scope 的 reuse

Tensor,tf.Variable 不起名，TF 用操作名起名 (Variable,Add ...+value_index (:0,:1...):tensor 第几个输出); 起重名，TF 会改为 不一样的 name(_1,_2...); 对应的 Op, 没有 value_index


```py
with tf.name_scope('name_sp1') as scp1:
	with tf.variable_scope('var_scp2') as scp2:
		with tf.name_scope('name_scp3') as scp3:
			a = tf.Variable(1, name='a')
			print(a) #name_sp1/var_scp2/name_scp3/a:0 
			a1 = tf.Variable(1, name='a')
			print(a1) #name_sp1/var_scp2/name_scp3/a_1:0 		
			b=tf.Variable(1)
			print(b) #name_sp1/var_scp2/name_scp3/Variable:0 	
			c=tf.Variable(1)
			print(c) #name_sp1/var_scp2/name_scp3/Variable_1:0 
			a_b=tf.add(a,b)
			print(a_b) #name_sp1/var_scp2/name_scp3/Add:0 
			a_c=tf.add(a,c)
			print(a_c) #name_sp1/var_scp2/name_scp3/Add_1:0
```

tf.name_scope, tf.variable_scope 的异，主要针对对 tf.get_variable 创建变量

**tf.variable_scope 主要是配合 tf.get_variable 进行变量重用 (共享)** 

**variable 两种创建方式:tf.Variable, tf.get_variable
设计目的不同, 创建方式不同:一个主要用于新创建，一个主要用 于重用**

- a=tf.Variable(initial_value=[4.], name=’var4’, dtype=tf.float32) 
- b=tf.get_variable(name=’var3’, shape=[1], dtype=tf.float32, initializer=initializer)
-  tf.Variable:initial_value(直接数值或初始化值函数) **必显式初始化**; tf.get_variable:initializer(初始化算子 xxx_initilizer) 可选 
- **创建新变量:tf.Variable:name 可重 (TF 自动别名);tf.get_variable: name 不能重**

**tf.get_variable:无视 name_scope, 只看 variable_scope** 

```py
with tf.name_scope('name_sp1') as scp1:
	with tf.variable_scope('var_scp2') as scp2: 
		with tf.name_scope('name_scp3') as scp3:
			a = tf.Variable(1, name='a') 
			b = tf.get_variable('b',[1])
```
等同于

```py
with tf.name_scope('name_sp1') as scp1:
	with tf.variable_scope('var_scp2') as scp2: 
		with tf.name_scope('name_scp3') as scp3:
			a = tf.Variable(1, name='a')

with tf.variable_scope('var_scp2') as scp2: 
	b = tf.get_variable('b',[1])
```

tf.get_variable 变量重用看 variable_scope 的 reuse 属性，false 则 创建新变量，name 不能重
reuse 下，tf.get_variable:根据 name 值，返回该变量，如果该name 不存在的话，则进行创建。


**get_variable 重用变量需要看 variable_scope 的 reuse 属性**

```py
with tf.variable_scope('scp',reuse=True) as scp: 
	a = tf.get_variable('a',[1]) #报 错
```
reuse=True, 强制共享, 不存在共享变量，所以报错


```py
with tf.variable_scope('scp',reuse=False) as scp: 
	a = tf.get_variable('a',[1])
	a = tf.get_variable('a') # 报错
```
reuse=False, 强制创建, 已存在同名变量，报错慎用 tf.AUTOREUSE, 如果明确知道下面的变量是要重用的


```py
with tf.variable_scope('scp',reuse=True) as scp:
	a = tf.get_variable('a')
	a = tf.get_variable('a') #a只 在 这 里 创 建 报 错
```

get_variable 第一次创建，要至少给 shape


**variable_scope 的 reuse 属性另一种设置方法**：
variable_scope 的 reuse 属性也可通过 scope.reuse_variable 在名字
空间内设置

```py
import tensorflow as tf
with tf.variable_scope('foo') as scp:
	aaa = tf.get_variable('aaa',[1]) bbb = tf.get_variable('bbb',[1])
	scope.reuse_variable()
	ccc = tf.get_variable('aaa')
print (aaa.name) #foo/aaa:0
print (bbb.name) #foo/bbb:0 print (ccc.name) #foo/aaa:0
```


## 案例:RNN 中的共享变量

```py
import  tensorflow as tf

def rnn(inputs,state,hidden_size):
    in_x = tf.concat([inputs,state],axis =1)
    W_shape = [int(in_x.get_shape()[1]),hidden_size]
    b_shape = [1,hidden_size]
    
    W = tf.get_variable(shape = W_shape, name="weight")
    b = tf.get_variable(shape = b_shape,name = "bias")
    
    out_linear = tf.nn.bias_add(tf.matmul(in_x,W),b)
    output = tf.nn.tanh(out_linear)
    return output
```


## 名字空间、变量重用简单总结
不考虑变量重用:
最好不用 tf.get_variable, 只用 tf.Variable 
name_scope,variable_scope 随意

考虑变量重用:
只用 tf.get_variable
用 variable_scope 划分名字空间
scope 的 reuse 属性用 tf.AUTOREUSE 方便复用定义

**搞不清将来要不要复用变量:**
用 tf.get_variable 创建变量
最好用 variable_scope 划分名字空间
scope 的 reuse 属性设置为每层传进来的参数 (...,reuse=reuse_flag)

## 案例:训练测试模型共享

```py
import tensorflow as tf

if __name__ =="__main__":
    X=tf.placeholder(tf.float32)
    def model(X):
        w=tf.Variable(name="w", initial_value=tf. random_normal(shape=[1]))
        m=tf.multiply(X,w)
        return m
    def train_graph(X):
        m=model(X)
        a=tf.add(m,X)
        return a
    def test_graph(X):
        m=model(X)
        b=tf.add(m,X)
        return b
    a=train_graph(X)
    b=test_graph(X)

    X_in = 1.2
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ar = sess.run(a, feed_dict={X:X_in})
        br = sess.run(b, feed_dict={X:X_in})
        print("ar=", ar)
        print("br=", br)
```
tf.Variable 重建了 w，不是原来的 .


# 变量赋值与更新

## 变量初始化
变量常用来表达我们机器学习中不断调整的参数 调参数需要干两件事:1. 赋初值;2. 更新参数 变量的初始化赋值主要分为两步:
1. 定义变量时给定初始化值函数：

```py
a = tf.Variable(initial_value = ...) 
b = tf.getvarialbe(initializer = ...)
```
2. Session 中执行初始化方法:

```py
init = tf.global_variables_initializer() #或 
init = tf.variable_initializer([a, b]) 
sess.run(init)
```

## 变量更新
**TF 通过 assign operation 赋值机制，来修改、更新变量值**
包括 assign(ref, value, validateshape = None, uselocking = None, name = None) 及一些变种:assign_add()、assign_sub()

**用法注意:**不是我们习惯的 ref = assign(assign_value) 而是 **assign_tensor = assign(ref, assign_value)**

**返回值 assign_tensor 是一个在赋值完成后，保留”ref” 的新值的张 量**

后续计算需要更新的值的时候，引用的是 assign_tensor 

不但可以改值，变量定义参数 validate_shape = False，还可以改变量形状

## 变量更新案例 1: 不用 assign

```py
import tensorflow as tf
import numpy as np 
b=tf.Variable([2],name='b')
b=tf.add(b,1) 
out=b*2
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) 
	for i in range(3):
		print(sess.run([out,b])
#[6,3] 
#[6,3] 
#[6,3]
```
可以看到输出结果是一样的

## 变量更新案例 2:assign 与数据驱动的机制
**数据驱动图，从一次 fetch 的最高节点倒推到涉及的最初节点，从下往上逐个计算**

```py
import tensorflow as tf 
import numpy as np
b=tf.Variable([2],name='b')
assign_op=tf.assign_add(b,[1]) 
out=assign_op*2
with tf.Session() as sess: 
	sess.run(tf.global_variables_initializer())
	for i in range(3): 
		print(sess.run([out,b])
#[6,3]
#[8,4]
#[10,5]
	y = out * b
	print(sess.run([y])) 
	#[72] 
	#为了得到y这个结果，计算图又会再跑一遍。
x=b+out
with tf.Session() as sess:
	print(sess.run([x]))
	# 报错，b未初始化
	
```

## 变量更新案例 3

```py
import tensorflow as tf
import  numpy as np

if __name__ =="__main__":
    var = tf.Variable(0.,name='var') #除第一次以外，不会通过这句初始化，而是直接输出当前值
    const = tf.constant(1.)
    add_op = tf.add(var,const,name='myAdd')

    assign_op = tf.assign(var, add_op, name='myAssign')

    out1 = assign_op*1
    out2 = assign_op*2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3):
            print "var:",sess.run(var),sess.run(out1),sess.run(out2)
            # 每个run的调用，都会重新执行一遍计算图

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3):
            print "var:",sess.run(var),sess.run([out1,out2])
'''
var: 0.0 1.0 4.0
var: 2.0 3.0 8.0
var: 4.0 5.0 12.0
var: 0.0 [1.0, 2.0]
var: 1.0 [2.0, 4.0]
var: 2.0 [3.0, 6.0]
'''
```

# class 封装

## 面向对象基本概念
- 类 (Class): 描述具有相同的属性和方法的对象的集合。对象是类 的实例。
- 对象:通过类定义的数据结构实例。对象包括两个数据成员(类变量和实例变量)和方法。
- 实例化:创建一个类的实例，类的具体对象。 其他面向对象概念:继承(不支持多态); 私有、公有、保护


## python 类声明、实例化举例

```py
#创建类
class Gre:
#类中的方法
    def Test(self): #类中方法的第一个参数必须是self，代表类的实例,
    #类似于C++中的this指针(self也可以统一换成其他名字)!
        pass
        #空 语 句， 啥 也 不 干， 保 持 程 序 结 构 完 整 性
        #定 义 空 函 数 会 报 错， 没 想 好 写 啥 就 先pass
    def Hi(self):
        print('Hi')
    def Hello(self,name):
        print("I'm␣%s" %name)

#根据类Gre声明 创建或说实例化类Gre的一个对象obj 
obj=Gre() #实 例 化 一 定 要 加'()'!
obj.Hi() #调 用Hi方 法

```

## python class 封装
封装(Encapsulation)、继承和多态:**面向对象的三大特征** 
封装含义:该藏的，把该露的露 
**封装重要的原因: 隔离复杂度**
其他目的:
- 隐藏类的实现细节
- 限制对属性的不合理访问
- 便于修改，提高代码的可维护性
- 内部检查机制，便于数据维护

## 对 TF 模块进行封装
TF 提供了很多比较基础的操作，相对 Keras 等其他框架更灵活，但构建网络相对繁琐

```py
import  tensorflow as tf

def _cnn_layer(self, scope_name, W_name, b_name, x, filter_shape , conv_strides, padding_tag='VALID'):

    with tf.variable_scope(scope_name): #添 加 更 灵 活 的 操 作， 但 构 建 繁琐
        conv_W = tf.get_variable(W_name, dtype=tf.float32, initializer=tf.truncated_normal(shape=filter_shape, mean= self.config.mu, stddev=self.config.sigma))
        conv_b = tf.get_variable(b_name, dtype=tf.float32, initializer=tf.zeros(filter_shape[3]))
        conv = tf.nn.conv2d(x, conv_W, strides=conv_strides, padding=padding_tag) + conv_b
        tf.summary.histogram('weights', conv_W)
        tf.summary.histogram('biases', conv_b)
        tf.summary.histogram('activations', conv)
        return conv
```

很难直接用上述逐条操作构建大型网络，所以将一些模块 (例如卷积层) 封装为一个函数
同理，常用来构建算法的基础网络 (base net)，也封装为类，便于维护和重复使用

## TF 网络封装案例:Lenet
我们封装一个 TF 模型想封装什么呢?
1. 一些可配置的参数。
2. 网络画图流程 (只是画 graph，建议用 xxx_graph 和其他方法区分开)。
3. 一些外部可调用的可运行的操作 (例如变量修改等，需要 sess.run)

```py
from tensorflow as tf
class Lenet(Network):
    #Network 父类，表示Lenet继承于Network
    # __init__类 似C++构 造 函 数， 实 例 化 创 建 对 象 时 调 用， 做 对 象 的 一 些 初始化工作
    def __init__(self, config):
        self.config = config
        self._build_graph()
        #没 有 前 缀 “_",公 有;"__",私 有;"_",保 护 
        # #涉及网络的所有画图build graph过程，常用一个build graph封起来
        def _build_graph(self, network_name='Lenet'):
            self._setup_placeholders_graph()
            self._build_network_graph(network_name)
            self._compute_loss_graph()
            self._compute_acc_graph()
            self._create_train_op_graph()
            self.merged_summary = tf.summary.merge_all()
```

常将网络层 build graph 过程, 封为类内的一个个函数

```py
    def _setup_placeholders_graph(self):
        self.x = tf.placeholder("float", shape=[None, 32, 32, 1], name='x')
        self.y_ = tf.placeholder("float", shape=[None, 10], name='y_ ')
        self.keep_prob = tf.placeholder("float", name='keep_prob')

    def _cnn_layer(self, scope_name, W_name, b_name, x, filter_shape, conv_strides, padding_tag='VALID'):
        pass
    def _pooling_layer(self, scope_name, x, pool_ksize,pool_strides, padding_tag='VALID'):
        pass
    def _fully_connected_layer(self, scope_name, W_name, b_name, x, W_shape):
        pass
```


通常将整个网络的 build graph 过程，封装为一个 build network 函数，其由一层层的网络层构建函数实现.

```
    def _build_network_graph(self, scope_name): 
        with tf.variable_scope(scope_name):
            conv1 = self._cnn_layer('layer_1_conv', 'conv1_w', ' conv1_b', self.x, (5, 5, 1, 6), [1, 1, 1, 1])
            self.conv1 = tf.nn.relu(conv1)
            self.pool1 = self._pooling_layer('layer_1_pooling', self.conv1, [1, 2, 2, 1], [1, 2, 2, 1])
            self.y_predicted = tf.nn.softmax(self.logits)
            tf.summary.histogram("y_predicted", self.y_predicted)

    def _compute_loss_graph(self):
        with tf.name_scope("loss_function"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar("cross_entropy", self.loss)
```


## TF 网络实例化与外部方法调用
注意实例化的时候，哪些在画图?哪些在做实际计算?
有的常在 __init__ 中 build graph, 那么创建对象时，已经把网络画好 否则需显示调用 build graph 画好图，其他计算调用也一般要在 build graph 之后调用


```py
class vgg16(object):
    def __init__(self):
        #当在创建的时候运行画图
        self._build_graph()

    def fix_conv1_from_RGB_to_BGR(self,sess):
        #外 部 调 用 的 计 算 过 程， 需 要 传 递 相 关 的sess会 话
        restorer_conv1.restore(sess, pretrained_model)
        sess.run(tf.assign(var_to_fix[...],new_value))


#main.py

net=vgg16() #此时整个计算图里已经添加了vgg16_obj的计算图 ....
sess.run(init) #先 初 始 化 好 变 量
#再恢复一些需要的预训练值
restorer.restore(sess, pretrained_model)
#再修改一些值
net.fix_conv1_from_RGB_to_BGR(sess, pretrained_model)
```


## TF 网络继承与覆盖
有时，一个算法可以配置不同的骨干网以便扩展 可用类继承机制，将骨干网共性进行抽取，表达为父类 具体网表达为子类，用覆盖的方式实现具体网方法的特性，例如具体 网络的 build network 方法

```py
class Network(object):
    def __init__(self):
        self._layers={}

    def _img_to_classifier1(self,is_training, reuse=None):
        raise NotImplementedError
    def _add_train_summary(self, var):
        #子 类 共 性 操 作， 父 类 已 具 体 实 现
        pass
    def _add_losses(self, var):
        #子 类 共 性 操 作， 父 类 已 具 体 实 现
        pass

    #vgg16.py
class vgg16(Network):
    # Network 父类中的其他方法,例如:_add_train_summary,_add_losses等vgg16都可以调用,vgg16类只用写其相比
    #Network不同的方法

    def __init__(self):

        Network.__init__(self)  # 初 始 化 时， 先 调 用 父 类 初 始 化 方 法

    def _img_to_classifier1(self, is_training, reuse=None):
        # 具体实现覆盖父类方法
        pass
```

 


