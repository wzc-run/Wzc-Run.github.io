---
layout:     post
title:      深度学习 | TensorFlow 可视化
subtitle:   使用tensorboard查看深度神经网络的内部细节
date:       2019-05-21
author:     JoselynZhao
header-img: img/post-bg-os-metro.jpg
catalog: true
tags:
    - Deep Learning
    - Python
    - TensorFlow
    - TensorBoard
---


# TensorBoard简介
## 什么是TensoBoard
TF 提供的可视化工具，通过网页浏览的方式可视化展示与我们 TF 构建的模型相关的信息

## Tensorboard 能帮助我们看到什么
- 看网络结构:graph 
- 看训练过程中的指标:loss、acc... 
- 看参数变化:参数均值、方差、分布 
- 看中间结果:中间生成图像、语音等 
- 看数据关系:样本、分类结果分布

## 怎样启动 Tensorboard?
1 在 TF 程序中添加记录并存储日志 events..xxxx
2. 命令行启动读取日志文件
3. tensorboard –logdir=logs (logs 目录不能包含中文，或者空格) 4. 打开浏览器(用 chrome)，按命令行提示输入本机地址和 tensorboard 通讯端口，刷新浏览例:http://DESKTOP-xxx:6006 5. 输入:http://127.0.0.1:6006 也可以


## Tensorboard 基础
**Summary 类： 负责汇总数据 并写入事件文件**
**使用TensorBoard 展示数据，需要在执行TensorFlow计算图的过程中，将各种类型的数据汇总并记录到日志文件中。** 然后使用TensorBoard读取这些日志文件，解析数据并产生数据可视化的web页面，让我们可以在浏览器中观察各项中汇总数据。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019060115545561.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## TF 程序中添加 Tensorboard 日志记录方法
1. **TF程序中添加TensorBoard日志记录方法：**
对感兴趣 tensor 添加记录操作:summary operation：
例如`tf.summary.scalar(′ name′ , variable)`


2. **汇总需要写入日志的记录**：

使用`merged = tf.merge_all_summaries()`

3. **实例化一个日志书写器**：
使用`summary_writer = tf.summary.FileWriter(logdir, graph = None, flush_secs = 120, max_queue = 10)`
可选同时传入模型graph。 或之后用`add_graph(graph,global_step= None) 添加

4. 运行汇总节点，得到汇总结果：`summary = sess.run(merged)`
5. 调用书写器实例将汇总日志写入文件`summary_writer.add_summary(summary, global_step = i)`
6. 缓存写入磁盘文件，关闭文件：`summary_writer.flush()` 写入，否则`flush_secs`间隔写入`summary_writer.close()`,写入加关闭文件


# 通过TensorBoard查找编程错误
**tf.summary.FileWriter(n)**：一个用于输出 Tensorboard 数据的Python类

```py
sess = tf.Session()
writer  = tf.summary.FileWriter(LOGDIR+'2')
writer.add_graph(sess.graph)
```

* 在 TensorBoard 里指定变量名，提高流图可读性*

## 输出中间数据
- Summary(n) 用于输出中间数
    据
- tf.summary.scalar (标量)
- tf.summary.image (图片)
- tf.summary.audio (声音)
- tf.summary.histogram (统计数 据)

## 添加代码以查看中间数据

```py
tf.summary.image('input',x_image,3)
tf.summary.scalar('cross_entropy',cross_entropy)
tf.summary.scalar('accuracy',accuracy)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602143947182.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602143957438.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

##  变量不能都初始化为 0
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602144052822.png)
我们的损失是 cross entropy，要和 softmax 一起配对使用

# TensorBoard进行超参数搜索
## 不同学习率以及卷积层数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190602203625402.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

# 梯度、特征可视化
## CNN可视化
卷积核:滤波器、模式相关性
有意义的卷积核:有一定规律的 pattern，不是特别随机，特别是 底层
不同层的模式规律不同
参数的泛化性能?冗余?稀疏？

## CNN 可视化 -卷积核

weight 可视化出来的效果图，左图存在很多噪点，右图则比较平滑，
举例:下图两张图都是将一个神经网络的第一个卷机层的 filter 出现左图这个情形，往往意味着我们模型训练过程出现了问题。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20190603071527910.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
## 梯度可视化
梯度可视化对网络调参的好处在训练过程中，由于设置了较高的 学习率，学习跨度太大，中间层的梯度可能会随着训练过程的推进逐 渐变为 0。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190603071609742.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190603071618420.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190603071627513.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190603071635671.png)

梯度消失当出现梯度消失或梯度弥散时，梯度图能很好的表现出
来。

# 数据分布可视化

