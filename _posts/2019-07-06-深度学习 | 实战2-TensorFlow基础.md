---
layout:     post
title:      深度学习 | 实战2-TensorFlow基础
subtitle:
date:       2019-07-06
author:     JoselynZhao
header-img: img/post-bg-os-metro.jpg
catalog: true
tags:
    - Deep Learning
    - Python
    - TensorFlow

---

[GitHub源码](https://github.com/zhaojing1995/DeepLearning.Advanceing/tree/master/DL-2/workd)
## 要求 
假设有函数y = cos(ax + b), 其中a为学号前两位，b为学号最后两位。首先从此函数中以相同步长（点与点之间在x轴上距离相同），在0<(ax+b)<2pi范围内，采样出2000个点，然后利用采样的2000个点作为特征点进行三次函数拟合(三次函数形式为 y = w1 * x + w2 * x^2 + w3 * x^3 + b, 其中wi为可训练的权值，b为可训练的偏置值，x和y为输入的训练数据 ) 。要求使用TensorFlow实现三次函数拟合的全部流程。拟合完成后，分别使用ckpt模式和PB模式保存拟合的模型。然后，针对两种模型存储方式分别编写恢复模型的程序。两个模型恢复程序分别使用ckpt模式和PB模式恢复模型，并将恢复的模型参数（wi和b）打印在屏幕上，同时绘制图像（图像包括所有的2000个采样点及拟合的函数曲线）。

请提交文档，内容包括模型参数配置、程序运行截图、拟合的三次函数以及绘制的图像，同时提交三个python脚本文件：1. 函数拟合及两种模式的模型保存程序2. ckpt模型恢复程序，3. PB模型恢复程序。

## 实验与结果
参数配置
图 1
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717164422406.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
图 2
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717164429265.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
一开始所有的参数配置都是按默认的来设定，程序编写完成之后，初步运行，发现loss会稳定在0.5左右，降不下去。
首先是想到了更换优化器，图2中可以看到被注释掉了的代码中尝试了各种各样的优化器，但是效果都很不好，训练了好几次之后，最低的loss 仍有0.46之高。
于是想到了调整变量的初始化，在图1的注释中，记录了各种参数配置 对应的loss值。可以看到，通过调整mean值，loss值最低降到了0.33。 最后又通过调整stddev将最后的loss值降到了0.04以下。训练结果如图3所示。
图 3
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717164438944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
训练的迭代次数为50000。
最后四位数分别为w1、w2、w3和b的值。


程序运行
图 4  ckpt模式恢复模型运行截图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717164452418.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
图 5 PB模式恢复模型运行截图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717164502217.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)



实验结果
图 6 ckpt.png
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717164514197.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

图 7 PB.png
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190717164521949.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## 源码展示
### 主程序

```py

save_path_ckpt = './save/dl-2-work.ckpt'
save_path_pb = './save/dl-2-work.pb'



if __name__  =="__main__":
    # import tensorflow.contrib.eager as tfe
    # tfe.enable_eager_execution()
    school_number = 18023032
    a =18.0
    b =32.0
    N = 2000
    x = np.linspace(float(-b/a),(2*math.pi-b)/a,N).reshape([-1,1])
    y = np.cos(a*x+b).reshape([-1,1])


    data = tf.placeholder(tf.float32,[None,1])
    label = tf.placeholder(tf.float32,[None,1])

    print(tf.shape(data))
    print(tf.shape(label))
    # print(tf.shape(data))


    w1 = tf.Variable(tf.random_normal([1,1],mean=0, stddev=200),dtype=tf.float32, name='s_w1')
    w2 = tf.Variable(tf.random_normal([1,1],mean=100, stddev=200),dtype=tf.float32, name='s_w2')
    w3 = tf.Variable(tf.random_normal([1,1],mean=200, stddev=200),dtype=tf.float32, name='s_w3')
    b = tf.Variable(tf.random_normal([1,1],mean=200, stddev=100),dtype=tf.float32, name='s_b')

    # 200 100   0.35
    # 200 10    0.36
    # 300 200 100 50 0.34995785
    # 500 100 50 100 0.33685145

    y_label = tf.add(tf.add(tf.matmul(data,w1),tf.matmul((data**2),w2)),tf.add(tf.matmul((data**3),w3),b),name ='op-to-store')
    loss = tf.reduce_mean(tf.square(label - y_label))
    train = tf.train.AdamOptimizer(0.2).minimize(loss)
    # train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # train = tf.train.AdadeltaOptimizer(0.1).minimize(loss)
    # train = tf.train.AdagradOptimizer(0.1).minimize(loss)
    # train = tf.train.FtrlOptimizer(0.01).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            sess.run(train, feed_dict={data:x,label:y})
            if i % 500 == 0:
                log_loss = sess.run(loss,feed_dict={data:x,label:y})
                print(i,log_loss)
        print(sess.run(w1))
        print(sess.run(w2))
        print(sess.run(w3))
        print(sess.run(b))

        # ckpt 保存
        save_path1 = saver.save(sess,save_path_ckpt)

        # pb保存
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op-to-store'])
        #constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['s_w1','s_w2','s_w3','s_b'])
        with tf.gfile.FastGFile(save_path_pb, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
```

### PB_load

```py


save_path_pb = './save/dl-2-work.pb'


if __name__ =="__main__":
    with tf.Session() as sess:
        with gfile.FastGFile(save_path_pb, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        sess.run(tf.global_variables_initializer())
        w1 = sess.graph.get_tensor_by_name('s_w1:0')
        w2 = sess.graph.get_tensor_by_name('s_w2:0')
        w3 = sess.graph.get_tensor_by_name('s_w3:0')
        b = sess.graph.get_tensor_by_name('s_b:0')
        print(sess.run(w1))
        print(sess.run(w2))
        print(sess.run(w3))
        print(sess.run(b))
        w1 = w1.eval()
        w2 = w2.eval()
        w3 = w3.eval()
        b = b.eval()

    school_number = 18023032
    aa = 18.0
    bb = 32.0
    N = 2000
    x1 = np.linspace(-bb / aa, (2 * math.pi - bb) / aa, N)
    y1 = np.cos(aa * x1 + bb)
    y2 = x1 * w1 + (x1 ** 2) * w2 + (x1 ** 3) * w3 + b
    y2 = np.reshape(y2, [-1, 1])
    plt.plot(x1, y1, 'r')
    plt.plot(x1, y2, 'g')
    plt.title("PB_load")
    plt.savefig('./save/PB.png')
    plt.show()
```

### ckpt_load

```py

save_path_ckpt = './save/dl-2-work.ckpt'

if  __name__ == "__main__":
    data = tf.placeholder(tf.float32, [None, 1])
    label = tf.placeholder(tf.float32, [None, 1])

    w1 = tf.Variable(tf.random_normal([1, 1], mean=0, stddev=200), dtype=tf.float32, name='s_w1')
    w2 = tf.Variable(tf.random_normal([1, 1], mean=100, stddev=200), dtype=tf.float32, name='s_w2')
    w3 = tf.Variable(tf.random_normal([1, 1], mean=200, stddev=200), dtype=tf.float32, name='s_w3')
    b = tf.Variable(tf.random_normal([1, 1], mean=200, stddev=100), dtype=tf.float32, name='s_b')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,save_path_ckpt)
        print(sess.run(w1))
        print(sess.run(w2))
        print(sess.run(w3))
        print(sess.run(b))
        w1 = w1.eval()
        w2 = w2.eval()
        w3 = w3.eval()
        b = b.eval()

    school_number = 18023032
    aa = 18.0
    bb = 32.0
    N = 2000
    x1 = np.linspace(-bb/aa,(2*math.pi-bb)/aa,N)
    y1 = np.cos(aa * x1 + bb)
    y2 = x1*w1+(x1**2)*w2+(x1**3)*w3+b
    y2 = np.reshape(y2,[-1,1])
    plt.plot(x1,y1,'r')
    plt.plot(x1,y2,'g')
    plt.title("ckpt_load")
    plt.savefig('./save/ckpt.png')
    plt.show()
```

