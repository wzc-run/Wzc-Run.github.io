---
layout:     post
title:      Improving person re-identiﬁcation by attribute and identity learning
subtitle:   Reading notes
date:       2019-08-04
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Re-ID
---
[论文连接](https://www.sciencedirect.com/science/article/pii/S0031320319302377?via%3Dihub#absh001)

# Abstract
## 研究现状 
Most existing re-ID methods only take identity labels  of  pedestrians  into  consideration. 
> 只考虑的行人的 **身份标签**。

## 属性

However,  we  ﬁnd  the  attributes,  **containing  detailed  local descriptions**,  are  beneﬁcial  in  allowing  the  re-ID  model  to  learn  **more  discriminative  feature  representations**. 

> 属性包含了详细的局部描述。
> 属性有助于re-ID 模型去学习更有辨别的特征表达
>

## 在文本中
in this paper, based on the **complementarity of attribute labels and ID labels**, we propose an attribute-person recognition (APR) network, a multi-task network which learns a re-ID embedding and  at  the  same  time  predicts  pedestrian  attributes.

> 基于 **属性** 标签和 身边（ID）标签的**互补性**，提出了——
>### attribute-person recognition (APR) network (属性-人物识别网络)
> - 多任务网络
> - 学习re-ID嵌入
> - 同时预测行人的**属性**

## 研究者做了什么
We  manually  annotate  attribute  labels  for  two large-scale  re-ID  datasets,  and  systematically  investigate  how  person  re-ID  and  attribute  recognition beneﬁt from each other. In addition, we **re-weight the attribute predictions** considering the dependencies and correlations among the attributes. 

> - 给两个大规模re-ID数据集 手动标注了**属性标签**
> - 系统地调查了人物 的re-ID 和属性识别如何互利
> 考虑到**属性之间**的依赖性和相关项，**re-weight（重新加权）**了**属性预测**

## 实验结果
**The experimental results** on two large-scale re-ID benchmarks demonstrate  that  by  learning  a  more  discriminative  representation,  APR  achieves  competitive  re-ID performance compared with the state-of-the-art methods.

 We use APR to speed up the retrieval process by ten times with a minor accuracy drop of 2.92% on Market-1501. Besides, we also apply APR on the attribute recognition task and demonstrate improvement over the baselines. 

> 通过学习更有辨别力的表达，APR 拥有了和state-of-the-art方法相比较的竞争力。
> 我们使用APR将检索过程加速十倍，精度下降幅度为2.92％
> 在**属性识别**任务上应用APR，并展示出了改进。

# Introduction
## 属性
Attributes describe detail information for a person, including gender, accessory, the color of clothes, etc .
> 属性： 性别、配饰、衣服的颜色 etc.

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190805192933881.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
## 在本文中
In  this  paper,  we  aim  to  **improve  the  performance  of large-scale person re-ID**, using **complementary cues** （互补线索）from attribute labels.  
> 使用属性标签的互补线索，改善了re-ID的性能

## 本文的动机
The  motivation  of  this  paper  is  that  existing  large-scale pedestrian datasets for re-ID contains only annotations of identity labels,  we  believe  that  attribute  labels  are  complementary  with identity labels in person re-ID.
>现有数据集仅标注了身份信息，我们却坚信 属性标签在re-ID任务上 和 身份标签是互补的。


## 属性标签的有效性是三倍的
The effectiveness of attribute labels is three-fold: 
**First, training with attribute labels improves the discriminative ability of a re-ID model.** 
Attribute labels can depict pedestrian images with **more detailed  descriptions.**  
These  local  descriptions  push  pedestrians **with similar appearances closer to each other** and those different away from each other 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190805201614691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**Second, detailed attribute labels explicitly guide the model to learn the person representation by designated human characteristics.** 
 With the attribute labels, the model  is  able  to  learn  to  classify  the  pedestrians  by  **explicitly focusing on some local semantic descriptions**, which greatly ease the training of models.
 
**Third, attributes can be used to accelerate the retrieval process of re-ID**
The main idea is to ﬁlter out some gallery images that do not have the same attributes as the query. 

> - 首先，使用属性标签进行训练可以提高re-ID模型的判别能力
>- 其次，详细的属性标签明确指导模型通过指定的人类特征来学习人物表示。
>- 主要思想是过滤掉一些与查询不具有相同属性的图库图像。

## 相关研究
In [6] , the  PETA  dataset  is  proposed  which  contains  both  attribute  and identity attributes. However, PETA is comprised of small datasets and most of the datasets only contain one or two images for an identity. 
> [6 ]提出了包含属性和身份属性的PETA数据集。但是，PETA由小数据集组成。大多数数据集仅包含一个或两个身份图像

When using attributes for re-ID, attributes can be  used  as  auxiliary  information  for  low  level  features  [9]  or used to better match images from two cameras [10–12] . 
> [9] 属性被用作辅助信息
> [10-12] 属性被用于更好地匹配来自两个摄像机的图像


 In recent years,  some  deep  learning  methods  are  proposed  [13–15] .  In these  works,  the  network  is  usually  trained  by  several  stages. Franco et al. 
 > 最近的深度学习方法中 网络分为几个阶段来训练
 
[13] propose a coarse-to-ﬁne learning framework. The network is comprised of a set of hybrid deep networks, and one of the networks is trained to classify the gender of a person. In this work, the networks are trained separately and thus may over- look the complementarity of the general ID information and the attribute details. Besides, since gender is the only attribute used in  the  work,  the  correlation  between  attributes  is  not  leveraged in [13] .
>[13]由混合深度网络组成. 
>- 其中一个网络 用于对人的性别进行分类。
>- 单独训练，可能会忽略一般ID信息和属性细节的互补性
>- 性别是使用的唯一属性，没有利用属性之间的相关性。


In [14,15] , the network is ﬁrst trained on an independent attribute dataset, and then the learned information is transferred to the re-ID task. 
> [14,15]首先在独立的属性数据集上训练网络，然后将学习的信息传送到re-ID任务。

A work closest to ours consists of [16] , in which the CNN embedding is only optimized by the attribute loss.  We will show that by combining the identiﬁcation and attribute recognition with an attribute re-weighting module, the APR network is superior to the method proposed in [16] . 
>[16] CNN嵌入仅通过属性损失进行优化
> **VS** 我们将通过将 **带有属性重置权重模块的属性识别** 和 **身份识别** 相结合来证明 ，APR网络优于[16]中提出的方法

## 和以前工作相比，我们工作主要的不同
**First,** our work systematically investigates how person re-ID and attribute recognition beneﬁt each other by a jointly learned network.
> 通过联合学习网络，我们系统地调查了 re-ID 和属性识别是如何互利的

 **On the one hand**, identity labels provide global descriptions for person images, which have been proved effective for  learning  a  good  person  representation  in  many  re-ID  works [17-19]
  **On the other hand**, attribute labels provide detailed local descriptions.
  > 身份标签 提供全局描述
  > 属性标签提供详细局部描述
—— 由此实现更高准确率的 属性识别和 re-ID 识别。


 **Second**, in previous works, the correlations of attributes are hardly considered. 
 > 之前的工作没有考虑属性之间的相关性。
 
In fact, many attributes usually  cooccur  for  a  person,  and  the  correlations  of  attributes may be helpful to re-weight the prediction of each attribute. 
We thereby introduce an Attribute Re-weighting Module to  utilize  correlations  among  attributes  and  optimize  attribute predictions. 

>事实上，许多属性通常是一个人共同出现的，属性的相关性可能有助于重新加权每个属性的预测。
>因此，我们引入属性重新加权模块以利用属性之间的相关性并优化属性预测。

## 在本文中（描述完整）
In  this  paper,  we  propose  the  **attribute-person  recognition (APR) network** to exploit both identity labels and attribute annotations  for  person  re-ID.  

By  combining  the  attribute  recognition task  and  identity  classiﬁcation  task,  the  APR  network  is  capable  of  learning  more  discriminative  feature  representations  for pedestrians,  including  global  and  local  descriptions.  
>结合**属性识别任务 ** 和 **身份分类任务** ， APR网络可以学习 更有辨别力的特征表达， 包括全局和局部描述。

Speciﬁcally, we  take  attribute  predictions  as  additional  cues  for  the  identity classiﬁcation. Considering the dependencies among pedestrian attributes, we ﬁrst re-weight the attribute predictions and then build identiﬁcation upon these re-weighted attributes descriptions. 
> 我们将**属性预测** 作为 **身份分类**的附加线索。
>  考虑属性之间的依赖性，我们首先 re-weight 了 属性预测 并且 在这些re-weight了的属性描述上 构建了身份。

The attribute is also used to speed up the retrieval process by ﬁltering out  the  gallery  images  with  different  attribute  from  the  query image.
> 这些属性加速了 检索过程。


## 实验结果
**In the experiment**, we show that by applying **the attribute acceleration process**, the evaluation time is saved to a signiﬁcant extent.  
> 因为有属性加速，所有评估时间缩短了。

We  evaluate  the  performance  of  the  proposed  method APR on two large-scale re-ID datasets and an attribute recognition dataset. **The experimental results** show that our method achieves competitive re-ID accuracy to the state-of-the-art methods. 
>APR的性能 和state-of-the-art methods 有得一比
>
In addition, we demonstrate that the proposed APR yields improvement in the attribute recognition task over the baseline in all the testing datasets.
> APR 在属性识别任务上有所改进。

## Contributions
(1) We have manually labeled a set of pedestrian attributes for the  Market-1501  dataset  and  the  DukeMTMC-reID  dataset. Attribute annotations of both datasets are publicly available on our website ( https://vana77.github.io ). 
> 给数据集手动标注属性

(2) We  propose  a  novel  attribute-person  recognition  (APR) framework.  It  learns  a  discriminative  Convolutional  Neural Network (CNN) embedding for both person re-identiﬁcation and attributes recognition. 
> 提出了APR 

(3) We  introduce  the  Attribute  Re-weighting  Module  (ARM), which corrects predictions of attributes based on the learned dependency and correlation among attributes. 
> 引入了 属性 re-weighing 模块（ARM）, 它根据学习到的属性之间的相关性和依赖性来纠正属性的预测。
> 
(4) We  propose  an  attribute  acceleration  process  to  speed  up the retrieval process by ﬁltering out the gallery images with different  attribute  from  the  query  image.  The  experiment shows that the size of the gallery is reduced by ten times, with only a slight accuracy drop of 2.92%. 
> 提出 了属性加速过程。 

(5) We achieve competitive accuracy compared with the state- of-the-art  re-ID  methods  on  two  large-scale  datasets, i.e.,  Market-1501  [17]  and  DukeMTMC_reID  [20] .  We  also demonstrate  improvements  in  the  attribute  recognition task. 
> 效果和state- of-the-art  re-ID  methods 有得一比。 在属性识别任务上有改进。
