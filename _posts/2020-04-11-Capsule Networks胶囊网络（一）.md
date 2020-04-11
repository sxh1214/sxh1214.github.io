---
layout:     post
title:      Capsule Networks胶囊网络（一）
subtitle:
date:       2020-04-11
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Capsule Networks
    - Computer Vision
    - Deep Learning
---

>author: Sargur Srihari srihari@buffalo.edu

This is part of lecture slides on Deep Learning: [http://www.cedar.buffalo.edu/~srihari/CSE676](http://www.cedar.buffalo.edu/~srihari/CSE676)

@[toc]
# Limitations of Convolutional Networks

## ConvolutionalNeuralNetworks
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020041020582280.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
>Source: https://hackernoon.com/ what-is-a-capsnet-or-capsule- network-2bfbe48769cc
- 与常规神经网络相比，将计算量最小化
- 卷积极大地简化了计算，而不会丢失数据的本质
- 擅长处理图像分类
- 在所有图像位置使用相同的知识

## Processing Steps and Training for ConvNets
1. Givenaninputimage,asetofkernelsorfiltersscan it and perform the convolution operation.
2. This creates a feature map inside the network.
	These features next pass via activation and pooling layers 
	• Activation layers, e.g., ReLU, induce nonlinearity
	• Pooling (eg: max pooling) helps in reducing the training time.
	（pooling实现子区域的摘要，实现不变性）
3. At the end, it will pass via a classifier sigmoid/softmax
	Training is based on back propagation（反向传播） of error matched against labeled data.
	（非线性也有助于解决消失的梯度问题）

## Pooling and Invariance 
（池化和不变性）
Pooling应该获得位置，方向，比例或旋转不变性。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200410210725401.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
Every input value changed, but only half the output values have changed because maxpool is only sensitive to max value in neighborhood not exact value.

## Example of CNN Limitation
### CNN to recognize faces extracts features from image
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200410221633872.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
与顺序无关，位置不对CNN也能进行识别

## Motivation for CapsNets
Caps nets are an improvement on CNNs
-  They are the next version of CNNs
- Solve problems due to max pooling and deep nets 
 	Loss of information regarding order and feature orientation
- Hinton: “The pooling operation used in CNNs is a big mistake and the fact that it works so well is a disaster”

## Solution offered by CapsNets
-  Low level features should also be arranged in a certain order for the object to be classified as a face
（排序低级特征）
- Order is determined during training when the network learns not only what features to look for but also what their relationships to one another should be （顺利由训练决定，不仅学习特征，还要学习特征之间的关系）
 具有特征顺序特征的图像才会被识别为人脸。


## Visual Fixation
（视觉固定）

### Human vision uses saccades
（人类视觉使用扫视）
- 通过仔细的固定顺序忽略无关的细节
- 确保仅以最高的分辨率处理光学阵列的一小部分
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200410222721201.png)

We assume a single fixation will give us
• Much more than a single identified object and its properties
• Assume our multi layer visual system creates a parse tree on each fixation
• We ignore coordination of parse trees（解析树） over multiple fixations
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200410222932857.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
###  Parse Tree of a Fixation
- 对于单个注视，
	从固定的多层神经网络中刻出一个分析树
  	像岩石上的雕塑
- 每层将被分成许多小的神经元组，称为“胶囊”
- 解析树中的每个节点将对应一个活动胶囊
### Activation is a likelihood
神经元的激活水平可以解释为检测到特定特征的可能性
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200410225719559.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
胶囊是一组神经元，不仅捕获可能性，而且捕获特定特征的参数。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200410225802213.png)

## CNN versus CapsNets
max pooling layers 获取图片的重要特征，但丢失了特征的结构取向
CNN 只检测特征是否存在，而不考虑位置

Capsnets replace scalar-output feature detectors with vector-output capsules and max-pooling with routing- by-agreement.
Capsnet用向量输出封装代替标量输出特征检测器，用按协议路代替最大池化。


# Definition of Capsule
胶囊是一组神经元，其输出表示同一实体的不同属性。
胶囊是一组神经元，其激活向量代表特定类型的实体（例如对象或对象部分）的实例化参数

### Example
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200411164008597.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200411164032724.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
2个神经元表示输出为二维向量，前项表方向，后向表大小。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200411164329177.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## Representing Entity in Input
within an active capsule
神经元表示实体的各种属性
- 姿势，变形，颜色，速度
- 实体在图像中是否存在是一个特殊的属性。

representing existence of entity
一个独立的logistic unit
- 输出表示实体存在的概率
- 向量的长度表示实体的存在，方向表示实体的属性

应用非线性使方向保持不变但减小其大小，保证胶囊的输出的向量长度不超过1
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200411170122418.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
（数值大小，方向，实体大小，粗细程度）

## Capsule Networks perform inverse Computer Graphics
Computer graphics generates images based on descriptions
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200411170353441.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## Pooling and Equivariance
- CNN中的Pooling是一个非常粗糙的方法
实际上，它消除了各种位置不变性
导致将图1中的右侧图像检测为正确的船
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020041117073290.png)
Equivariance 使网络了解旋转或比例的变化，并据此进行自适应，以使图像内部的空间定位不会丢失。
### Example of equivariance with Capsnet
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200411171047762.png)
CNN会正确提取下层的特征，但会错误地激活上层的神经元以进行面部识别
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200411171056727.png)
CapsNet：神经元包含可能性以及特征的属性，例如包含[可能性，方向，大小]的向量

一旦模型认识到对该父神经元贡献最大信号的神经元在大小和方向上不一致，则用于面部检测的神经元的激活率将低得多。

## From Convolution to Capsule
CNN基于简单的机制
- 视觉系统需要在所有图像位置使用相同的知识
- 这可以通过绑定特征检测器的权重来实现，以便在一个位置学习的特征在其他位置可用。
- CNN使用学习过的特征检测器的翻译后的副本，这使它们可以将关于在图像中一个位置处获取的良好权重值的知识转换为其他位置。
- 卷积胶囊扩展了跨位置的知识共享，以包括关于表征熟悉形状的部分-整体关系的知识。

# Follow-up content
- Dynamic Routing
- Computing input/output vectors of capsule
- Routing by Agreement
- CapsNet Architecture
- Performance of Capsule Networks
- Matrix capsules with EM routing



