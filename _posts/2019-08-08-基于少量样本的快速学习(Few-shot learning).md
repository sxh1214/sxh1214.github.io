---
layout:     post
title:      基于少量样本的快速学习报告（Few-shot learning）
subtitle:   Academic Report
date:       2019-08-08
author:     JoselynZhao
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - few-shot
    - SSL 
---
# 背景
## 人工智能
**连接主义**：核心是神经元网络和深度学习，仿造人的神经系统，以此仿造智能（John Hopfield）
**逻辑主义（符号主义）**：核心是符号推理与机器推理，用符号的方式来研究智能、推理（Marvin Minsky）

## 神经网络的三次浪潮
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808084428212.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## 深度学习
模型规模与日俱增
- 错误率越来越低
- 应用范围越来越广：图像识别、语音识别，机器翻译…

依赖大量的**带标签训练数据——Data Hungry**

## 人工智能困境
特定任务需要专门的模型从零开始训练
- 训练时间
- 计算资源

模型训练需要大量的带标签数据
- 打标签需要耗费大量的人工成本
- 所需训练数据本身难以获得


## 人工智能 → 人类智能
基于少数样本快速识别目标的能力——学习能力
- 对猫、狗等生物体的识别
- 根据一张照片快速识别目标的能力

结合已有知识快速学习新知识的能力——泛化能力
- 学会C++后继续学习Java
- 学习词汇后对英语等外语的学习过程

# 定义及数值原理
## 机器学习定义
A computer program is said to learn form experience E with respect to some classes of task T and performance measure P if its performance can improve with E on T measured by P
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808084753761.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
优化目标
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808084833726.png)

## 数值原理
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808090250351.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808090333335.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808090437971.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

# 数据增强
## 数据预处理
翻转，剪切，缩放，反射，裁剪，旋转…

基于已有的丰富标签数据训练**转化器**，以此对小样本数据集进行处理，所得数据与原始数据构成新的数据集

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808090650853.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808090658235.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## 综合运用其他已有数据
无标签数据集——半监督学习
- 通过在标签数据中引入无标签数据，增强监督分类效果
- 通过在无标签数据中引入标签数据，增强无监督聚类效果

相似数据集
- 关键：相似性的衡量
- 2018年，Gao1等人通过GAN实现了对数据集类与类之间的衡量，从而实现相似数据集的确定

# 网络模型
## 和相关任务共同训练——多任务学习模型
**定义**：基于共享表示，把多个相关的任务放在一起学习的机器学习方法
**参数硬（Hard）共享**
- 所有任务共享隐藏层
- 不同任务使用不同输出层

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808091309476.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**这种模型能够极大降低模型过拟合的风险**


**参数软（Soft）共享**
- 所有任务拥有单独模型
- 使用正则项保证不同模型参数尽可能相似

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808091428719.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## 生成模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019080809151324.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## 嵌入学习模型（Embedding Learning）
通过映射，将数据从高维空间映射到低维空间
**分类**
- 任务专用（Task Specific）
- 多任务共享（Task Invariant）
- 两者结合——元学习
以任务训练数据，对多任务共享得到的embedding函数进行调整
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019080809202597.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**元学习：** 学会学习（Learning to Learn）
- 组件：基础学习器，元学习器，外部存储器- 
- 学习发生在两个层面
- 基础学习器在输入任务空间执行，元学习器在任务不可知的元空间中运行（负责学习学习能力）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808092226554.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
## 带外部存储的学习模型
利用外部存储对**已经训练后的信息进行存储**
存储器记忆更新策略：
- 最近最少使用
- 记忆年龄
- 记忆Loss值
…
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808092329734.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## 塑性网络模型
核心思想——突触可塑性
- 赫布规则
- 如果一个神经元反复参与另一个神经元的活动，它们之间的联系就会加强

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808092601452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808092656493.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808093201628.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808093333611.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

模式分类
- 数据集：Omniglot
	50种文字
	1623类手写字符
	每类字符仅有20个样本
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808093501288.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

- 训练过程
    1、随机选择5类字符，每类字符1个样本
    2、从上述5类字符中随机选择一个样本进行预测分类
    3、利用预测分类与真实分类误差进行反向传播

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019080809363889.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808093654771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

- 对miniImageNet可见光图像分类
正确率达到75%
非塑性神经网络方法的最好性能为58%

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019080809372039.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## 贝叶斯程序学习（BPL）模型
- 每一个“概念”均由多个简单的“基元”组成，基元之间存在位置、时间、因果上的关系，这些基元按照上述关系进行组合，就得到相应“概念”的实例
-  贝叶斯模型可以将各种“关系”参数化，通过计算机自动学习这些参数
- 文章将BPL用于手写字符的单样本概念学习，并实现了模拟手写字符、归类手写字符、创造并书写新字符
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808093925109.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808094142463.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

BPL参数解析过程
- 对图片进行处理，得到符号的细“骨架”
- 使用**随机游走**方法解析所得符号的书写方式及BPL参数

然而，解析需要的先验概率分布信息由预训练得到，论文中并没有提到预训练（由元学习训练）的过程，并且在所给出的代码也是直接给出了已经训练好的预训练模型

每一套过程相当于一个字符的program。虽然看起来比较复杂，但是其实是个马尔可夫过程，即并不考虑n>1步之前的笔画或子笔画。所以只需要训练一个转移矩阵就好。

训练的时候，先给例子字符，以及这个字符怎么拆成笔画和子笔画。这些部件，以及笔画间的转移概率会被学到。然后生成的时候，就按照上述过程随机生成大概率的笔画组合，看起来就像同一风格的文字。

先随机生成一个字的笔画数，然后生成每个笔画的子笔画数。然后随机从一个库里选每个子笔画的类型，比如竖线、左撇等。
# 优化算法
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019080809481295.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
# 研究展望
证明是→ 证明否 —— 反正（证伪）学习

借鉴逻辑主义学派思想（如：贝叶斯模型）提高样本的利用率
# 参考文献
1. H. Gao, Z. Shou, A. Zareian, H. Zhang, and S. Chang. 2018. Low-shot Learning via Covariance-Preserving Adversarial Augmentation Networks. In Advances in Neural Information Processing Systems. 983–993.
2. Miconi T, Clune J, Stanley K O. Differentiable plasticity: training plastic neural networks with backpropagation[J]. arXiv preprint arXiv:1804.02464, 2018.
3.  Lake B M , Salakhutdinov R , Tenenbaum J B . Human-level concept learning through probabilistic program induction[J]. Science, 350.

# 特别鸣谢
感谢本篇博文的内容提供者：奉涌泉 · HPCL
