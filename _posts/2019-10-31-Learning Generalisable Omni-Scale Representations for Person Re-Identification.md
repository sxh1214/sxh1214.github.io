---
layout:     post
title:      Learning Generalisable Omni-Scale Representations for Person Re-Identification
subtitle:
date:       2019-10-31
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Re-ID
    - Computer Vision
---


@[toc](Learning Generalisable Omni-Scale Representations for Person Re-Identification)
> Reference：[https://mp.weixin.qq.com/s/C3ATMEL173ojCBaWcud_CQ](https://mp.weixin.qq.com/s/C3ATMEL173ojCBaWcud_CQ)
作者 | Kaiyang Zhou, Xiatian Zhu, Yongxin Yang, Andrea Cavallaro, and Tao Xiang
译者 | TroyChang
编辑 | Jane
出品 | AI科技大本营（ID：rgznai100）

这是一篇关于行人再识别领域的**新研究**（Person Re-Identification , re-ID）。这篇论文中，作者们提出了一个新的CNN架构——**OSNet**，在实验效果方面，这一新模型架构击败了**最新的无监督域自适应方法**。

>论文地址： 
https://arxiv.org/pdf/1910.06827.pdf 
Github：
https://github.com/KaiyangZhou/deep-person-reid



# Abstract
有效的行人再识别模型**应该学习特征表示**，这些特征表示既可以用于区别外观相似的人员，*又可以在无需任务调整下用于跨数据集部署*。

在本文中，我们提出了新的CNN架构来应对这两个挑战。首先，我们提出了一个被称为**全尺度网络（OSNet）的CNN来学习特征，它不仅可以捕捉不同的空间尺度，而且可以封装多个尺度的协同组合，即全尺度特征**。基本构建块由多个**卷积流**组成，每个卷积流检测一定范围内的特征。对于全尺度特征学习，提出了一种统一的聚合门，将多尺度特征与信道权值动态融合。**OSNet是轻量级的，因为它的构建块包含分解卷积。**

其次，为了改进通用特征学习，我们在OSNet中引入**实例规范化层**来处理跨数据集的差异。为了确定这些层在体系结构中的最佳位置，我们提出了一种有效的**可微体系结构搜索算法**。

大量的实验表明，在传统的相同数据集设置下，尽管OSNet比现有的re-ID模型要小得多，但它仍能实现最先进的性能。在更具挑战性和实用性的跨数据集设置中，OSNet击败了最新的无监督域自适应方法，同时并不需要任何目标数据来进行模型自适应。

# Introduction
行人再识别(re-ID)是一个**细粒度的实例识别问题**，其目的是在没有重叠视野的摄像机视图中**匹配行人**。随着深度学习技术的发展，近年来对人再识别的研究已经从繁琐的**特征工程转向了利用深度神经网络进行端到端特征表示学习，尤其是卷积神经网络**。
 
尽管在CNN的端到端表示学习帮助下，re-ID的性能得到了显著提升，但是还有两个问题没有解决。

**两个没有解决的问题：**
- 第一个问题是**判别特征学习**。
作为一个实例级识别任务，在不相交的摄像机视图下重新识别人需要克服类内变化大和类间模糊两大困难。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031102239528.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
>例如，在图1(a)中，相机之间的视角变化（从正面到背面）给背包区域带来了较大的外观变化，使对同一个人的匹配变得具有挑战性。此外，从远处看，就像在监控视频中常见的那样，人们看起来非常相似，如图1中的错误匹配就是一个例子。**这需要使用re-ID特性来捕获细微的细节(例如，图1(d)中的太阳眼镜)，用以区分具有相似外表的人**。

- 第二个问题是**通用特征学习**。
由于光**线条件、背景、视点**等方面的差异造成的re-ID数据集之间存在固有的区域差距(见图1)，**直接将在源数据集上训练的re-ID模型应用到不可见的目标数据集上，通常会导致的性能大幅下降** *(所以需要域适应)*。这表明所**学习的re-ID特性严重地过拟合标记数据，并且不能进行区域泛化**。

在本文中，我们设计了新的CNN架构来解决这两个问题。


# OSNet网络结构
首先，我们认为这些特征需要**全尺度**的，定义为变**量同构和异构尺度的组合**，每一个都由多个尺度的混合组成。从图1可以明显看出对全尺度特性的需求。为了匹配和区分人与冒名顶替者，**与局部小区域(如鞋子、眼镜)和整体身体区域相对应的特征是很重要的**。

例如，给定图1(a)(左)中的查询图像，查看全局范围的特性(例如，年轻人，白t恤+灰色短裤组合)将搜索范围缩小到真正的匹配(中)和冒名顶替者(右)。现在，局部尺度（local-scale）特征开始发挥作用——鞋子区域暴露了右边的人是骗子的事实(运动鞋vs.凉鞋)。

然而，对于更具挑战性的情况，即使是变量同构尺度的特征也不够。需要更复杂和更丰富的跨多个尺度的特性。例如，要消除图1(b)(右)中的冒名顶替者，**需要在前面具有特定标识的白色T恤上添加一些特征**。

请注意，这个标志本身并没有什么特别之处——如果没有白色T恤作为背景，它可能会与许多其他图案混淆。同样，白色T恤在夏天随处可见(如图1(a))。它是独特的组合，**由跨越小(标志尺寸)和中(上身尺寸)尺度的异构特性捕获，这使得这些特性最有效**。
 
因此，我们提出了一种**全新的CNN体系结构OSNet（Omni-scale Network，OSNet）,它是专门为学习全尺度特征表示设计的**。托换构建块（building block）由多个不同的卷积特征流组成(如图2所示)，每个流所关注的特征尺度由**指数（exponent）决定**，指数是一个新的维度因子，跨流线性增加，以确保每个块中捕获不同尺度。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031203907653.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

关键的是，由统一聚合门(AG)生成的通道权值动态融合得到的多尺度特征图。**AG是一种跨所有流共享参数的子网络，具有许多有效的模型训练所需的特性。** 在可训练的AG下，生成的**信道权值依赖于输入**，从而实现了动态尺度融合。
**这种新颖的AG设计为全尺度特征学习提供了极大的灵活性：** 根据特定的输入图像，门可以通过为特定的流/尺度分配主导权重来聚焦于单个尺度；或者，它可以选择和混合，从而产生异构的特征尺度。


**OSNet的另一个关键特性是轻量级**。
**轻量级的ReID模型有两个优点：**
- 由于收集跨摄像头匹配的人图像的困难，**ReID数据集通常是中等大小**。因此，具有少量参数的轻量级网络不容易出现过拟合；
- 在大规模的监控应用中(例如全市范围内使用数千个摄像头的监控)，re-ID最**实用的方式**是在摄像头端进行特征提取，将提取的特征发送到中央服务器，而不是原始视频。对于设备上的处理，小型的re-ID网络显然是首选1。为此，在我们的构建块中，我们将**标准卷积分解为点卷积和深度卷积，使OSNet不仅在特征学习上有区别，而且在实现和部署上也很高效**。
>1 这个解释有些牵强。 点卷积和深度卷积？

解决第二个问题，是由不同re-ID数据集造成的差距,我们注意到这些差距通常反映在不同的图像样式，如亮度、颜色温度和角度(参见图1)。这些风格差异是由不同的照明条件和相机/设置在不同的摄像机网络特征。现有的工作使用**无监督域适应(UDA)方法解决了这个问题**。这些需要未标记的目标域数据来进行模型调整。

相反，我们将其视为一个更一般的域泛化问题，**而不使用任何目标域数据**。通过消除给定新目标域的数据收集和模型更新的繁琐过程，使用我们的方法，可以对任何**未知的目标数据集开箱即用地应用使用源数据集训练的re-ID模型**。


OSNet是通过将提出的轻量级瓶颈（OS块）逐层堆叠来构建的。详细的网络架构如图3所示。与标准卷积相同的网络架构有690万个参数和33849万个多添加操作，比精简3×3卷积层设计的OSNet大3倍。图3中的标准OSNet在实践中可以很容易地伸缩，以平衡模型大小、计算成本和性能。为此，**我们在之后使用了一个宽度倍增器4和一个图像分辨率增器。**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031210842100.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**以下内容摘自论文原文**

# OMNI-SCALE NETWORK FOR PERSON RE-ID

We first discuss **depthwise separable convo-lutions**. Then, we introduce our **novel omni-scale residual block**. Finally, to enhance generalisation to unseen datasets,we extend OSNet by **instance normalisation (IN)** and further present a differentiable **architecture search mechanism** to automatically infer the optimal IN placement。

>[有道词典]我们首先讨论深度可分的组合。然后，我们介绍了我们的新型全尺度残差块。最后，为了增强对不可见数据集的通用性，我们通过实例规范化(IN)扩展了OSNet，并进一步提出了一个可区分的架构搜索机制来自动推断最优的布局

## Depthwise Separable Convolutions

**基本思想：**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031213048936.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031213150684.png)
## Omni-Scale Residual Block
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031213328170.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019103121334590.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**Multi-Scale Feature Learning **
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031213520227.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

> 细节太复杂了。暂时略过。



# 实验

简单说一下实验，对当前**七个广泛使用的re-ID数据集**进行实验，包括**Market1501 ， CUHK03 ， DukeMTMC-reID (Duke) ， MSMT17** ， VIPeR ， GRID和CUHK01。前四个通常被认为是**大型的ReID数据集**，尽管它们的大小相当适中(对于最大的数据集MSMT17，大约有30k的训练图像)。其余三个数据集通常太小，如果没有适当的训练前，就无法训练深度模型。

对于CUHK03，我们使用767/700 split来检测图像。对于VIPeR、GRID和CUHK01，我们遵循，**对大型的re-ID数据集进行模型预训练，然后对目标数据集进行微调**，平均结果为10个随机分割。对于re-ID评价指标，我们使用累积匹配特征(CMC)秩精度和平均精度(mAP)，其中结果以百分比报告。
 
本文在同区域行人再识别和跨区域行人在识别问题上分别与**当前SOTA**的方法进行了比较。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031214200974.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

从上图可以看出在VIPeR上，可以观察到OSNet的性能显著优于所有其他选择(超过11%)。GRID比VIPeR更具挑战性，因为除了额外的干扰物之外，它只有250张125个身份的训练图像。在CUHK01上，有大约1900张训练图像，OSNet的表现明显优于主轴和JLML，分别为6.7%和16.8%。总体而言，OSNet在这些小数据集上的性能是优越的，这表明它在没有大规模训练数据的实际应用中有很大的优势。


# 总结
在本文中，我们提出了一种轻量级的CNN架构OSNet，它能够学习人的全方位特征表示。与现有的ReID CNNs相比，**OSNet具有在每个构建块内显式学习多尺度特征的独特能力，其中统一聚合门动态融合多尺度特征生成全尺度特征。**

为了改进跨域的泛化，我们通过可微架构搜索为OSNet配备了实例规范化，从而产生了一种称为OSNet- ain的域自适应变体。在相同域的re-ID设置中，结果显示OSNet在比基于resnet的竞争对手小得多的同时，还能达到最先进的性能。

在跨域的ReID设置中，OSNet-AIN**在不可见的目标数据集上表现出了非凡的泛化能力**，甚至在没有对目标域数据进行每域模型自适应的情况下，也击败了最新的UDA方法。

