---
layout:     post
title:      Weakly Supervised Classification: Towards Accurate Machine Learning with Low Labeling Costs
subtitle:   Masashi Sugiyama:弱监督机器学习研究进展
date:       2019-10-24
author:     JoselynZhao
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - Weakly Supervised
    - Machine Learning
    - SSL 
---

> reference link : [HPCL-智能计算 · NUDT](http://202.197.9.252/blog/2019/10/21/masashi-sugiyama%E5%BC%B1%E7%9B%91%E7%9D%A3%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%A0%94%E7%A9%B6%E8%BF%9B%E5%B1%95/?nsukey=y3LQ7vhLZ1CNy5zWobCAk8eCMeeAdjIh7krqj%2FM6Ty2ecxukNYFzUR1eeNxgENefF%2BYk0hhtYYKIuw4vlIVYMnszbzeD%2B66tuxtQAuwOnTt%2BPna4iiLjp9fPeuF1PChTgKjljapea1YZSUeHeVln0l5Q9xlSt7Q5vUNaUavG073JI8Hbc9ivMpEvGas9K2%2Fr5BUBaJOneP%2FX%2BGPHQTimTQ%3D%3D)
> 
Title： Weakly Supervised Classification: Towards Accurate Machine Learning with Low Labeling Costs

报告人：Prof. Masashi Sugiyama, The University of Tokyo

# 报告摘要
仍然有大量应用领域数据标签的采集不够充分，使得**基于充分标签数据的学习行不通**。

本次报告将介绍**基于经验风险最小化弱监督机器学习的最新进展**， 包括将两种类别的无标签数据进行分类、将有标签与无标签数据进行分类、一个对于半监督分类问题的通用方法、以及对于有标签数据的分类。

# 监督学习、非监督学习和半监督学习概述
首先还是要去关注一个最简单的问题，就是**二元分类的问题。**
## 监督分类
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024111512684.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
Optimal convergence rate： $O(n^{-1/2})$

我们希望也能够对**无标注数据进行分类**，这就是**无监督分类**的由来。


## 无监督分类
其实**无监督分类和聚类是一样的**，比如下面这张图中的数据点聚成了两簇，每一个簇代表一个类别，这样才是非常好的分类结果。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024111739381.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
遗憾的是，无监督分类的结果无法得到验证。

由于我们有大量的无标注数据和少量的标注数据，那么基**于少量的标注数据就能在一定程度上找到边界**，然后结合所找到的边界和大量无标注数据的聚类结果，从而找出更多的边界。这就是**半监督分类**。


## 半监督学习
Use  a large number of unlabeled samples and a small number of labeled samples:
Find a decision boundary along cluster structure induced by unlabeled samples:
然而，半监督分类和无监督分类面临同样的问题，也就是簇要能够跟类别对应起来。
**如果一个簇总能对应着一个类别，这样就完美**了。但事实并非如此，这就是我们今天所要讨论的内容。


对于监督分类，能够取得很高的分类准确率但同时标注的成本非常高；而对于半监督和非监督分类，标注的成本都比较低（甚至没有），但取得的分类准确率并不高。

**如何让左下角的这两种方法（即半监督和非监督分类）能够取得较高的分类准确率，同时保持比较低的标注成本？**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024140652379.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

achieving high classification accuracy with low labeling costs is always a big challenge!

模型方面，从简单到复杂，我们有**线性模型、增量模型、基于核函数的模型**和深度学习模型等；机器学习方法方面，有**监督学习、无监督学习、半监督学习和增强学习**等。

任意的学习方法和模型都是可以相结合的.

**学习方法：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024141311122.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
接下来会给大家介绍四种不同的分类方法

# 弱监督学习的研究进展
> P: Positive
> N: Negative
> U: Unlabeled
> Conf: Confidence
> S: Similar
> Comp: Complementary

## PU数据分类
个话题要谈的是如何处理PU (Positive, Unlabeled)的数据，也就是只有**正例数据和未标注的数据**。

我们有两类数据样本，一种是属于正类的，另外一类是未标注的。**当然未标注的数据里包含了正类和负类两种数据**，但是我们并不知道其中哪些是正类，哪些是负类。

对应**这种数据类型的一个例子**是，比如有一些你点击和未点击的网站，对于那些未点击的网站中，你既有未来可能会点击的，也会有你不会点击的（或者有想点击的，但可能由于忙而没有真正点击的），对这些网站点击数据，我们可以应用PU的分类方法。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024142149327.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
正例是蓝色的圆圈，未标注数据是黑色的方框。
我们接下来看一下**分类器的风险函数**。 

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024142451904.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
我们用到了损失函数，用了l表示； y是用f(x)表示；我们用R(f)表示风险函数，代表数据分类的风险，可以看到风险函数包括正类数据的分类风险以及负类数据的分类风险。

我们可以通过PU的学习，从PU的数据中得出**PN**的信息。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024142655227.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

在左边PU的边界可以比PN的边界要小，**我们一开始是用PU的结果而没有PN的，这是我们的起点**。但如果满足了这个条件，PU数据的学习要比PN数据的学习更好，但前提是我们要有大量PU数据；因为如果说有大量的PU的数据的话，我们左侧边界值就会变的小一些。**所以说，PU的学习有时候可以比PN更好一些**，这让我们研究出下一种方法，我后面会给大家看另外一种方法。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024142857878.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**我们再来看一下之前的PN的风险函数公式，就是基于P数据和N数据的风险对U数据的分类风险进行估计。**

根据这个定义，N数据的风险是非负类的，但是它是PU的样本，在现实当中我们要对样本进行估计.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024142958684.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

我们看到，对于非负类的PU分类，先从虚线的蓝线开始看起，是PN的测试数据上的误差结果（蓝色的实线），这表示模型是收敛的。

再看一下红色的虚线，是PN的训练数据上的误差结果，在到某个点的时候会变成负，这表示模型的训练已经出现了过拟合。因为当在训练数据的误差值变成负了之后，PU测试数据上的误差值开始增长了。一个简单的解决过拟合的方法是，限制这些误差值为非负。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024143127535.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019102414320663.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

所以，我们在CIFAR10当中创建了很多的正类的数据，蓝色线代表PN测试。在这里可以看到，如果np等于1000错误率下降非常快；如果说是对于非负的测试数据，比如说就是这条黄色线和蓝色虚线的话，它的错误率下降就并不是那么的明显；如果说我们使用一些ReLU方法，PU做的比PN要好的多。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019102414324756.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
接下来做一个简单的总结：PU数据分类是怎么做的？我们做的非常简单，就是把P和U数据，就是黑色跟蓝色符号数据进行分开（黑色符号数据中其实还含有蓝方符号的数据），最简单的方式就是做偏置。如果使用线性模式能够实现这样的一个二次方差的方法，那么能够保证在P跟U当中的损耗是一样的，所以在实验当中我们也证明了这样的方法是很有作用的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024143410752.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## PNU数据分类
接下来我们介绍一下PNU (Positive, Negative, Unlabeled)分类，就是正类、负类和无标签数据的分类。**PNU分类其实就是一个半监督的学习方法**。

现在我们对于PU分类学习有了解决方案，对NU分类学习也有自己的解决方案，**所以PU跟NU基本上一样的**。

对于PU、PN和NU分类学习中能不能使用半监督的方法，我们是希望能把其中的两者结合起来，就是蓝点或黑框或者红叉和黑框结合起来。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024143623262.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
将 PNU 分解为PU，PN，和NU：每一个都是有解决方案的，现在我们只需要将他们线性结合起来。

在没有聚类的假设下， PN分类是可训练的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019102414391856.png)
最简单就是PU和NU要把它结合起来，我们要把这两者整合起来。所以原理就是，第一步把PN和PU结合起来，第二步把PN和NU结合起来，这样的话我们总是能获得最优的方法，这是我们现在做的一个研究工作。

所以，**我们的方法就是把它们结合起来，进行一个组合，根据我们自己假设性的数据进行切换**，如果 是零，那就是变成一个PN分类学习，如果是 是负，那就加上PU学习，如果 变成正数我们就加上NU学习。后续继续选择，**基本上在三者之间自由组合，添加一个不同PN、PU和NU的组合来实现自由分类。**



## Pconf数据分类
Only P data is available, even not U data:
- Data from rival companies cannot be obtained.
- Only positive results are reported(publication bias).

"Only-P learning" is unsupervised. (无监督的 ？？？)

From positive-confidence data, ERM is possible!  ???
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024145746500.png)
>看confidence 大小？


## UU数据分类

首先看一下UU (Unlabeled, Unlabeled) 分类，U代表的无标注的数据（Unlabeled data）。

假设我们有两个未标注的数据集，**它们唯一的不同在于类先验（class-priors，即所属的类别）的不同。**


它们的函数分布如图中的左右下角，数量上各占50%左右，其实我们并不需要知道具体的比例。基于这种假设，我们需要训练一个分类器

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024150051186.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024150127391.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
## SU数据分类
Delicate classification (income,religion...):
- Highly hesitant to directly answer questions.
- Less reluctant to just say "same as him/her".

From similar and unlabeled data. (相似和未标注数据)
PN classifiers are trainable by ERM!

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024150435165.png)
>在一堆未标注数据里面 ，有相似关系。

- Decoupling S-pairs results in UU classification!
- Learning from dissimilar pairs is also possible.


## 互补型标准类别
Complementary Labels
因为如果在1000个不同的标签当中来选择一个正确的标注类别描述这个类的话，其实这是很耗时的，**这时候需要互补的标注类别。**

我们选择其中的一类，就是错误的一类。这个做起来就简单了，1000类个候选当中，我们只需要把它随机抽选，如果说这个是错的话，那么我们就选，如果是正确我们就不选，我们选下一个，换**句话说我们只选择错误的，帮助我们更快的选择最后正确的那一类，这个算法对于我们来也非常具有借鉴意义。**

From complementary labels, classifiers are trainable by ERM!

换句话说，**其实就是使用类别的互补性**，更容易选择大样本正确的类。我们现在假设是这样的，正常的标签，都是来自于p(x, y)，但是是互补标签。所以，从这样的假设来看的话，我们没有办法确定它这样的一个一般性的标签和我们互补标签到底应该是以什么样的方法选择，但是如果说我们定好了这样的一个公式的话，我们就可以从互补标签的分类当中进行学习。

合并普通标签
将多标签转换为yes-on标签
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024151428745.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

假设我们做c类的分类，我们把R(f)和gy拿出来，gy就是单个class的分类风险，我们会对这个分类风险进行一个分析，今天我只举其中的一个风险的分析的公式。我们把两个分类进行对比，然后去算它的损失，我们会有这样的一个程度对称性的损失，就得出它的风险。


# 应用
**Learning from Weak Supervision**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024151636483.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**Model vs. Learning Methods**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024151719285.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**Noise-Robust Supervised Learning**

Noise robustness is highly important.
- Sensor error, human error, ...

Traditional approches：
- Unsupervised outlier detection: ineffective (无监督异常检测)
- Robust loss: not Strong enough
- Regularization: not strong enough
- Estimating noise transition: difficult

Novel approaches are needed,
in particular for deep learning!

**CO-teaching**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024161436243.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**Pumpout**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024161618181.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)


# 总结

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024161733415.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
