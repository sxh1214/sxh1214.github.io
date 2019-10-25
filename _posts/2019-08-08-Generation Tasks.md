---
layout:     post
title:      Generation Tasks
subtitle:   Academic Report
date:       2019-08-08
author:     JoselynZhao
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - GAN
    - Computer Vision
---

@[toc](Generation Tasks)
# Improved GAN
如何从语言描述的场景来生成图像
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808103312108.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## related tasks 
- image synthesis
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808103359130.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808103455640.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
可以通过 RNN+GAN 生成实现
## Improved GAN-Image Generation
### GAN
Improved GAN-Image Generation
- *Image Generation from Scene Graphs (CVPR2018)*
- *Stackgan: Text to photo-realistic image synthesis with 
   stacked generative adversarial networks. (ICCV 2017)*
   使用堆积的生成对抗网络实现从文本到真实图像的合成
   
### Problems of GAN
carefully balance the capabilities between the generator and the discriminator
仔细平衡发生器和鉴别器之间的能力

unsuitable parameter settings can degrade the performance
不合适的参数设置会降低性能

### Improved GAN
*Evolutionary Generative Adversarial Networks (TEVC 2019)*

### New paper from DeepMind
*Vector Quantized Variational AutoEncoder model（VQ-VAE2 wins bigGAN）*
矢量量化变分自动编码器模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808104413568.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

# Semi-parametric Strategy
半参数策略
## Image Generation
**Parametric models**
end-to-end training of highly expressive models. （高度表达模型）

**Nonparametric models** （非参数模型）
extract materials from large real image datasets during testing 
在测试的时候提取信息

**Semi-parametric Strategy**
*Semi-parametric Image Synthesis (CVPR2018)*

**The problem of existing image generation methods** （现有方法的问题）
give stunning results on limited domains such as 
descriptions of birds, flowers or people.
Low-quality image of other domains.
作用领域有限

~~Using semi-parametric strategy to improve  Image generation method~~ 

# Message-Passing Strategy
## Message Passing Strategy for Feature Sharing.
用于特征共享的消息传递策略
*Scene Graph Generation from Objects, Phrases and Region Captions(ICCV2017)*

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808105028318.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
# Domain Adaptation Strategy
区域适应策略
give stunning results on limited domains such as descriptions of birds, flowers or people.

Using Domain Adaptation to improve Image generation method  in more domains.

## Video generation

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808105240373.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
输入 深度图像
输出 对手关节、姿势的估计

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019080810535317.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
*3D Hand Shape and Pose Estimation from a Single RGB Image（CVPR19 Oral）* 

## Text Effects Transfer 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808105554187.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019080810561382.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
创新的分布感知数据增强策略
*TET-GAN: Text Effects Transfer via Stylization and Destylization
（AAAI2019）*

## Scene graph Generation
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019080810583568.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
modeling these statistical correlations between object pairs and relationships can effectively regularize the semantic prediction space （structured knowledge graph ）
对这些对象和关系之间的统计相关性建模可以有效地规范语义预测空间（结构化知识图）

*Knowledge-Embedded Routing Network for   Scene Graph Generation （CVPR2019）*
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808110011786.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## Robot Control
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808110044234.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
*Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection(2017)* 

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808110105395.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
*Towards Accurate Task Accomplishment with Low-Cost Robotic Arms（CVPR2019）*

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808110129957.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## Cross-Modal Zero-Shot Hashing retrieval
跨模态 zero-shot 哈希检索
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190808110252563.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

*Attribute-Guided Network for Cross-Modal Zero-Shot Hashing(T-NNLS19)*
 
# Reference
1. Image Generation from Scene Graphs (CVPR2018)
2. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. (ICCV 2017)
3. Evolutionary Generative Adversarial Networks  (TEVC 2019)
4. Semi-parametric Image Synthesis (CVPR2018)
5. Scene Graph Generation from Objects, Phrases and Region Captions(ICCV2017)
6. 3D Hand Shape and Pose Estimation from a Single RGB Image（CVPR19 Oral）
7. TET-GAN: Text Effects Transfer via Stylization and Destylization
（AAAI2019） 
8. Knowledge-Embedded Routing Network for   Scene Graph Generation （CVPR2019）
9. Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection(2017) 
10. Towards Accurate Task Accomplishment with Low-Cost Robotic Arms（CVPR2019） 
11. Attribute-Guided Network for Cross-Modal Zero-Shot Hashing(T-NNLS19)

# Acknowledgment
感谢本篇博文的内容提供者：崔亚文（Cui Yawen）· HPCL
