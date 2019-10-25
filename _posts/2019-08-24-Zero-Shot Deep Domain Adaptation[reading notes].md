---
layout:     post
title:      ZDDA : Zero-Shot Deep Domain Adaptation
subtitle:   
date:       2019-08-24
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - zero-shot
    - SSL
    - Domain Adaptation
---
# Abstract
Domain adaptation(域适应) is an important tool to transfer knowledge about a task.

**Current approaches**:
假设 task-relevant target-domain数据在训练期间是可用的。
而我们展示了如何在 上述数据不可用的情况下实现 Domain adaptation的

**为了解决这个问题，我们提出了zero-shot deep domain adaptation (ZDDA)**

使用来自 任务无关的双领域对 的 特权信息， 学习一个源领域的表示，不仅适合于兴趣任务，还接近于目标域的表达。

联合源领域表达训练的 源领域解决方法 可以使用 源表达和目标表达。


**数据集：**
Using the MNIST, FashionMNIST, NIST, EMNIST, and SUN RGB-D datasets

**效果**
可以在不是访问 任务相关目标域训练数据的情况下，实现分类任务的域适应。
我们还通过模拟与任务相关的源域数据的任务相关目标域表示，扩展ZDDA以在SUN RGB-D场景分类任务中执行传感器融合

ZDDA是第一个域适应和传感器融合方法，它不需要任务相关的目标域数据。
基本原则并不特定于计算机视觉数据，但应该可扩展到其他领域。


# Introduction
domain shift[17]  会导致将解决方法转移到另外领域时的性能下降。
DA任务的目标是为源域和目标域导出TOI的解决方案。


The state-of-the-art DA methods： [1, 14–16, 25, 30, 35, 37, 39–41, 43,44, 47, 50]   假设任务相关数据，可以直接应用和关系到TOI，在训练时在目标域可用。但这些假设在真实情况下往往不是这样的。

sensor fusion [31,48]

**ZDDA** learns from the task-irrelevant dual-domain training pairs without using the task-relevant target-domain training data, where we use the term task-irrelevant data to refer to the data which is not task-relevant. # 任务不相关
**在后文中，我们用T-R表示任务相关，用T_I表示任务无关。**


![在这里插入图片描述](https://img-blog.csdnimg.cn/20190824142822684.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**图一** ：当任务相关的目标域训练数据不可用时，ZDDA从与任务无关的双域对中学习。
  
> 这个图没有太明白
> 
> 
DA task MNIST [27]→MNIST-M [13]，source domian 灰度模式，target domain RGB模式
TOI： 在MNIST[27]和[13]做测试的数字分类。
假设不能会用MNIST[13]来训练数据

在例子中：
 ZDDA 使用 MNIST [27] training data and
the T-I gray-RGB pairs from the Fashion-MNIST [46] dataset and the Fashion-MNIST-M dataset  to **train digit classifiers** for MNIST [27] and MNIST-M [13] images。

ZDDA  achieves this by 使用灰度模式图像 模拟 the RGB 表达
and 创建联合网络 with the supervision of the TOI in the gray scale
domain.  We present the details of ZDDA in Sec. 3.
We make the following two **contributions**：
- ZDDA, 第一个基于深度学习的 域适应 方法，从一个图像形态到另一个不同的图像形态 (not just different datasets in the same modality such as the
Office dataset [32]) **without using the task-relevant target-domain training data.**  We show ZDDA’s efficacy using the MNIST [27], Fashion-MNIST [46], NIST [18], EMNIST [9], and SUN RGB-D [36] datasets with cross validation.）

- Given no task-relevant target-domain training data, we show that ZDDA
can 执行传感器融合 and that 和a naive fusion approach 相比， ZDDA is **more robust** to **noisy testing data** in either source or target or both domains in the scene classification task from the SUN RGB-D [36] dataset.


# Related work
域适应DA 被广泛地应用于计算机视觉和图像分类.[1,14–16,25,30,35,
37,39–41,43,44,47,50] 还有 **语义分割[45,51]** 和**图像字幕[8].**

在结合深度神经网络的情况下:
the state-of-the-art methods successfully perform DA with (fully or partially) labeled [8,15,25,30, 39] or unlabeled [1,14–16,35,37,39–41,43–45,47,50] T-R target-domain data. (最好的方法使用了 标注\部分标注\无标注 的T-R 目标域数据)

用于改善DA任务性能的策略:
- domain adversarial loss [40] 域对抗
- domain confusion loss[39]

大多数存在的方法都是需要T_R 目标域 训练数据. (但现实情况下,不可行)

ZDDA 在不实用T-R目标域训练数据的情况下 从T-I双域对中学习.

ZDDA包括使用源域数据模拟目标域表示，[19,21]中提到了类似的概念。但是[19,21]需要访问T-R双域训练对.


![在这里插入图片描述](https://img-blog.csdnimg.cn/20190824145513112.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

Table 1, which shows that the ZDDA problem setting is different from those of UDA, MVL, and DG.

MVL DG 给出了多个域中的T-R训练数据

但是，在ZDDA中，T-R目标域训练数据不可用，并且唯一可用的T-R训练数据在一个源域中。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190904155143327.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

only ZDDA can work under all four conditions.

**在传感器融合方面**:

[....]

# Our Proposed Method — ZDDA
ZDDA被设计来实现两个目标:
- Domain adaptation
- Sensor fusion

## Domain adaptation
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190904170809807.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190904170823386.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190904170834992.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190904171056192.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

REFERENCE
1. Aljundi, R., Tuytelaars, T.: Lightweight unsupervised domain adaptation by convolutional filter reconstruction. In: Hua, G., J´egou, H. (eds.) ECCV 2016. LNCS,
vol. 9915, pp. 508–515. Springer, Cham (2016). https://doi.org/10.1007/978-3-319-
49409-8 43
2. Arbelaez, P., Maire, M., Fowlkes, C., Malik, J.: Contour detection and hierarchical
image segmentation. IEEE Trans. Pattern Anal. Mach. Intell. 33, 898–916 (2011)
3. BAIR/BVLC: BAIR/BVLC AlexNet model. http://dl.caffe.berkeleyvision.org/
bvlc alexnet.caffemodel. Accessed 02 March 2017
4. BAIR/BVLC: BAIR/BVLC GoogleNet model. http://dl.caffe.berkeleyvision.org/
bvlc googlenet.caffemodel. Accessed 02 March 2017
5. BAIR/BVLC: Lenet architecture in the Caffe tutorial. https://github.com/BVLC/
caffe/blob/master/examples/mnist/lenet.prototxt
6. Blitzer, J., Foster, D.P., Kakade, S.M.: Zero-shot domain adaptation: a multi-view
approach. In: Technical Report TTI-TR-2009-1. Technological institute Toyota
(2009)
7. Bousmalis, K., Silberman, N., Dohan, D., Erhan, D., Krishnan, D.: Unsupervised
pixel-level domain adaptation with generative adversarial networks. In: The IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3722–3731.
IEEE (2017)
8. Chen, T.H., Liao, Y.H., Chuang, C.Y., Hsu, W.T., Fu, J., Sun, M.: Show, adapt
and tell: adversarial training of cross-domain image captioner. In: The IEEE International Conference on Computer Vision (ICCV), pp. 521–530. IEEE (2017)
9. Cohen, G., Afshar, S., Tapson, J., van Schaik, A.: EMNIST: An extension of
MNIST to handwritten letters. arXiv preprint arXiv: 1702.05373 (2017)
10. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A large-scale
hierarchical image database. In: The IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), pp. 248–255. IEEE (2009)
11. Ding, Z., Shao, M., Fu, Y.: Missing modality transfer learning via latent low-rank
constraint. IEEE Trans. Image Proces. 24, 4322–4334 (2015)
12. Fu, Z., Xiang, T., Kodirov, E., Gong, S.: Zero-shot object recognition by semantic
manifold distance. In: The IEEE Conference on Computer Vision and Pattern
Recognition (CVPR). IEEE (2015)
13. Ganin, Y., Lempitsky, V.: Unsupervised domain adaptation by backpropagation.
In: Bach, F., Blei, D. (eds.) Proceedings of the 32nd International Conference on
Machine Learning (ICML-2015), vol. 37, pp. 1180–1189. PMLR (2015)
14. Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F.,
Marchand, M., Lempitsky, V.: Domain-adversarial training of neural networks. J.
Mach. Learn. Res. (JMLR) 17(59), 1–35 (2016)
15. Gebru, T., Hoffman, J., Li, F.F.: Fine-grained recognition in the wild: A multi-task
domain adaptation approach. In: The IEEE International Conference on Computer
Vision (ICCV), pp. 1349–1358. IEEE (2017)
16. Ghifary, M., Kleijn, W.B., Zhang, M., Balduzzi, D., Li, W.: Deep reconstructionclassification networks for unsupervised domain adaptation. In: Leibe, B., Matas,
J., Sebe, N., Welling, M. (eds.) ECCV 2016. LNCS, vol. 9908, pp. 597–613.
Springer, Cham (2016). https://doi.org/10.1007/978-3-319-46493-0 36
17. Gretton, A., Smola, A.J., Huang, J., Schmittfull, M., Borgwardt, K.M., Sch¨olkopf,
B.: Covariate shift and local learning by distribution matching, pp. 131–160. MIT
Press, Cambridge (2009)

18. Grother, P., Hanaoka, K.: NIST special database 19 handprinted forms and characters database. National Institute of Standards and Technology (2016)
19. Gupta, S., Hoffman, J., Malik, J.: Cross modal distillation for supervision transfer.
In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 2827–2836. IEEE (2016)
20. Haeusser, P., Frerix, T., Mordvintsev, A., Cremers, D.: Associative domain adaptation. In: The IEEE International Conference on Computer Vision (ICCV), pp.
2765–2773. IEEE (2017)
21. Hoffman, J., Gupta, S., Darrell, T.: Learning with side information through modality hallucination. In: The IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 826–834. IEEE (2016)
22. Iandola, F.N., Han, S., Moskewicz, M.W., Ashraf, K., Dally, W.J., Keutzer, K.:
SqueezeNet v1.1model. https://github.com/DeepScale/SqueezeNet/blob/master/
SqueezeNet v1.1/squeezenet v1.1.caffemodel. Accessed 11 Feb 2017
23. Iandola, F.N., Han, S., Moskewicz, M.W., Ashraf, K., Dally, W.J., Keutzer, K.:
SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
size. arXiv preprint arXiv: 1602.07360 (2016)
24. Jia, Y., et al.: Caffe: Convolutional architecture for fast feature embedding. arXiv
preprint arXiv: 1408.5093 (2014)
25. Koniusz, P., Tas, Y., Porikli, F.: Domain adaptation by mixture of alignments of
second- or higher-order scatter tensors. In: The IEEE Conference on Computer
Vision and Pattern Recognition (CVPR), pp. 4478–4487. IEEE (2017)
26. Krizhevsky, A., Sutskever, I., Hinton, G.E.: ImageNet classification with deep convolutional neural networks. In: Pereira, F., Burges, C.J.C., Bottou, L., Weinberger,
K.Q. (eds.) Advances in Neural Information Processing Systems (NIPS), vol. 25,
pp. 1097–1105. Curran Associates, Inc. (2012)
27. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P.: Gradient-based learning applied to
document recognition. Proc. IEEE 86(11), 2278–2324 (1998)
28. Li, D., Yang, Y., Song, Y.Z., Hospedales, T.M.: Deeper, broader and artier
domain generalization. In: The IEEE International Conference on Computer Vision
(ICCV). IEEE (2017)
29. Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S., Dean, J.: Distributed representations of words and phrases and their compositionality. In: Burges, C.J.C.,
Bottou, L., Welling, M., Ghahramani, Z., Weinberger, K.Q. (eds.) Advances in
Neural Information Processing Systems, vol. 26, pp. 3111–3119. Curran Associates
Inc. (2013)
30. Motiian, S., Piccirilli, M., Adjeroh, D.A., Doretto, G.: Unified deep supervised
domain adaptation and generalization. In: The IEEE International Conference on
Computer Vision (ICCV), pp. 5715–5725. IEEE (2017)
31. Ngiam, J., Khosla, A., Kim, M., Nam, J., Lee, H., Ng, A.Y.: Multimodal deep
learning. In: Getoor, L., Scheffer, T. (eds.) Proceedings of the 28th International
Conference on Machine Learning (ICML-2011), pp. 689–696. Omnipress (2011)
32. Saenko, K., Kulis, B., Fritz, M., Darrell, T.: Adapting visual category models to
new domains. In: Daniilidis, K., Maragos, P., Paragios, N. (eds.) ECCV 2010.
LNCS, vol. 6314, pp. 213–226. Springer, Heidelberg (2010). https://doi.org/10.
1007/978-3-642-15561-1 16
33. Saito, K., Ushiku, Y., Harada, T.: Asymmetric tri-training for unsupervised domain
adaptation. In: Precup, D., Teh, Y.W. (eds.) Proceedings of the 34th International
Conference on Machine Learning (ICML-2017), vol. 70, pp. 2988–2997. PMLR
(2017)
Zero-S


34. Sener, O., Song, H.O., Saxena, A., Savarese, S.: Learning transferrable representations for unsupervised domain adaptation. In: Lee, D.D., Sugiyama, M., Luxburg,
U.V., Guyon, I., Garnett, R. (eds.) Advances in Neural Information Processing
Systems (NIPS), vol. 29, pp. 2110–2118. Curran Associates, Inc. (2016)
35. Sohn, K., Liu, S., Zhong, G., Yu, X., Yang, M.H., Chandraker, M.: Unsupervised
domain adaptation for face recognition in unlabeled videos. In: The IEEE International Conference on Computer Vision (ICCV), pp. 3210–3218. IEEE (2017)
36. Song, S., Lichtenberg, S., Xiao, J.: SUN RGB-D: a RGB-D scene understanding benchmark suite. In: The IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 567–576. IEEE (2015)
37. Sun, B., Saenko, K.: Deep CORAL: correlation alignment for deep domain adaptation. In: Hua, G., J´egou, H. (eds.) ECCV 2016. LNCS, vol. 9915, pp. 443–450.
Springer, Cham (2016). https://doi.org/10.1007/978-3-319-49409-8 35
38. Szegedy, C., et al.: Going deeper with convolutions. In: The IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 1–9. IEEE (2015)
39. Tzeng, E., Hoffman, J., Darrell, T., Saenko, K.: Simultaneous deep transfer across
domains and tasks. In: The IEEE International Conference on Computer Vision
(ICCV), pp. 4068–4076. IEEE (2015)
40. Tzeng, E., Hoffman, J., Saenko, K., Darrell, T.: Adversarial discriminative domain
adaptation. In: The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 7167–7176. IEEE (2017)
41. Venkateswara, H., Eusebio, J., Chakraborty, S., Panchanathan, S.: Deep hashing
network for unsupervised domain adaptation. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5018–5027. IEEE (2017)
42. Wang, W., Arora, R., Livescu, K., Bilmes, J.: On deep multi-view representation
learning. In: Bach, F., Blei, D. (eds.) Proceedings of the 32th International Conference on Machine Learning (ICML-2015), vol. 37, pp. 1083–1092. PMLR (2015)
43. Wang, Y., Li, W., Dai, D., Gool, L.V.: Deep domain adaptation by geodesic distance minimization. In: The IEEE International Conference on Computer Vision
(ICCV), pp. 2651–2657. IEEE (2017)
44. Wu, C., Wen, W., Afzal, T., Zhang, Y., Chen, Y., Li, H.: A compact DNN:
approaching GoogLeNet-level accuracy of classification and domain adaptation.
In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 5668–5677. IEEE (2017)
45. Wulfmeier, M., Bewley, A., Posner, I.: Addressing appearance change in outdoor
robotics with adversarial domain adaptation. In: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 1551–1558. IEEE (2017)
46. Xiao, H., Rasul, K., Vollgraf, R.: Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv: 1702.05374 (2017)
47. Yan, H., Ding, Y., Li, P., Wang, Q., Xu, Y., Zuo, W.: Mind the class weight bias:
Weighted maximum mean discrepancy for unsupervised domain adaptation. In:
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.
2272–2281. IEEE (2017)
48. Yang, X., Ramesh, P., Chitta, R., Madhvanath, S., Bernal, E.A., Luo, J.: Deep multimodal representation learning from temporal data. In: The IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 5447–5455. IEEE (2017)
49. Yang, Y., Hospedales, T.M.: Zero-shot domain adaptation via kernel regression on
the grassmannian. In: Drira, H., Kurtek, S., Turaga, P. (eds.) BMVC Workshop
on Differential Geometry in Computer Vision. BMVA Press (2015)

50. Zhang, J., Li, W., Ogunbona, P.: Joint geometrical and statistical alignment for
visual domain adaptation. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1859–1867. IEEE (2017)
51. Zhang, Y., David, P., Gong, B.: Curriculum domain adaptation for semantic segmentation of urban scenes. In: The IEEE International Conference on Computer
Vision (ICCV), pp. 2020–2030. IEEE (2017)
