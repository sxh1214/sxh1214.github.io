---
layout:     post
title:      Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification
subtitle:   
date:       2019-10-21
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Re-ID
    - SSL
    - Domain Adaptation
---



# main methords
In our at-tempt, we present a “learning via translation” framework.In the baseline, we translate the labeled images from sourceto target domain in an unsupervised manner.


we propose to preserve two types of unsupervised similari-ties, 1) self-similarity of an image before and after transla-tion, and 2) domain-dissimilarity of a translated source im-age and a target image. Both constraints are implementedin the similarity preserving generative adversarial network(SPGAN) which consists of an Siamese network and a Cy-cleGAN.


![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021161642969.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021161810116.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021172955414.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021204548228.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)



![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021173732827.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**A brief summary of different methods considered in this paper ispresented in Table 1.**![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021161913998.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021162936817.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021164816465.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021171458275.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021173041335.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

References
[1] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean,M. Devin, S. Ghemawat, G. Irving, M. Isard, M. Kudlur,J. Levenberg, R. Monga, S. Moore, D. G. Murray, B. Steiner,P. A. Tucker, V. Vasudevan, P. Warden, M. Wicke, Y. Yu, andX. Zheng. Tensorflow: A system for large-scale machinelearning. In OSDI, 2016. 5
[2] H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, andM. Marchand. Domain-adversarial neural networks. arXivpreprint arXiv:1412.4446, 2014. 2
[3] S. Benaim and L. Wolf. One-sided unsupervised domainmapping. In NIPS, 2017. 2
[4] K. Bousmalis, N. Silberman, D. Dohan, D. Erhan, and D. Kr-ishnan. Unsupervised pixel-level domain adaptation withgenerative adversarial networks. In CVPR, 2017. 1, 2
[5] T. Q. Chen and M. Schmidt. Fast patch-based style transferof arbitrary style. arXiv preprint arXiv:1612.04337, 2016. 2
[6] H. Fan, L. Zheng, and Y. Yang. Unsupervised person re-identification: Clustering and fine-tuning. arXiv preprintarXiv:1705.10444, 2017. 1, 3, 6, 8
[7] M. Farenzena, L. Bazzani, A. Perina, V. Murino, andM. Cristani. Person re-identification by symmetry-driven ac-cumulation of local features. In CVPR, 2010. 2
[8] P. Felzenszwalb, D. McAllester, and D. Ramanan. A dis-criminatively trained, multiscale, deformable part model. InCVPR, 2008. 5
[9] B. Fernando, A. Habrard, M. Sebban, and T. Tuytelaars. Un-supervised visual domain adaptation using subspace align-ment. In ICCV, 2013. 2
[10] Y. Ganin and V. S. Lempitsky. Unsupervised domain adap-tation by backpropagation. In ICML, 2015. 2
[11] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle,F. Laviolette, M. Marchand, and V. S. Lempitsky. Domain-adversarial training of neural networks. Journal of MachineLearning Research, 2016. 2
[12] L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transferusing convolutional neural networks. In CVPR, 2016. 2
[13] B. Gong, Y. Shi, F. Sha, and K. Grauman. Geodesic flowkernel for unsupervised domain adaptation. In CVPR, 2012.2
[14] D. Gray and H. Tao. Viewpoint invariant pedestrian recogni-tion with an ensemble of localized features. In ECCV, 2008.2
[15] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Sch ¨olkopf, andA. J. Smola. A kernel two-sample test. Journal of MachineLearning Research, 2012. 2
[16] R. Hadsell, S. Chopra, and Y. LeCun. Dimensionality re-duction by learning an invariant mapping. In CVPR, 2006.4
[17] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learningfor image recognition. In CVPR, 2016. 4, 6
[18] J. Hoffman, E. Tzeng, T. Park, and J.-Y. Zhu. Cycada: Cycle-consistent adversarial domain adaptation. arXiv preprintarXiv:1711.03213, 2017. 1, 2, 3
[19] X. Huang and S. J. Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV, 2017.2
[20] P. Isola, J. Zhu, T. Zhou, and A. A. Efros. Image-to-imagetranslation with conditional adversarial networks. In CVPR,2017. 2
[21] J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses forreal-time style transfer and super-resolution. In ECCV, 2016.2
[22] T. Kim, M. Cha, H. Kim, J. K. Lee, and J. Kim. Learning todiscover cross-domain relations with generative adversarialnetworks. In ICML, 2017. 1, 2
[23] C. Li and M. Wand. Precomputed real-time texture synthesiswith markovian generative adversarial networks. In ECCV,2016. 2
[24] Y. Li, C. Fang, J. Yang, Z. Wang, X. Lu, and M. Yang. Di-versified texture synthesis with feed-forward networks. InCVPR, 2017. 2
[25] Y. Li, N. Wang, J. Liu, and X. Hou. Demystifying neuralstyle transfer. In IJCAI, 2017. 2
[26] S. Liao, Y. Hu, X. Zhu, and S. Z. Li. Person re-identificationby local maximal occurrence representation and metriclearning. In CVPR, 2015. 2, 8
[27] M. Liu, T. Breuel, and J. Kautz. Unsupervised image-to-image translation networks. In NIPS, 2017. 1, 2, 3
[28] M. Liu and O. Tuzel. Coupled generative adversarial net-works. In NIPS, 2016. 1, 2
[29] Z. Liu, D. Wang, and H. Lu. Stepwise metric promotion forunsupervised video person re-identification. In ICCV, 2017.3
[30] M. Long, Y. Cao, J. Wang, and M. I. Jordan. Learning trans-ferable features with deep adaptation networks. In ICML,2015. 2
[31] M. Long, G. Ding, J. Wang, J. Sun, Y. Guo, and P. S. Yu.Transfer sparse coding for robust image representation. InCVPR, 2013. 2
[32] B. Ma, Y. Su, and F. Jurie. Covariance descriptor basedon bio-inspired features for person re-identification and faceverification. Image Vision Comput., 2014. 2
[33] T. Matsukawa, T. Okabe, E. Suzuki, and Y. Sato. Hierarchi-cal gaussian descriptor for person re-identification. In CVPR,2016. 2
[34] S. Motiian, M. Piccirilli, D. A. Adjeroh, and G. Doretto. Uni-fied deep supervised domain adaptation and generalization.In ICCV, 2017. 2
[35] P. Peng, T. Xiang, Y. Wang, M. Pontil, S. Gong, T. Huang,and Y. Tian. Unsupervised cross-dataset transfer learning forperson re-identification. In CVPR, 2016. 3, 8
[36] E. Ristani, F. Solera, R. Zou, R. Cucchiara, and C. Tomasi.Performance measures and a data set for multi-target, multi-camera tracking. In European Conference on ComputerVision workshop on Benchmarking Multi-Target Tracking,2016. 5
[37] K. Saenko, B. Kulis, M. Fritz, and T. Darrell. Adapting vi-sual category models to new domains. In ECCV, 2010. 
2[38] B. Sun, J. Feng, and K. Saenko. Return of frustratingly easydomain adaptation. In AAAI, 2016. 2
[39] Y. Sun, L. Zheng, W. Deng, and S. Wang. SVDNet for pedes-trian retrieval. In ICCV, 2017. 7, 8
