---
layout:     post
title:      ZstGAN: An Adversarial Approach forUnsupervised Zero-Shot Image-to-Image Translation
subtitle:   
date:       2019-10-24
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - GAN
    - zero-shot
    - SSL
    - Computer Vision
    - Domain Adaptation
---


# Abstract
 **In this work In this workwe,**
we propose a framework calledZstGAN: By introducing an adversarial training scheme,ZstGAN learns to model each domain with domain-specificfeature distribution that is semantically consistent on visionand attribute modalities. 

**Our code is publicly available at https://github.com/linjx-ustc1106/ZstGAN-PyTorch.**


# Introduction

**Existing image-to-image translation usually workson the following setting:**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023201129840.png)

**One limitation of existing models is**, the fij ’s can onlyachieve mappings among these given domains, without thegeneralization abilities to other unseen domains. 

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023201212942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

 Therefore we aim to generalizef to unseen domains as shown in the bottom half part ofFigure 1.

**In order to generalize to unseen classes, a com-mon assumption in zero-shot learning assuming** is that someside-information about the classes is available, such as classattributes or textual descriptions, which provides semanticinformation about the classes.


**wepropose a new problem, unsupervised zero-shot image-to-image translation (briefly, UZSIT).**


**Compared to the stan-dard ZSL, UZSIT is more challenging:**
(1) The targetof image translation is more complex than classification,which not only requires us to generate representative fea-tures across seen and unseen domains but also generate rea-sonable translation images. 

(2) Unlike ZSL methods trainedin a supervised way on seen domains, we do not have anypaired data between any two domains. 


**There aretwo key steps in ZstGAN.**

1、 We model each seen/unseen domain using a domain-specific feature distribution constrained by semantic con-sistency. 

2、We disentangle domain-invariant features from thedomain-specific features and combine them to generatetranslation results, which is achieved by one adversariallearning loss and two reconstruction losses.

**We work on two datasets** commonly used in ZSL,Caltech-UCSD-Birds 200-2011 (CUB) [31] and OxfordFlowers (FLO) [25], 


# Methods
## Problem Formulation
We provide a mathematical formulation of UZSIT in thissubsection.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023203836873.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**The objective of UZSIT** is to train an image-to-imagetranslation model f on S without touching U .

**An assumption that S and U shares a common semanticspace is required.** Specifically, while S and U have differ-ent category sets (Lsand Lu), they are required to share thesame image and attribute spaces (X and A) where semanticinformation is extracted from.

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023203948497.png)

在现有的图像-图像翻译模型中，通常只提取不同领域的特定特征，而不将其描述在一个共同的语义空间中。
领域特定的特征不仅应该区分不同的领域，而且应该具有代表性，以便在一个共同的语义空间中对齐不同的领域。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023204038864.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## Architecture
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023204250170.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024103045356.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**The objective function is designed according to the fol-lowing criteria:**
### (1) Domain-specific features with semantic consistency
In detail, the adversarialtraining objective for Ev and Ea isIn detail, the adversarialtraining objective for Ev and Ea is
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024103558506.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024103617328.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

### (2) Domain-invariant features disentanglement

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024103651946.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024103707687.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

### (3)The overall training objective
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191024103729721.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

	

# Reference
[1] Z. Akata, S. Reed, D. Walter, H. Lee, and B. Schiele. Evaluation of output embeddings for ﬁne-grained image classiﬁcation. InProceedingsoftheIEEEConferenceonComputer Vision and Pattern Recognition, pages 2927–2936, 2015. 
[2] M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein generativeadversarialnetworks. InInternationalConferenceon Machine Learning, pages 214–223, 2017. 
[3] S.BenaimandL.Wolf. One-shotunsupervisedcrossdomain translation. arXiv preprint arXiv:1806.06029, 2018. 
[4] M. Bucher, S. Herbin, and F. Jurie. Generating visual representations for zero-shot classiﬁcation. In International Conference on Computer Vision (ICCV) Workshops: TASK-CV: Transferring and Adapting Source Knowledge in Computer Vision, 2017. 
[5] W. Chao, S. Changpinyo, B. Gong, and F. Sha. An empirical study and analysis of generalized zero-shot learning for object recognition in the wild. In Computer Vision - ECCV 2016 - 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II, pages 52– 68, 2016. 
[6] X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, andP.Abbeel. Infogan: Interpretablerepresentationlearning by information maximizing generative adversarial nets. In Advances in neural information processing systems, pages 2172–2180, 2016.
 [7] Y. Choi, M. Choi, M. Kim, J.-W. Ha, S. Kim, and J. Choo. Stargan: Uniﬁed generative adversarial networks for multidomainimage-to-imagetranslation. InTheIEEEConference on Computer Vision and Pattern Recognition (CVPR), June 2018. 
[8] R. Felix, V. B. Kumar, I. Reid, and G. Carneiro. Multimodal cycle-consistent generalized zero-shot learning. In
Proceedings of the European Conference on Computer Vision (ECCV), pages 21–37, 2018.
 [9] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D.Warde-Farley,S.Ozair,A.Courville,andY.Bengio. Generative adversarial nets. In Advances in neural information processing systems, pages 2672–2680, 2014. 
[10] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville. Improved training of wasserstein gans. In Advances in Neural Information Processing Systems, pages 5767–5777, 2017.
 [11] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 
[12] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30, pages 6626–6637. Curran Associates, Inc., 2017. 
[13] X.Huang,M.-Y.Liu,S.Belongie,andJ.Kautz. Multimodal unsupervised image-to-image translation. In ECCV, 2018. 
[14] P. Isola, J. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5967–5976, July 2017.
 [15] T. Kim, M. Cha, H. Kim, J. K. Lee, and J. Kim. Learning to discover cross-domain relations with generative adversarial networks. In Proceedings of the 34th International Conference on Machine Learning, pages 1857–1865, 2017. 
[16] D. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
 [17] D. P. Kingma and M. Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 
[18] C. H. Lampert, H. Nickisch, and S. Harmeling. Learning to detect unseen object classes by between-class attribute transfer. In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pages 951–958. IEEE, 2009. 
[19] H.-Y. Lee, H.-Y. Tseng, J.-B. Huang, M. K. Singh, and M.H. Yang. Diverse image-to-image translation via disentangled representations. In European Conference on Computer Vision, 2018. 
[20] J. Lin, Y. Xia, S. Liu, T. Qin, Z. Chen, and J. Luo. Exploring explicit domain supervision for latent space disentanglement in unpaired image-to-image translation. arXiv preprint arXiv:1902.03782, 2019.
 [21] J. Lin, Y. Xia, T. Qin, Z. Chen, and T.-Y. Liu. Conditional image-to-image translation. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)(July 2018), pages 5524–5532, 2018. 
[22] Y. Long, L. Liu, F. Shen, L. Shao, and X. Li. Zero-shot learning using synthesised unseen visual data with diffusion regularisation. IEEE transactions on pattern analysis and machine intelligence, 40(10):2498–2512, 2018.
[23] L. v. d. Maaten and G. Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(Nov):2579–2605, 2008. 
[24] X.Mao,Q.Li,H.Xie,R.Y.Lau,Z.Wang,andS.PaulSmolley. Least squares generative adversarial networks. In ProceedingsoftheIEEEInternationalConferenceonComputer Vision, pages 2794–2802, 2017.
 [25] M.-E. Nilsback and A. Zisserman. Automated ﬂower classiﬁcation over a large number of classes. In Computer Vision, Graphics & Image Processing, 2008. ICVGIP’08. Sixth Indian Conference on, pages 722–729. IEEE, 2008.
  [26] M. Norouzi, T. Mikolov, S. Bengio, Y. Singer, J. Shlens, A.Frome,G.S.Corrado,andJ.Dean. Zero-shotlearningby convexcombinationofsemanticembeddings. arXivpreprint arXiv:1312.5650, 2013. 
[27] S. Reed, Z. Akata, H. Lee, and B. Schiele. Learning deep representations of ﬁne-grained visual descriptions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 49–58, 2016. 
[28] B. Romera-Paredes and P. Torr. An embarrassingly simple approach to zero-shot learning. In International Conference on Machine Learning, pages 2152–2161, 2015. 
[29] R.Socher,M.Ganjoo,C.D.Manning,andA.Ng. Zero-shot learningthroughcross-modaltransfer. InAdvancesinneural information processing systems, pages 935–943, 2013. 
[30] R.Socher,M.Ganjoo,C.D.Manning,andA.Ng. Zero-shot learning through cross-modal transfer. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 26, pages 935–943. Curran Associates, Inc., 2013.
 [31] C.Wah,S.Branson,P.Welinder,P.Perona,andS.Belongie. The Caltech-UCSD Birds-200-2011 Dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. 
[32] D.Wang,Y.Li,Y.Lin,andY.Zhuang. Relationalknowledge transfer for zero-shot learning. In AAAI, volume 2, page 7, 2016. 
[33] W. Wang, Y. Pu, V. K. Verma, K. Fan, Y. Zhang, C. Chen, P.Rai,andL.Carin. Zero-shotlearningviaclass-conditioned deep generative models. In Thirty-Second AAAI Conference on Artiﬁcial Intelligence, 2018.
 [34] Y. Xian, T. Lorenz, B. Schiele, and Z. Akata. Feature generating networks for zero-shot learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2018. 
[35] Z. Yi, H. Zhang, P. Tan, and M. Gong. Dualgan: Unsupervised dual learning for image-to-image translation. In The IEEE International Conference on Computer Vision (ICCV), Oct 2017. 
[36] J.-Y.Zhu,T.Park,P.Isola,andA.A.Efros. Unpairedimageto-image translation using cycle-consistent adversarial networks. In The IEEE International Conference on Computer Vision (ICCV), Oct 2017.

