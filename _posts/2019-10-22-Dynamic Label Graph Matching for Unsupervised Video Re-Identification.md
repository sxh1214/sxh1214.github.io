---
layout:     post
title:      Dynamic Label Graph Matching for Unsupervised Video Re-Identification
subtitle:   
date:       2019-10-22
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Re-ID
    - SSL
---


# Abstract
This pa-per focuses on cross-camera label estimation, which can be subsequently used in feature learning to learn robust re-IDmodels.

Specifically, we propose to construct a graph forsamples in each camera, and then graph matching schemeis introduced for cross-camera labeling association.

this paper propose **a dynamic graphmatching (DGM) method.** 

DGM iteratively updates theimage graph and the label estimation process by learninga better feature space with intermediate estimated labels.

DGM is advantageous in two aspects:

1) the accuracy of es-timated labels is improved significantly with the iterations;

2) DGM is robust to noisy initial training data. 

>Code is available at www.comp.hkbu.edu.hk/%7emangye/



# Introduction
Different from theexisting unsupervised person re-ID methods, this paper isbased on a more customized solution, i.e.**, cross-cameralabel estimation. In other words, we aim to mine the la-bels (matched or unmatched video pairs) across cameras.With the estimated labels, the remaining steps are exactlythe same with supervised learning.**

In light of the above discussions, **this paper proposes adynamic graph matching (DGM) method to improve thelabel estimation performance for unsupervised video re-ID(the main idea is shown in Fig. 1).** Specifically, our pipelineis an iterative process. In each iteration, a bipartite graph isestablished, labels are then estimated, and then a discrim-inative metric is learnt. Throughout this procedure, labelsgradually become more accurate, and the learnt metric morediscriminative. Additionally, our method includes a labelre-weighting strategy which provides soft labels instead ofhard labels, a beneficial step against the noisy intermediatelabel estimation output from graph matching.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022111813641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**the main contributions are summarized as follows:**

We propose a dynamic graph matching (DGM) methodto estimate cross-camera labels for unsupervised re-ID, which is robust to distractors and noisy initial train-ing data. 

Our experiment confirms that DGM is only slightly in-ferior to its supervised baselines and yields competi-tive re-ID accuracy compared with existing unsuper-vised re-ID methods on three video benchmarks.

# Methods
## Graph Matching for Video Re-ID
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022164002995.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019102216402113.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022164033180.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022164104911.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022164146997.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022164158437.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
关于公示5的解释：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022190801615.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022191625250.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
这张图告诉了我们，video based和image based的区别。video 对应多张图，而image对应一张图.


## Dynamic Graph Matching
**However, there still remains two obvi-ous shortcomings:**

we need to learna discriminative feature space to optimize the graphmatching results.

Therefore, it is reasonable to re-encode theweights of labels for overall learning, especially forthe uncertain estimated positive video pairs.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022195028374.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

### Label Re-weighting
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022193513148.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022193542966.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022193602653.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022193621511.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**The label re-weighting scheme has the following advan-tages:**

(1) for positive video pairs, it could filter some falsepositives and then assign different positive sample pairs dif-ferent weights; 

(2) for negative video pairs, a number ofeasy negatives would be filtered. The re-weighing schemeis simple but effective as shown in the experiments.

### Metric Learning with Re-weighted Labels
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022194718908.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022194730563.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

### Iterative Updating
**The proposed approach is summarized in Algorithm 1.**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191022200530189.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
# References
[1] A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAMjournal on imaging sciences, 2009. 5
[2] M. Cho, J. Lee, and K. M. Lee. Reweighted random walksfor graph matching. In ECCV, 2010. 2
[3] H. Fan, L. Zheng, and Y. Yang. Unsupervised person re-identification: Clustering and fine-tuning. ar Xiv preprintar Xiv:1705.10444, 2017. 2
[4] S. Hamid Rezatofighi, A. Milan, Z. Zhang, Q. Shi, A. Dick,and I. Reid. Joint probabilistic data association revisited. InICCV, 2015. 2, 3
[5] S. Hamid Rezatofighi, A. Milan, Z. Zhang, Q. Shi, A. Dick,and I. Reid. Joint probabilistic matching using m-best solu-tions. In CVPR, 2016. 2, 3
[6] M. Hirzer, C. Beleznai, P. M. Roth, and H. Bischof. Personre-identification by descriptive and discriminative classifica-tion. In Image analysis, 2011. 5
[7] S. Karanam, Y. Li, and R. J. Radke. Person re-identificationwith discriminatively trained viewpoint invariant dictionar-ies. In ICCV, 2015. 7, 8
[8] F. M. Khan and F. Bremond. Unsupervised data associationfor metric learning in the context of multi-shot person re-identification. In AVSS, 2016. 2, 7, 8
[9] E. Kodirov, T. Xiang, Z. Fu, and S. Gong. Person re-identification by unsupervised l1 graph learning. In ECCV,2016. 1, 2, 7, 
8[10] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenetclassification with deep convolutional neural networks. InNIPS, 2012. 7
[11] X. Lan, A. J. Ma, P. C. Yuen, and R. Chellappa. Joint sparserepresentation and robust feature-level fusion for multi-cuevisual tracking. IEEE TIP, 2015. 1
[12] X. Lan, P. C. Yuen, and R. Chellappa. Robust mil-basedfeature template learning for object tracking. In AAAI, 2017.1
[13] M. Leordeanu, R. Sukthankar, and M. Hebert. Unsupervisedlearning for graph matching. In IJCV, 2012. 2, 3
[14] S. Liao, Y. Hu, X. Zhu, and S. Z. Li. Person re-identificationby local maximal occurrence representation and metriclearning. In CVPR, 2015. 1, 6, 7, 8
[15] S. Liao and S. Z. Li. Efficient psd constrained asymmetricmetric learning for person re-identification. In ICCV, 2015.5, 6, 7, 8
[16] K. Liu, B. Ma, W. Zhang, and R. Huang. A spatio-temporal appearance representation for viceo-based pedes-trian re-identification. In ICCV, 2015. 1, 3, 5, 7, 8
[17] Z. Liu, D. Wang, L. Zheng, and H. Lu. A labeling-by-searchapproach for unsupervised person re-identification. In ICCV,2017. 2
[18] A. J. Ma, P. C. Yuen, and J. Li. Domain transfer support vec-tor ranking for person re-identification without target cameralabel information. In ICCV, 2013. 2
[19] T. Matsukawa, T. Okabe, E. Suzuki, and Y. Sato. Hierarchi-cal gaussian descriptor for person re-identification. In CVPR,2016. 1
[20] N. Mc Laughlin, J. Martinez del Rincon, and P. Miller. Re-current convolutional network for video-based person re-identification. In CVPR, 2016. 1
[21] R. Panda, A. Bhuiyan, V. Murino, and A. K. Roy-Chowdhury. Unsupervised adaptive re-identification in openworld dynamic camera networks. In CVPR, 2017. 2
[22] P. Peng, T. Xiang, Y. Wang, and a. et. Unsupervised cross-dataset transfer learning for person re-identification. InCVPR, 2016. 1, 2
[23] Y. Tian, J. Yan, H. Zhang, Y. Zhang, X. Yang, and H. Zha. Onthe convergence of graph matching: Graduated assignmentrevisited. In ECCV, 2012. 5
[24] T. Wang, S. Gong, X. Zhu, and S. Wang. Person re-identification by video ranking. In ECCV, 2014. 5, 7
[25] Z. Wang, R. Hu, C. Liang, and et al. Zero-shot person re-identification via cross-view consistency. In IEEE TMM,2016. 3
[26] Z. Wang, R. Hu, Y. Yu, and et al. Statistical inference ofgaussian-laplace distribution for person verification. In ACMMM, 2017. 5
[27] Y. Xu, L. Lin, W.-S. Zheng, and X. Liu. Human re-identification by matching compositional template with clus-ter sampling. In ICCV, 2013. 2
[28] J. Yan, M. Cho, H. Zha, X. Yang, and S. M. Chu. Multi-graph matching via affinity optimization with graduated con-sistency regularization. In IEEE TPAMI, 2016. 1, 2
[29] M. Yang, P. Zhu, L. Van Gool, and L. Zhang. Face recogni-tion based on regularized nearest points between image sets.In FG, 2013. 6
[30] Y. Yang, J. Yang, J. Yan, S. Liao, D. Yi, and S. Z. Li. Salientcolor names for person re-identification. In ECCV, 2014. 1
[31] M. Ye, J. Chen, Q. Leng, and et al. Coupled-view basedranking optimization for person re-identification. In MMM,2015. 3
[32] M. Ye, C. Liang, Z. Wang, Q. Leng, and J. Chen. Rankingoptimization for person re-identification via similarity anddissimilarity. In ACM MM, 2015. 3
[33] M. Ye, C. Liang, Y. Yu, and et al. Person re-identificationvia ranking aggregation of similarity pulling and dissimilar-ity pushing. In IEEE TMM, 2016. 1
[34] J. You, A. Wu, X. Li, and W.-S. Zheng. Top-push video-based person re-identification. In CVPR, 2016. 1, 5, 6
[35] Z. Zhang and V. Saligrama. Prism: Person re-identificationvia structured matching. In IEEE TCSVT, 2016. 2
[36] R. Zhao, W. Ouyang, and X. Wang. Unsupervised saliencelearning for person re-identification. In CVPR, 2013. 1, 2, 7,8
[37] L. Zheng, Z. Bie, Y. Sun, J. Wang, C. Su, S. Wang, andQ. Tian. Mars: A video benchmark for large-scale personre-identification. In ECCV, 2016. 1, 5, 6, 7, 8
[38] L. Zheng, Y. Yang, and A. G. Hauptmann. Person re-identification: Past, present and future. ar Xiv, 2016. 1
[39] L. Zheng, Y. Yang, and Q. Tian. SIFT meets CNN: A decadesurvey of instance retrieval. IEEE TPAMI, 2017. 1
[40] X. Zhu, X.-Y. Jing, F. Wu, and H. Feng. Video-based personre-identification by simultaneously learning intra-video andinter-video distance metrics. In IJCAI, 2016. 5, 6
