---
layout:     post
title:      Stepwise Metric Promotion for Unsupervised Video Person Re-identification
subtitle:   
date:       2019-10-21
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Re-ID
    - SSL
---

# Abstract
**two assumptions two assumptions**
1) different video track-lets typically contain different persons。

2)within each tracklet, the frames are mostly of the same per-son.
\
Our method is built onreciprocal nearest neighbor search and can eliminate thehard negative label matches, i.e., the cross-camera nearestneighbors of the false matches in the initial rank list. Thetracklet that passes the reciprocal nearest neighbor checkis considered to have the same ID with the query. 

# Introduction

**our workis motivated from three aspects**.
 First, videos contain muchricher information than single images,
Second, video tracklets, produced by pedestrian detec-tion and tracking (Fig. 1), are reliable data source for unsu-pervised learning methods. 
Third, feature learning using tracklets from the sameview may result in low discriminative ability. 

 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021211255520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
## framework
**In brief,two steps are involved:**

1) classiﬁer initialization using thetracklets in the same camera  

2) iterations between cross-camera tracklet association and featurelearning. 


## Our Method.
 The proposed metric promotion approachiterates between model update and label estimation. Forlabel estimation, under camera A, we use each tracklet asquery to search for its k nearest neighbors (NNs) in cameraB. Among these k candidates, the best match is selected asbeing associated with the query tracklet. We employ neg-ative mining to reduce the impact of false positive matchesin the k-NNs. This k-NN search process is then reverselyrepeated using the best match as query to see whether theinitial query is its best match, a conﬁrmation protocol toensure that the initial query and the best match are truly as-sociated. The the associated pairs are adopted for modelupdating. 

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191021212411304.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

# References
[1] Y. Cho and K. Yoon. Improving person re-identiﬁcation via pose-aware multi-shot matching. In CVPR, 2016.
[2] J. Dai, Y. Zhang, and H. Lu. Cross-view semantic projection learningfor person re-identiﬁcation. In PR, 2017.
[3] A. Dehghan, S. M. Assari, and M. Shah. GMMCP tracker: Globallyoptimal generalized maximum multi clique problem for multiple ob-ject tracking. In CVPR, 2015.
[4] W. F and Z. C. Label propagation through linear neighborhoods. InTKDE, 2008.
[5] H. Fan, L. Zheng, and Y. Yang. Unsupervised person re-identiﬁcation: Clustering and ﬁne-tuning. arXiv, 2017
[6] M. Farenzena, L. Bazzani, A. Perina, V. Murino, and M. Cristani.Person re-identiﬁcation by symmetry-driven accumulation of localfeatures. In CVPR, 2010.
[7] P. F. Felzenszwalb, R. B. Girshick, D. A. McAllester, and D. Ra-manan. Object detection with discriminatively trained part-basedmodels. TPAMI, 2010.
[8] D. Gray and H. Tao. Viewpoint invariant pedestrian recognition withan ensemble of localized features. In ECCV, 2008.
[9] O. Hamdoun, F. Moutarde, B. Stanciulescu, and B. Steux. Personre-identiﬁcation in multi-camera system by signature based on inter-est point descriptors collected on short video sequences. In ICDSC,2008.
[10] J. F. Henriques, J. Carreira, R. Caseiro, and J. Batista. Beyond hardnegative mining: Efﬁcient detector learning via block-circulant de-composition. In ICCV, 2013.
[11] M. Hirzer, C. Beleznai, P. M. Roth, and H. Bischof. Person re-identiﬁcation by descriptive and discriminative classiﬁcation. InSCIA, 2011.
[12] S. Karanam, Y. Li, and R. J. Radke. Person re-identiﬁcation withdiscriminatively trained viewpoint invariant dictionaries. In ICCV,2015.
[13] S. Karanam, Y. Li, and R. J. Radke. Sparse re-id: Block sparsity forperson re-identiﬁcation. In CVPR, 2015.
[14] M. K ¨ostinger, M. Hirzer, P. Wohlhart, P. M. Roth, and H. Bischof.Large scale metric learning from equivalence constraints. In CVPR,2012.
[15] N. Li, R. Jin, and Z. Zhou. Top rank optimization in linear time. InNIPS, 2014.
[16] W. Li, R. Zhao, T. Xiao, and X. Wang. Deepreid: Deep ﬁlter pairingneural network for person re-identiﬁcation. In CVPR, 2014.
[17] Y. Li, Z. Wu, S. Karanam, and R. J. Radke. Multi-shot human re-identiﬁcation using adaptive ﬁsher discriminant analysis. In BMVC,2015.
[18] Z. Li, S. Chang, F. Liang, T. S. Huang, L. Cao, and J. R. Smith.Learning locally-adaptive decision functions for person veriﬁcation.In CVPR, 2013
.[19] S. Liao, Y. Hu, X. Zhu, and S. Z. Li. Person re-identiﬁcation by localmaximal occurrence representation and metric learning. In CVPR,2015.
[20] S. Liao, G. Zhao, V. Kellokumpu, M. Pietik¨ainen, and S. Z. Li. Mod-eling pixel process with scale invariant local patterns for backgroundsubtraction in complex scenes. In CVPR, 2010
[21] C. Liu, C. C. Loy, S. Gong, and G. Wang. POP: person re-identiﬁcation post-rank optimisation. In ICCV, 2013.
[22] K. Liu, B. Ma, W. Zhang, and R. Huang. A spatio-temporal appear-ance representation for viceo-based pedestrian re-identiﬁcation. InICCV, 2015.[23] W. Liu and T. Zhang. Bidirectional label propagation over graphs.IJSI, 2013.
[24] X. Liu, M. Song, D. Tao, X. Zhou, C. Chen, and J. Bu. Semi-supervised coupled dictionary learning for person re-identiﬁcation.In CVPR, 2014.[25] B. M, H. S, and J. F. Hard negative mining for metric learning basedzero-shot classiﬁcation. In ECCV Workshops, 2016.[26] B. Ma, Y. Su, and F. Jurie. Bicov: a novel image representation forperson re-identiﬁcation and face veriﬁcation. In BMVC, 2012.
[27] B. Ma, Y. Su, and F. Jurie. Covariance descriptor based on bio-inspired features for person re-identiﬁcation and face veriﬁcation.IVC, 2014.[28] X. Ma, X. Zhu, S. Gong, X. Xie, J. Hu, K.-M. Lam, and Y. Zhong.Person re-identiﬁcation by unsupervised video matching. PR, 2017.[29] N. McLaughlin, J. Martinez del Rincon, and P. Miller. Recurrentconvolutional network for video-based person re-identiﬁcation. InCVPR, 2016.
[30] M. Niall, M. del Rincon Jesus, and M. Paul. Recurrent convolutionalnetwork for video-based person re-identiﬁcation. In CVPR, 2016.[31] S. Pedagadi, J. Orwell, S. A. Velastin, and B. A. Boghossian. Localﬁsher discriminant analysis for pedestrian re-identiﬁcation. In CVPR,2013.
[32] B. J. Prosser, W. Zheng, S. Gong, and T. Xiang. Person re-identiﬁcation by support vector ranking. In BMVC, 2010.
[33] P. Siva, C. Russell, and T. Xiang. In defence of negative mining forannotating weakly labeled data. In ECCV, 2012.
[34] C. Sun, D. Wang, and H. Lu. Person re-identiﬁcation via distancemetric learning with latent variables. 2017.
[35] M. Tetsu, O. Takahiro, S. Einoshin, and S. Yoichi. Hierarchical gaus-sian descriptor for person re-identiﬁcation. In CVPR, 2016
.[36] R. Wang, S. Shan, X. Chen, and W. Gao. Manifold-manifold distancewith application to face recognition based on image set. In CVPR,2008.
[37] T. Wang, S. Gong, X. Zhu, and S. Wang. Person re-identiﬁcation byvideo ranking. In ECCV, 2014.
[38] T. Wang, S. Gong, X. Zhu, and S. Wang. Person re-identiﬁcation bydiscriminative selection in video ranking. TPAMI, 2016.
[39] Y. Yan, B. Ni, Z. Song, C. Ma, Y. Yan, and X. Yang. Person re-identiﬁcation via recurrent feature aggregation. In ECCV, 2016
.[40] M. Ye, C. Liang, Y. yu, and etal. Person re-identiﬁcation via rankingaggregation of similarity pulling and dissimilarity pushing. In TMM,2016.
[41] M. Ye, J. Ma, J. Li, L. Zheng, and P. Yuen. Label graph matching forunsupervised video re-identiﬁcation. In ICCV, 2017.
[42] J. You, A. Wu, X. Li, and W.-S. Zheng. Top-push video-based personre-identiﬁcation. In CVPR, 2016.
[43] L. Zhang, T. Xiang, and S. Gong. Learning a discriminative nullspace for person re-identiﬁcation. In CVPR, 2016.
[44] Y. Zhang, B. Li, and H. L. A. I. X. Ruan. Sample-speciﬁc svm learn-ing for person re-identiﬁcation. In CVPR, 2016
.[45] Z. Zhang, X. Jing, and T. Wang. Label propagation based semi-supervised learning for software defect prediction. ASE, 2017.
[46] Z. Zhang, M. Zhao, and T. W. S. Chow. Label propagation andsoft-similarity measure for graph based constrained semi-supervisedlearning. In IJCNN, 2014
.[47] R. Zhao, W. Ouyang, and X. Wang. Unsupervised salience learningfor person re-identiﬁcation. In CVPR, 2013.
[48] L. Zheng, Z. Bie, Y. Sun, J. Wang, C. Su, S. Wang, and Q. Tian.MARS: A video benchmark for large-scale person re-identiﬁcation.In ECCV, 2016.
[49] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian. Scalableperson re-identiﬁcation: A benchmark. In ICCV, 2015.
[50] L. Zheng, Y. Yang, and A. G. Hauptmann. Person re-identiﬁcation:Past, present and future. arXiv, 2016.
[51] W. Zheng, S. Gong, and T. Xiang. Person re-identiﬁcation by proba-bilistic relative distance comparison. In CVPR, 2011.
[52] Z. Zheng, L. Zheng, and Y. Yang. Unlabeled samples generated bygan improve the person re-identiﬁcation baseline in vitro. In ICCV,2017.
[53] Z. Zhong, L. Zheng, D. Cao, and S. Li. Re-ranking person re-identiﬁcation with k-reciprocal encoding. In CVPR, 2017.
[54] X. Zhu, X. Jing, F. Wu, and H. Feng. Video-based person re-identiﬁcation by simultaneously learning intra-video and inter-videodistance metrics. In IJCAI, 2016.
