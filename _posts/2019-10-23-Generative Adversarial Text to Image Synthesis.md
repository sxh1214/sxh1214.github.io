---
layout:     post
title:      Generative Adversarial Text to Image Synthesis
subtitle:   
date:       2019-10-23
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - GAN
    - Computer Vision
---


# Abstract
**In this work,** we develop a novel deeparchitecture and GAN formulation to effectivelybridge these advances in text and image model-ing, translating visual concepts from charactersto pixels. 
# Introduction
In this work we are interested in translating text in the formof single-sentence human-written descriptions directly intoimage pixels.

Motivated by these works, **we aim to learn a mapping di-rectly from words and characters to image pixels.**

**To solve this challenging problem requires solving two sub-problems:**
first, learn a text feature representation that cap-tures the important visual details; 
and second, use these fea-tures to synthesize a compelling image that a human mightmistake for real.
> 其次，利用这些特征来合成一幅令人信服的图像，而人类可能会把这幅图像误认为是真实的。

**However, one difficult remaining issue not solved by deeplearning alone is that** 
the distribution of images conditionedon a text description is highly multimodal, in the sense thatthere are very many plausible configurations of pixels thatcorrectly illustrate the description. 
>在文本描述中，图像的分布是高度多模态的，在这个意义上说，有很多像素的合理配置可以正确地描述。

This conditional multi-modality is thus a very natural ap-plication for generative adversarial networks (Goodfellowet al., 2014), in which the generator network is optimized tofool the adversarially-trained discriminator into predictingthat synthetic images are real.
> 就是说 适合 用 GAN 来解决

**Our main contribution** in this work is to develop a sim-ple and effective GAN architecture and training strat-egy that enables compelling text to image synthesis ofbird and flower images from human-written descriptions.

# Background
In this section we briefly describe several previous worksthat our method is built upon.
>主要是介绍GAN

## Generative adversarial networks
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023105237650.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## Deep symmetric structured joint embedding

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023152502634.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)


# Methods

Our approach is to train a deep convolutional generativeadversarial network (DC-GAN) conditioned on text fea-tures encoded by a hybrid character-level convolutional-recurrent neural network.


## Network architecture
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023160151614.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023160350723.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023160358302.png)


## Matching-aware discriminator (GAN-CLS)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023162529773.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023162604752.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023162611727.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

## Learning with manifold interpolation (GAN-INT)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191023165114552.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)


Note that t1 and t2 may comefrom different images and even different categories.1

## Inverting the generator for style transfer
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019102316595177.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**Our implementation was builton top of dcgan.torch2.**
https://github.com/soumith/dcgan.torch

# Reference
Akata, Z., Reed, S., Walter, D., Lee, H., and Schiele, B. Evaluation of Output Embeddings for Fine-Grained Image Classiﬁcation. In CVPR, 2015.
 Ba, J. and Kingma, D. Adam: A method for stochastic optimization. In ICLR, 2015. Bengio, Y., Mesnil, G., Dauphin, Y., and Rifai, S. Better
 mixing via deep representations. In ICML, 2013.
Denton, E. L., Chintala, S., Fergus, R., et al. Deep generative image models using a laplacian pyramid of adversarial networks. In NIPS, 2015.
Donahue, J., Hendricks, L. A., Guadarrama, S., Rohrbach, M., Venugopalan, S., Saenko, K., and Darrell, T. Longterm recurrent convolutional networks for visual recognition and description. In CVPR, 2015.
Dosovitskiy, A., Tobias Springenberg, J., and Brox, T. Learning to generate chairs with convolutional neural networks. In CVPR, 2015.
Farhadi, A., Endres, I., Hoiem, D., and Forsyth, D. Describing objects by their attributes. In CVPR, 2009.
Fu, Y., Hospedales, T. M., Xiang, T., Fu, Z., and Gong, S. Transductivemulti-viewembeddingforzero-shotrecognition and annotation. In ECCV, 2014.
Gauthier, J. Conditional generative adversarial nets for convolutional face generation. Technical report, 2015.
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. In NIPS, 2014.
Gregor, K., Danihelka, I., Graves, A., Rezende, D., and Wierstra,D.Draw: Arecurrentneuralnetworkforimage generation. In ICML, 2015.
Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.
Karpathy, A. and Li, F. Deep visual-semantic alignments for generating image descriptions. In CVPR, 2015.
Kiros, R., Salakhutdinov, R., and Zemel, R. S. Unifying visual-semantic embeddings with multimodal neural language models. In ACL, 2014.
Kumar,N.,Berg,A.C.,Belhumeur,P.N.,andNayar,S.K. Attribute and simile classiﬁers for face veriﬁcation. In ICCV, 2009.
Lampert,C.H.,Nickisch,H.,andHarmeling,S. Attributebased classiﬁcation for zero-shot visual object categorization. TPAMI, 36(3):453–465, 2014.
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In ECCV. 2014.
Mansimov, E., Parisotto, E., Ba, J. L., and Salakhutdinov, R. Generating images from captions with attention. ICLR, 2016.
Mao, J., Xu, W., Yang, Y., Wang, J., and Yuille, A. Deep captioning with multimodal recurrent neural networks (m-rnn). ICLR, 2015.
Mirza, M. and Osindero, S. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014.
Ngiam, J., Khosla, A., Kim, M., Nam, J., Lee, H., and Ng, A. Y. Multimodal deep learning. In ICML, 2011.
Parikh, D. and Grauman, K. Relative attributes. In ICCV, 2011.
Radford, A., Metz, L., and Chintala, S. Unsupervised representation learning with deep convolutional generative adversarial networks. 2016.
Reed,S.,Sohn,K.,Zhang,Y.,andLee,H. Learningtodisentangle factors of variation with manifold interaction. In ICML, 2014.
Reed, S., Zhang, Y., Zhang, Y., and Lee, H. Deep visual analogy-making. In NIPS, 2015.
Reed,S.,Akata,Z.,Lee,H.,andSchiele,B. Learningdeep representations for ﬁne-grained visual descriptions. In CVPR, 2016.
Ren, M., Kiros, R., and Zemel, R. Exploring models and data for image question answering. In NIPS, 2015.
Sohn, K., Shang, W., and Lee, H. Improved multimodal deep learning with variation of information. In NIPS, 2014.
Srivastava, N. and Salakhutdinov, R. R. Multimodal learning with deep boltzmann machines. In NIPS, 2012.
Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. In CVPR, 2015.
Vinyals, O., Toshev, A., Bengio, S., and Erhan, D. Show and tell: A neural image caption generator. In CVPR, 2015.
Wah, C., Branson, S., Welinder, P., Perona, P., and Belongie, S. The caltech-ucsd birds-200-2011 dataset. 2011.
Wang, P., Wu, Q., Shen, C., Hengel, A. v. d., and Dick, A. Explicit knowledge-based reasoning for visual question answering. arXiv preprint arXiv:1511.02570, 2015.
Xu, K., Ba, J., Kiros, R., Courville, A., Salakhutdinov, R., Zemel, R., and Bengio, Y. Show, attend and tell: Neural imagecaptiongenerationwithvisualattention. InICML, 2015. Yan, X., Yang, J., Sohn, K., and Lee, H. Attribute2image: Conditional image generation from visual attributes. arXiv preprint arXiv:1512.00570, 2015. Yang, J., Reed, S., Yang, M.-H., and Lee, H. Weaklysupervised disentangling with recurrent transformations for 3d view synthesis. In NIPS, 2015. Zhu, Y., Kiros, R., Zemel, R., Salakhutdinov, R., Urtasun, R., Torralba, A., and Fidler, S. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In ICCV, 2015
