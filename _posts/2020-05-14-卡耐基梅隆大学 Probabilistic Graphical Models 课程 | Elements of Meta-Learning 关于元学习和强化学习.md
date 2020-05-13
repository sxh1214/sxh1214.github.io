---
layout:     post
title:      Elements of Meta-Learning 关于元学习和强化学习
subtitle:   卡耐基梅隆大学 Probabilistic Graphical Models 课程
date:       2020-05-14
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Graphical Models
    - Deep Learning
    - Meta Learning
---


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513071136297.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**Goals for the lecture:**
>Introduction & overview of the key methods and developments.
[Good starting point for you to start reading and understanding papers!]

[toc]
# Probabilistic Graphical Models | Elements of Meta-Learning
## 01 Intro to Meta-Learning
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513071814958.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
### Motivation and some examples
**When is standard machine learning not enough?**
Standard ML finally works for well-defined, stationary tasks.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513072004470.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
But how about the complex dynamic world, heterogeneous data from people and the interactive robotic systems?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513072123117.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
### General formulation and probabilistic view 
**What is meta-learning?**
**Standard learning:** Given a distribution over examples (single task), learn a function that minimizes the loss:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513072216953.png)
**Learning-to-learn**: Given a distribution over tasks, output an adaptation rule that can be used at test time to generalize from a task description
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513072423872.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**A Toy Example: Few-shot Image Classification**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513073708315.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513073906878.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**Other (practical) Examples of Few-shot Learning**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513074228241.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513074254909.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513074311352.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513074326299.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

### Gradient-based and other types of meta-learning
**Model-agnostic Meta-learning (MAML)** *与模型无关的元学习*
- Start with a common model initialization $$\theta$$
- Given a new task  $$T_i$$ , adapt the model  using a gradient step:
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051307472067.png)
- Meta-training is learning a shared initialization for all tasks:
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513074744155.png)
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513074837867.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
 
 **Does MAML Work?**
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513074859797.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**MAML from a Probabilistic Standpoint**
Training points: ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513075334371.png)
 testing points:![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513075348890.png) 
 MAML with log-likelihood loss*对数似然损失*:
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513075425477.png)
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513075436965.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
 

**One More Example: One-shot Imitation Learning** *模仿学习*
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051307450029.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**Prototype-based Meta-learning**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513080255117.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**Prototypes:**
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051308041013.png)
**Predictive distribution:**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513080430470.png)
**Does Prototype-based Meta-learning Work?**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513080509705.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**Rapid Learning or Feature Reuse** *特征重用*
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513080610519.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513080621173.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513080803928.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051308084427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
### Neural processes and relation of meta-learning to GPs
**Drawing parallels between meta-learning and GPs**
**In few-shot learning:**
- Learn to identify functions that generated the data from just a few examples.
- The function class and the adaptation rule encapsulate our prior knowledge. 

**Recall Gaussian Processes (GPs):** *高斯过程*
- Given a few (x, y) pairs, we can compute the predictive mean and variance. 
- Our prior knowledge is encapsulated in the kernel function.

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051308134428.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**Conditional Neural Processes**  *条件神经过程*
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051308143236.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051308150437.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513081516594.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513081722453.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)


**On software packages for meta-learning**
 A lot of research code releases (code is fragile and sometimes broken)
A few notable libraries that implement a few specific methods: 
- Torchmeta (https://github.com/tristandeleu/pytorch-meta)
- Learn2learn (https://github.com/learnables/learn2learn)
- Higher (https://github.com/facebookresearch/higher)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513081926940.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**Takeaways**
- Many real-world scenarios require building **adaptive systems** and cannot be solved using “learn-once” standard ML approach.
- Learning-to-learn (or meta-learning) attempts extend ML to rich multitask scenarios—instead of learning a function, learn a learning algorithm.
- Two families of widely popular methods:
	- Gradient-based meta-learning (MAML and such)
	- Prototype-based meta-learning (Protonets, Neural Processes, ...) 
	- Many hybrids, extensions, improvements (CAIVA, MetaSGD, ...)
- Is it about adaptation or learning good representations? Still unclear and depends on the task; having good representations might be enough.
- Meta-learning can be used as a mechanism for causal discovery.*因果发现* (See Bengio et al., 2019.)
## 02 Elements of Meta-RL
### What is meta-RL and why does it make sense?
**Recall the definition of learning-to-learn**
**Standard learning**: Given a distribution over examples (single task), learn a function that minimizes the loss：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513110447912.png)
**Learning-to-learn:** Given a distribution over tasks, output an adaptation rule that can be used at test time to generalize from a task description
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513110512805.png)
**Meta reinforcement learning (RL)**: Given a distribution over environments, train a policy update rule that can solve new environments given only **limited or no initial experience**.
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051311061682.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**Meta-learning for RL**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513110657817.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
### On-policy and off-policy meta-RL
**On-policy RL: Quick Recap** *符合策略的RL：快速回顾*
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513110804301.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**REINFORCE algorithm:**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513110827281.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**On-policy Meta-RL: MAML (again!)**
- Start with a common **policy** initialization $$\theta$$
- Given a new task  $$T_i$$ , collect data using initial policy, then adapt using a gradient step:
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051307472067.png)
- Meta-training is learning a shared initialization for all tasks:
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513074744155.png)
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513074837867.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**Adaptation as Inference** *适应推理*
Treat policy parameters, tasks, and all trajectories as random variables*随机变量*
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513111236201.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
meta-learning = learning a prior and **adaptation = inference**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513111337810.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
**Off-policy meta-RL: PEARL**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513111414738.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051311143358.png)

**Key points:**
- Infer latent representations z of each  task from the trajectory data.
- The inference networkq is decoupled from the policy, which enables off-policy learning.
- All objectives involve the inference and policy networks.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513111628751.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**Adaptation in nonstationary environments** *不稳定环境*
**Classical few-shot learning setup:**
- The tasks are i.i.d. samples from some underlying distribution.
- Given a new task, we get to interact with it before adapting.
- What if we are in a nonstationary environment (i.e. changing over time)? Can we still use meta-learning?
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513111828762.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
Example: adaptation to a learning opponent
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513111929628.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)Each new round is a new task. Nonstationary environment is a sequence of tasks.

**Continuous adaptation setup:** 
- The tasks are sequentially dependent.
- meta-learn to exploit dependencies
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513112155609.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

### Continuous adaptation

Treat policy parameters, tasks, and all trajectories as random variables
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513121640420.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

RoboSumo: a multiagent competitive env
an agent competes vs. an opponent, the opponent’s behavior changes over time
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200513121817180.png)

**Takeaways**
- Learning-to-learn (or meta-learning) setup is particularly suitable for multi-task reinforcement learning
- Both on-policy and off-policy RL can be “upgraded” to meta-RL:
	- On-policy meta-RL is directly enabled by MAML
	- Decoupling task inference and policy learning enables off-policy methods
- Is it about fast adaptation or learning good multitask representations? (See discussion in Meta-Q-Learning: https://arxiv.org/abs/1910.00125)
- Probabilistic view of meta-learning allows to use meta-learning ideas beyond distributions of i.i.d. tasks, e.g., continuous adaptation.
- Very active area of research.
