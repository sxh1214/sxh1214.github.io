---
layout:     post
title:      深度学习实验环境配置记录(ubuntu+docker+python+conda+pytorch)
subtitle:
date:       2019-11-22
author:     JoselynZhao
header-img: img/post-bg-os-metro.jpg
catalog: true
tags:
    - Deep Learning
    - Python
    - PyTorch
    - Docker
---

@[toc]
# ubuntu下安装docker
参考链接 :[菜根谭](https://www.cnblogs.com/jiyang2008/p/9014960.html)

## 安装
- 由于apt官方库里的docker版本可能比较旧，所以先卸载可能存在的旧版本
 `sudo apt-get remove docker docker-engine docker-ce docker.io`
- 更新apt包索引
`sudo apt-get update`

- 安装以下包以使apt可以通过HTTPS使用存储库（repository）
 `sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common`

- 添加Docker官方的GPG密钥：
 `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -`
- 使用下面的命令来设置stable存储库：
 `sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $$(lsb_release -cs) stable"`
- 再更新一下apt包索引：
 `sudo apt-get update`
- 安装最新版本的Docker CE：
 `sudo apt-get install -y docker-ce`
- 列出可用的版本：
 `apt-cache madison docker-ce`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122102234390.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
- 选择要安装的特定版本，第二列是版本字符串，第三列是存储库名称，它指示包来自哪个存储库，以及扩展它的稳定性级别。要安装一个特定的版本，将版本字符串附加到包名中，并通过等号(=)分隔它们：
` sudo apt-get install docker-ce=<VERSION>`

## 验证
- 查看docker服务是否启动：
`
systemctl status docker
`

- 若未启动，则启动docker服务：
 `
sudo systemctl start docker
`

- 经典的hello world：
`sudo docker run hello-world`


# docker 添加容器
参考链接: [菜鸟教程](https://www.runoob.com/docker/docker-container-usage.html)
- 查看docker是否安装成功
 `docker`
- 使用 ubuntu 镜像启动一个容器
 `docker run -it ubuntu /bin/bash`
> -i: 交互式操作。
-t: 终端。
ubuntu: ubuntu 镜像。
/bin/bash：放在镜像名后的是命令，这里我们希望有个交互式 Shell，因此用的是 /bin/bash。

- docker 退出容器
 `exit`
- docker 查看所有的容器
 `docker ps -a`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122103158617.png)
- docker 查看正在运行的容器
 `docker ps`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122103221445.png)
- docker 启动已经停止的容器
 `docker  start  xxxx`
> xxx  为容器的container ID

- docker 停止容器
 `docker stop xxxx`

- docker 进入容器
 `docker attach xxxx ` 或者 `docker exec -it xxxx /bin/bash`(exit之后,容器仍然保持启动状态)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122103605647.png)

# docker容器配置
- 搜索和anaconda相关的容器
 `docker search anaconda`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122162559333.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
- 选择需要的容器,并pull下来,我们这里选择用第一个continuumio/anaconda3
 `docker pull continuumio/anaconda3`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122162722406.png)
- 查看所有的容器
 `docker images`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122162820547.png)
可以看到我们刚才pull的容器已经位列其中了 .

- 进入容器
 `docker run -ti continuumio/anaconda3 /bin/bash`
- 查看conda list
 `conda list`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122163123796.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

- 查看conda 虚拟环境
 `conda env list`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122164118905.png)
- 拷贝一个已有的虚拟环境到新的虚拟环境
 `conda  create -n  new_name  --clone exist_name`
- 创建一个新的虚拟环境,同时制定Python版本..
 `conda create -n env_name python=3.6`
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122164640965.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
- 激活新的虚拟环境
 `conda activate eug`
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112216481087.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

# conda 安装 pytorch
参考链接:  [Plenari](https://www.jianshu.com/p/1d5b91b19fb5)
- 添加Anaconda的TUNA镜像
` conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/`
- 设置搜索时显示通道地址
 `conda config --set show_channel_urls yes`
- pytorch GPU的命令如图所示：
 `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
- pytorch CPU的命令如图所示
 `conda install pytorch-cpu -c `
 `pytorch pip3 install torchvision`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122171304835.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
# 实验项目准备
- 拷贝实验项目数据集到docker的容器中,在服务器上执行
`docker cp DukeMTMC-VideoReID.zip 521fa25090a7:/home/`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122173239167.png)
- 通过git clone 项目
 `git clone https://.......`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122173604386.png)
 > 剩下的工作就不用我说了 ......

# 附录
附录的故事太长,简直一言难尽.
做完刚才的哪些准备之后,发现, 没有显卡驱动,没有cuda.
所以 ,最开始docker search 的时候,还是先search一下带pytorch的容器吧. 这样里面的显卡环境是配置好了的.
最后在带pytorch的容器里 还是安装了一下 pytorch-GPU
参考链接:[pytorch.org](https://pytorch.org/get-started/locally/)
```shell
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

最后就是看你的项目还需要什么了, 缺什么就装什么
大多数都通过`conda install ` 进行安装. 值得一提的是,`metric_learning` 用conda安装不上, 灵机一动 `pip install ` 就可以啦.
