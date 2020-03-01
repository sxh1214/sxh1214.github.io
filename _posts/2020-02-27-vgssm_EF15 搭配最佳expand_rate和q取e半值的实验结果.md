---
layout:     post
title:      vgssm_EF15 搭配最佳expand_rate和q取e半值的实验结果
subtitle:   方差实验组
date:       2020-02-27
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - 科研之路
---

# 说明
- 运行文件： vrm/vcf03.py
- 运行命令：
```
python3.6 vcf03.py --exp_name vgssm_EF15 --exp_order 0 --EF 15 
# expand_rate = [1.3,1.2,1.1,1.08,1.06,1.04,1.02]
# stop_vari_step = len(expand_rate)
python3.6 vcf03.py --exp_name vgssm_EF15 --exp_order 0 --EF 15  --dataset mars --max_frames 100
# expand_rate = [1.2,1.1,1.08,1.06,1.04,1.02]
# stop_vari_step = len(expand_rate)
```
- 实验结果保存在 vgssm_EF15/0目录下
- baseline性能如下
![image](http://note.youdao.com/yws/res/48074/4F08C6E38B9A43099BB8336D313E6884)
！[image](http://note.youdao.com/yws/public/resource/0b969242ea3c48fbaaa18f9f9d222f23/FE239AD35E354E14BD452E2971535AFA?ynotemdtimestamp=1583034128243)


# 参数设定
## q=1/2e
### Duke

 || 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
---|---|---|---|---|---|---|---|---
query_rate| 0.65 |0.6|0.55|0.54|0.53|0.52|0.51|-|
expand_rate | 1.3| 1.2|1.1|1.08|1.06|1.04|1.02| -
### Mars
 || 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
---|---|---|---|---|---|---|---|---
query_rate|0.6|0.55|0.54|0.53|0.52|0.51|-|-
expand_rate | 1.2|1.1|1.08|1.06|1.04|1.02| -|-

# 实验结果
## duke
![image](http://note.youdao.com/yws/res/48082/WEBRESOURCEc6cff3387ac61a47ab9a43954db76391)
与baseline性能相比：![image](http://note.youdao.com/yws/res/48085/465972862D0E4002A5189CDF3EE69B59)

| | mAP | rank-1| rank-5| rank-10 | rank-20|
|--|--|--|--|--|--|
ours | 59.06 | 68.09 | 81.91 | 84.33 | 87.75 | 
baseline | 59.21 | 69.08 | 81.19 | - | 88.88|
up | -0.15 | +0. 01 | 
## mars
![image](http://note.youdao.com/yws/res/48080/WEBRESOURCE61a274bdf5832c521a1b42a9f6ee72e3)
与baseline性能相比:![image](http://note.youdao.com/yws/res/48097/7A6238B0BEC146FEAD141B8103101D1A)


| | mAP | rank-1| rank-5| rank-10 | rank-20|
|--|--|--|--|--|--|
ours | 30.39 | 54.03 | 65.51 | 71.06 | 75.1 | 
baseline | 29.56 | 52.32 | 64.29 | - | 73.08 | 
up | -0.85 | +1.71 | 

