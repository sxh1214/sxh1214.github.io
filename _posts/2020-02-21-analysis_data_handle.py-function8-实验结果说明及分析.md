---
layout:     post
title:      analysis_data_handle.py-function8-实验结果说明及分析
subtitle:   方差实验组
date:       2020-02-21
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - 科研之路
---

>实验结果下载链接：http://note.youdao.com/noteshare?id=8299b9cd136b1c576b0e002a1f22720d


## 结果说明
因为实验数据复杂，请实验过程中代码变动很大，所有图例所述不尽详实。
在这里做具体的实验说明。


![image](http://note.youdao.com/yws/res/47501/EE5FD11F21C046CEB30802224934729C)

如上图，[0/xxx]表示采样数据范围。
model_0表示仅根据one-shot部分带标注样本训练后得到的模型。model_1表示加入了model采样出来的135个伪标签样本后的训练集训练出来的模型，依次类推。

整个过程模拟了在实际训练中的 渐进采样过程，总的step数量为11.

以下图为例。

![image](http://note.youdao.com/yws/res/47509/60E77C67D0A246F4873EC687F1E4090B)

- 标题的区间指示采样区间，model_x表示是第几个model的标签估计和采样结果。
- 横坐标的按照欧几里得距离排好序的样本。
- 左侧纵坐标是FN_acc，表示前N个样本的标签估计准确率，对应着图中的蓝色曲线。
- 右侧纵坐标是variance（方差），对应着图中的橙色曲线
- 绿色的散点指示各个样本的伪标签正确与否。若估计正确，则值为0.2，否则值为0.
- 红色的散点指示入选的伪标签样本，若该样本入选，则红点值=其方差值，否则值为0.3。


下图是终端运行结果。
![image](http://note.youdao.com/yws/res/47527/WEBRESOURCE5109f10def48e0669f1b084505edc4de)
- acc1 表示仅适用距离排序的情况下，筛选出来的伪标签样本正确个数/正确率。
- acc2 表示在使用距离排序的情况下，再从头到尾进行方差排序，筛选出来的伪标签样本正确个数/正确率。
- acc3 表示在使用距离排序的情况下，再从查询点（query_point）到尾进行方差排序，筛选出来的伪标签样本正确个数/正确率。


## 结果分析

方差虽然和欧几里得距离一样，总体上与标签估计的准确率存在线性关系，但在使用方差采样的时候，并没有很好的将错误的标签挑选出来。







