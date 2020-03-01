#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:markdown_figure_handel.py
@TIME:2020/3/1 12:58
@DES:
'''

import json
import re
import requests
def converUrl(url):
    reg = r'id=(.*?)&'
    s = re.search(reg, url)
    if s is not None:
        url = "http://note.youdao.com/yws/public/note/%s?editorType=0&cstk=orBX-yw0" % s.group()[3:-1]
        return url
    else:
        return None
def getHtml(url):
    r =requests.get(url=url)
    #print(r.status_code)
    #print(r.text)
    return r.text;
def getImageUrls(url):
    html =getHtml(url)
    try:
        js = json.loads(html)
        ss = js["content"]
    except Exception:
        return None
    reg = r'src="(.*?)"'
    pattern = re.compile(reg)
    ret = re.findall(pattern, ss)
    return ret

if __name__=="__main__":
    url = raw_input('url:').strip()
    # url = "http://note.youdao.com/noteshare?id=3f12b7cb6d8928ed43b30d4495aacc4c&sub=A6D2DC88F50E4A6B8D0FD78BA5A9BFBC"
    #        http://note.youdao.com/noteshare?id=0b969242ea3c48fbaaa18f9f9d222f23
    url = converUrl(url)
    print(url)
    print(getImageUrls(url))


