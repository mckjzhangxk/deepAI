#coding:utf-8
from django.shortcuts import render
from django.http import HttpResponse
from .utils import abc
import json
def getCoord(request):
    postBody = request.body.decode(encoding='utf-8')

    data=json.loads(postBody, encoding='utf-8')
    w,h,offset=int(request.GET['width']),int(request.GET['height']),int(request.GET['offset'])

    result=abc(data,seed=0,width=w,height=h,offset=offset)

    return HttpResponse(json.dumps(result))
