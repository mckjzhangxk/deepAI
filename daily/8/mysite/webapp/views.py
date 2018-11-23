#coding:utf-8

from django.shortcuts import render
from django.http import HttpResponse
import pickle
from . import Persondb as db
import os.path
import json

#database
baseweb,_=os.path.split(os.path.realpath(__file__))
path_np=os.path.join(baseweb,'faces.pickle')
#upload Image
path_db=os.path.join(baseweb,'facedb1')
#image to be detected
path_detect=os.path.join(baseweb,'detect')

if (os.path.exists(path_np)):
    with open(path_np, 'rb') as handle:
        service = pickle.load(handle)
else:
    service=db.FaceService(D=512)
    service.loaddb(path_db,path_np)

def checkface(request):
    params=request.GET
    filename=params['filename']
    name,confident=service.detectFace(os.path.join(path_detect,filename))
    retstr=json.dumps({'name':name,'confident':confident})
    return HttpResponse(retstr)
def addface(request):
    params=request.GET
    filename=params['filename']
    bl=service.addFace2db(os.path.join(path_db,filename))

    retstr='fail'
    if(bl):
        with open(path_np, 'wb') as handle:
            pickle.dump(service, handle, protocol=pickle.HIGHEST_PROTOCOL)
        retstr='ok'
    return HttpResponse(retstr)
def listfaces(request):
    retStr=json.dumps({'faceids':service.faceid})
    return HttpResponse(retStr)

def removeface(request):
    params=request.GET
    filename=params['filename']
    bl=service.deleteFile(filename)
    retstr = 'FaceNotExist'

    if bl:
        removeImagePath=os.path.join(path_db,filename)
        if os.path.exists(removeImagePath):
            os.remove(os.path.join(path_db,filename))
        with open(path_np, 'wb') as handle:
            pickle.dump(service, handle, protocol=pickle.HIGHEST_PROTOCOL)
            retstr='ok'
    return HttpResponse(retstr)
