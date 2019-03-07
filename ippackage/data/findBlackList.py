import os
import data.dbutils as dbutils
from data.Beans import Package_FreeGate
from data.Conf import DataConf


basePath='/home/zhangxk/AIProject/ippack/vpndata/out'
IP = '192.168.060.160' #笔记本
# IP='192.168.000.096'    #home

def sameNet(ip1,ip2):
    subnet='192.168'
    if ip1.startswith(subnet) and ip2.startswith(subnet):
        return True
    else:
        return False

def perserveIp(ip1,ip2):
    return False
def hasInBalckList(ip1,ip2):
    return ip1 in DataConf.BLACK_LIST or ip2 in DataConf.BLACK_LIST
def checkVPNConnect(key):


    sps=key.split('->')
    ip1=sps[0]
    sps=sps[1].split('_')
    ip2=sps[0]

    res=True

    if IP in ip1 or IP in ip2:
        if sameNet(ip1,ip2):
            res=False
        if perserveIp(ip1,ip2):
            res=False
        if hasInBalckList(ip1,ip2):
            res=False
    else:
        res=False
    return res

def retrive(key):

    sps=key.split('->')
    ip1,ip2_type=sps[0],sps[1]
    sps=ip2_type.split('_')
    ip2,type=sps[0],sps[1]

    if ip1.startswith(IP):
        return ip1+'--->'+ip2
    else:
        return ip2 + '--->' + ip1
if __name__ == '__main__':

    suspect=[]
    total=0

    r=dbutils.get_package_info(basePath,Package_FreeGate,True)
    total+=len(r)
    for k in r.keys():
        if checkVPNConnect(k):
            suspect.append(k)
    suspect=set(suspect)
    for x in suspect:
        print(x)
    print('total',total)