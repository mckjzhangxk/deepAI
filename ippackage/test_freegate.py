from ML import get_dataset,run_analysis
from data.Beans import Package_FreeGate


paths=[
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_1',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_2',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_3',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_4',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_5',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_6',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_7',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_8',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_9',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_10',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_11',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_12',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_13',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_14',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_15',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_16',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_17',
    '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_18'
]
features=['upcount','up_rate','downcount','downsize','down_rate']
for path in paths:
    db,np_db=get_dataset(path,features,False,False,windowSize=100,beanFunc=Package_FreeGate)
    rs = run_analysis(np_db, db, window=51, threshold=0.3)
    print(path,'\n',rs)
    print('-----------------------------------------------------------------')
