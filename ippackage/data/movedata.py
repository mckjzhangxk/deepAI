import  os,shutil

def  move(fro,to,threshold):
    cnt=[]
    flist = os.listdir(fro)
    for t,fname in enumerate(flist):
        # if fname=='20190216092606_0.out':
        #     print('xx')
        filepath=os.path.join(fro,fname)
        cn=int(fname.replace('.out','').split('_')[1])

        if cn<threshold:
            cnt.append(cn)
            shutil.move(filepath,os.path.join(to,fname))
    print(sorted(cnt))
move('/home/zhangxk/AIProject/ippack/ip_capture/out',
     '/home/zhangxk/AIProject/ippack/ip_capture/out-2-26_18',30000)
