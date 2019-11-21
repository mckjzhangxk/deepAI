import mxnet.recordio as io
import numpy as np
import cv2
import os




def extract2Output(prefix,database_output,search_output,samples=5):
    from pathlib import Path

    p=Path(database_output)
    if p.exists():
        import shutil
        shutil.rmtree(database_output)
        p.mkdir()
    p=Path(search_output)
    if p.exists():
        import shutil
        shutil.rmtree(search_output)
        p.mkdir()


    reader=io.MXIndexedRecordIO(prefix+'.idx',prefix+'.rec','r')

    #第0行是全部种类的信息，获得全部种类的索引
    s=reader.read_idx(0)
    header,_=io.unpack(s)
    labels=range(int(header.label[0]),int(header.label[1]))
    ###############获得种类下实例的索引，imgs保存的是某一个种类下的实例索引####
    imgs=[]
    for l in labels:
        s=reader.read_idx(int(l))
        header,_=io.unpack(s)
        a,b=int(header.label[0]),int(header.label[1])
        imgs.append(range(a,b))


    ##########extract feature of every image##############
    import tqdm


    for ii,imgidxs in tqdm.tqdm(enumerate(imgs)):
        sc_path=os.path.join(search_output,str(ii))
        db_path=os.path.join(database_output,str(ii))
        os.mkdir(sc_path)
        os.mkdir(db_path)

        imgcount=len(list(imgidxs))
        smidx=[]
        if samples>0:
            smidx=np.random.choice(list(imgidxs),samples,False)

        for id in imgidxs:
            s=reader.read_idx(id)
            h,img=io.unpack_img(s)
            if id not in smidx:
                o=os.path.join(db_path,str(id)+'.jpg')
                cv2.imwrite(o,img)
            else:
                o = os.path.join(sc_path, str(id) + '.jpg')
                cv2.imwrite(o, img)

if __name__ == '__main__':
    db='/home/zxk/AI/data/faces_umd/db'
    search='/home/zxk/AI/data/faces_umd/sample'
    bin='/home/zxk/AI/data/faces_umd/train'
    extract2Output(bin,db,search,5)