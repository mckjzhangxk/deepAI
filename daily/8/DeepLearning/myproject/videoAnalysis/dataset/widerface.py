import os

class WinderFace():
    def __init__(self,imagepath,labelpath):
        self.imagepath=imagepath
        self.labelpath=labelpath
        self._initdb()
    def _initdb(self):
        '''
        运行完成这个方法后self.db被设在
        key=image full path
        value=[[x1,y1,w,h],[x1,y1,w,h],[x1,y1,w,h]]
        
        注意标注 是x1,x2,w,h格式
        0--Parade/0_Parade_marchingband_1_849.jpg
        1
        449 330 122 149 0 0 0 0 0 0 
        :return: 
        '''
        self.db={}
        with open(self.labelpath,'r') as fs:
            while True:
                line=fs.readline().strip()
                if line=='':break
                name=line
                imgpath=os.path.join(self.imagepath,name)
                assert os.path.exists(imgpath),'文件%s不存在'%imgpath

                line = fs.readline().strip()
                count=int(line)
                faces=[]
                for i in range(count):
                    row=fs.readline()
                    splits=row.split()[:4]
                    face=tuple(map(int,splits[0:4]))
                    faces.append(face)
                self.db[imgpath]=faces

    def summary(self):
        imgnum=len(self.db)
        facenum=0
        for v in self.db.values():
            facenum+=len(v)
        print('#%d images,#%d faces'%(imgnum,facenum))

    def toYoloDataset(self,outpath,labelname='train',chunckSize=500):
        '''
        所有的图片文件一刀outpath/labelname文件夹下面
        在outpath文件夹建立${labelname}.txt
        
        ${labelname}.txt 的每一行是
            ${labelname}/chunckid/imageid.jpg
        
        生成如下目录结构:
            outpath/labelname
                   ........../chunckid/imageid.jpg
                   ........../chunckid/imageid.txt
            outpath/labelname.txt
            
            vi ${labelname}.txt 的每一行是
                   ${labelname}/chunckid/imageid.jpg
            
        标注是 labe pcx pcy pw ph,p表示百分比
        :param outpath: 
        :param labelname: 
        :return: 
        '''

        import shutil

        target_image_dir=os.path.join(outpath, labelname)

        target_label_file= open(os.path.join(outpath,labelname+'.txt'),'w')

        dirname=os.path.basename(outpath)
        if os.path.exists(target_image_dir):
            shutil.rmtree(target_image_dir)
        os.mkdir(target_image_dir)

        import tqdm
        import PIL.Image as Image
        for i,(imagepath,faces) in enumerate(tqdm.tqdm(self.db.items())):
            img=Image.open(imagepath,'r')
            W,H=img.size

            ii=str(i//chunckSize)
            if i%chunckSize==0:
                os.mkdir(os.path.join(target_image_dir,ii))
            prefix=os.path.join(ii,str(i))
            t_img   = os.path.join(target_image_dir,prefix+'.jpg')
            t_label = os.path.join(target_image_dir,prefix+ '.txt')
            ##########################################################
            # 标注生成
            with open(t_label,'w') as anfile:
                for face in faces:
                    x1,y1,w,h=face
                    cx,cy=x1+w//2,y1+h//2
                    content = ' '.join(map(str, [0, cx/W, cy/H, w/W, h/H]))
                    anfile.write(content+'\n')

            ##########################################################
            shutil.copy(imagepath,t_img)

            target_label_file.write(os.path.join(dirname,labelname,prefix+'.jpg')+'\n')

        target_label_file.close()

if __name__ == '__main__':
    # imgpath='/home/zhangxk/下载/WIDER_train/images'
    # labelpath='examples/widerface/wider_face_train_bbx_gt.txt'
    # target='train'
    imgpath = '/home/zhangxk/AIProject/数据集与模型/WINDER_Face/WIDER_val/images'
    labelpath='examples/widerface/wider_face_val_bbx_gt.txt'
    target='test'


    winder=WinderFace(imgpath,labelpath)
    winder.summary()
    target_dir=os.path.expanduser('~/widerface')
    winder.toYoloDataset(target_dir,target)
