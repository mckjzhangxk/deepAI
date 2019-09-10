import os
def evaluate(labelfile, baseoutputpath,imagepath=None,detector=None):
    '''
    使用一个detector对标记文件做预测，输出到baseoutputpath
    
    detector:predict(list of cv2 image),return list of tuple5 (x1,y1,x2,y2)
    
    :param labelfile: 
    :param baseoutputpath: 
    :param imagepath: 
    :param detector: 
    :return: 
    '''

    progress=0
    with open(labelfile, 'r') as fs:
        while True:
            line = fs.readline().strip()
            if line == '': break
            name = line
            dirname,_=name.split('/')

            if not os.path.exists(os.path.join(baseoutputpath, dirname)):
                os.mkdir(os.path.join(baseoutputpath, dirname))

            line = fs.readline().strip()
            count = int(line)
            for i in range(count): fs.readline()

            #####################图片检测#################################
            import cv2
            # inputbatch=[cv2.imread(os.path.join(imagepath,name))]
            # result=detector.predict(inputbatch)[0]
            import PIL.Image
            result = detector.detect_faces(PIL.Image.open(os.path.join(imagepath,name)))[0]
            cnt=len(result)
            #############################################################
            with open(os.path.join(baseoutputpath, name[:-3] + 'txt'), 'w') as ff:
                ff.write(name+'\n')
                ff.write(str(cnt) + '\n')
                for r in result:
                    face=[int(r[0]),int(r[1]),int(r[2]-r[0]),int(r[3]-r[1]),float(1)]
                    ff.write(' '.join(map(str, face)) + '\n')
            progress+=1
            if progress%10==0:
                print(progress)

def abc(labelpath,basepath='examples/test'):
    with open(labelpath, 'r') as fs:
        while True:
            line = fs.readline().strip()
            if line == '': break

            name = line
            dirname,xx=name.split('/')
            if not os.path.exists(os.path.join(basepath,dirname)):
                os.mkdir(os.path.join(basepath,dirname))

            with open(os.path.join(basepath,name[:-3]+'txt'),'w') as ff:
                ff.write(name+'\n')
                line = fs.readline().strip()
                ff.write(line+'\n')
                count = int(line)

                for i in range(count):
                    row = fs.readline()
                    splits = row.split()[:4]
                    face = list(map(int, splits[0:4]))+[1.0]
                    ff.write(' '.join(map(str,face))+'\n')
                    print(' '.join(map(str,face)))
# abc('examples/widerface/wider_face_val_bbx_gt.txt')

if __name__ == '__main__':
    from  yolov3 import  CCPD_YOLO_Detector
    cfg='../yolov3/cfg/widerface/widerface.cfg'
    weight='../yolov3/cfg/widerface/widerface_last.weights'
    # cfg=None
    # weight=None
    device='cuda'
    imgpath='/home/zxk/AI/WIDER_val/images'

    # detector = CCPD_YOLO_Detector(cfg=cfg, weight=weight, device=device,half=False)
    from  RetinaFace import RetinaFaceDetector

    detector = RetinaFaceDetector('../RetinaFace/model/R50',
                                  device=device,
                                  scale=(1024,1980),
                                  thresh=0.8)
    evaluate('examples/widerface/wider_face_val_bbx_gt.txt','/home/zxk/下载/eval_tools/eval_tools/pred_retina',imgpath,detector)

