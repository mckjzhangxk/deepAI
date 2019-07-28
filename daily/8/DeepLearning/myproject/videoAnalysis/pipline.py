from abc import abstractmethod
from yolov3 import CCPD_YOLO_Detector
from facenet_pytorch import MTCNN,InceptionResnetV1
from time import sleep
import glob
from utils import readVideo,ioa,videoWriter,prewhiten
import os
import datetime
import uuid
import json
import shutil
import cv2
import numpy as np
import torch
from config import loadConfig
from tqdm import tqdm
def getId(filename):
    bs=os.path.basename(filename)
    ii=bs.index('.')
    return bs[:ii]
class BaseService():
    def start(self):
        while True:
            filelist=glob.glob(self.inputpath)
            if len(filelist)>0:
                for filename in filelist:
                    self._process(filename)
            sleep(self.refresh_interval)
    @abstractmethod
    def _process(self,filename):
        raise  NotImplemented
    def _defaultReturn(self,filename):
        with open(filename) as fs:
            upcoming=json.load(fs)
            return upcoming
class ObjDetectionService(BaseService):
    def __init__(self,config):
        self.det=CCPD_YOLO_Detector(
            cfg=config['yolo_conf'],
            weight=config['yolo_weight'],
            img_size=config['yolo_img_size'],
            device=config['device']
        )
        self.refresh_interval=config['yolo_refresh_interval']
        self.inputpath=config['job1_input']
        self.skip=config['skipRate']
        self.config=config
        self.outputpath=config['job1_output']
        with open(config['yolo_objs_name'],'r') as fs:
            self.cls=[x.strip() for x in fs]

    def _defaultReturn(self,id):
        now=datetime.datetime.now()
        final={
            'id':id,
            'status':'success',
            'track':{
                'ts':[],
                'objs':[]
            },
            'time':str(now.strftime("%Y-%m-%d %H:%M:%S"))
        }
        return final
    

    def _after_precessing(self,final,videofile):
        '''
        
        '''
        id=getId(videofile)
        with open(os.path.join(self.outputpath,id+'.json'),'w') as fs:
            json.dump(final,fs,indent=1)
        # if os.path.exists(os.path.join(self.outputpath,os.path.basename(videofile))):
        #     os.remove(os.path.join(self.outputpath,os.path.basename(videofile)))
        os.remove(videofile)

    def _relation_between_object(self,objs_at_t):
        pass

    def _process(self,videofile):
        '''
            videofile:要处理的视频文件

            这里对类型进行检测
            把 目标类型,时间点,坐标,以及附属关系

            输出到相应的 目录一个json文件
        '''
        print('开始任务1:%s'%videofile)

        stream=None
        writer=None
        final=self._defaultReturn(getId(videofile))
        try:
            stream,videoinfo=readVideo(videofile)
            outputfile=os.path.join(self.outputpath,os.path.basename(videofile))
            writer=videoWriter(outputfile,
                        scale=(videoinfo['width'],videoinfo['height']),
                        fps=30//self.skip)
            final['job2_video']=outputfile

            frames=videoinfo['frames']
            for t in tqdm(range(0,frames,self.skip)):
                stream.set(1,t)
                retval, frame = stream.read()
                if not retval: break
                
                # if t % self.skip==0:
                result = self.det.predict(frame)
                objs_at_t=[]
                for x1, y1, x2, y2, conf, label in result:
                    obj={
                            'id':str(uuid.uuid1()).replace('-',''),
                            'type':self.cls[label],
                            'confident':conf,
                            'box':[int(x1),int(y1),int(x2),int(y2)]
                        }
                    objs_at_t.append(obj)
                    ####################计算object 之间的关系##############################
                self._relation_between_object(objs_at_t)

                # ###############################################
                if len(objs_at_t)>0:
                    final['track']['ts'].append(t)
                    final['track']['objs'].append(objs_at_t)
                    writer.write(frame)
            print('任务1:%s完成'%videofile)
        except Exception as e:
            final['status']='fail'
            final['error']='At job1,'+str(e)
            if self.config['debug']:
                raise e
        finally:
            if stream is not None:
                stream.release()
            if writer is not None:
                writer.release()

        self._after_precessing(final,videofile)

class FaceDetectionService(BaseService):

    def __init__(self,config):
        device=config['device']
        self.det=MTCNN(
            image_size=config['mtcnn']['out_image_size'],
            margin=0,
            min_face_size=config['mtcnn']['min_face_size'],
            thresholds=config['mtcnn']['thresholds'],
            factor=config['mtcnn']['factor'],
            prewhiten=False,
            select_largest=True,#True,boxes按照面积的大小降序排列
            keep_all=True,
            device=device
        )
        self.refresh_interval=config['yolo_refresh_interval']
        self.inputpath=os.path.join(config['job1_output'],'*.json')
        self.config=config
        self.outputpath=config['job2_output']
        print('完成了人脸检测的初始化')
    def _insertFace(self,objs,faceboxes):
        if faceboxes is None or len(faceboxes)==0:
            return
        indexes=[]
        personboxes=[]

        for ii,obj in enumerate(objs):
            if obj['type']=='person':
                indexes.append(ii)
                personboxes.append(obj['box'])
        if len(personboxes)>0:
            face_person=ioa(faceboxes,personboxes).argmax(axis=1)
            for faceindex,person_index in enumerate(face_person):
                obj_index=indexes[person_index]
                objs[obj_index]['face_box']=list(faceboxes[faceindex])
        else:
            for face in faceboxes:
                newperson = {
                    'id': str(uuid.uuid1()).replace('-', ''),
                    'type': 'person',
                    'confident': 0.0,
                    'box': list(face),
                    'face_box': list(face)
                }
                objs.append(newperson)

    def _process(self,filename):
        '''
            videofile:要处理的视频文件

            这里对类型进行检测
            把 目标类型,时间点,坐标,以及附属关系

            输出到相应的 目录一个json文件
        '''
        id=getId(filename)
        print('开始任务2:%s'%filename)

        stream=None
        final=self._defaultReturn(filename)
        try:
            if final['status']=='success':
                videofile=final['job2_video']
                stream,videoinfo=readVideo(videofile)
                frames=videoinfo['frames']

                for t in tqdm(range(frames)):
                    retval, frame = stream.read()
                    if not retval: break
                    _,faceboxes = self.det(frame)
                    objs_at_t=final['track']['objs'][t]
                    self._insertFace(objs_at_t,faceboxes)
                self._after_precessing(final,videofile,filename)
                print('任务2:%s完成'%videofile)
        except Exception as e:
            final['status']='fail'
            final['error']='At job2,'+str(e)
            if self.config['debug']:
                raise e
        finally:
            if stream is not None:
                stream.release()

    def _after_precessing(self,final,videofile,jobfile):
        '''
        
        '''
        if os.path.exists(jobfile):
            os.remove(jobfile)

        outputpath=os.path.join(self.outputpath,os.path.basename(videofile))
        if os.path.exists(outputpath):
            os.remove(outputpath)
        shutil.move(videofile,outputpath)
        id=getId(videofile)
        

        final['job3_video']=outputpath
        with open(os.path.join(self.outputpath,id+'.json'),'w') as fs:
            json.dump(final,fs,indent=1)

class FaceFeatureService(BaseService):
    def __init__(self,config):
        self.device= config['device']
        self.refresh_interval = config['yolo_refresh_interval']
        self.inputpath = os.path.join(config['job2_output'], '*.json')

        self.config = config
        self.outputpath = config['job3_output']
        self.batchsize=config['facenet_batch_size']
        self.imagesize=config['facenet_image_size']

        self.model = InceptionResnetV1(pretrained=config['facenet_pretained_model']).to(self.device).eval()
        print('完成了人脸特征提取的初始化')

    def _get_input(self,frame, obj):
        x1,y1,x2,y2=obj['face_box']
        x1=max(x1,0)
        x2 = max(x2, 0)
        y1 = max(y1, 0)
        y2 = max(y2, 0)

        I=frame[y1:y2,x1:x2]
        I=I[:,:,::-1]
        I=cv2.resize(I,(self.imagesize,self.imagesize))
        I=np.transpose(I,(2,0,1))

        I=prewhiten(I)

        return (obj,I)
    def _handle(self,queue):
        Iin=torch.stack([img for obj,img in queue],0).to(self.device)
        faceids=self.model(Iin).cpu().data.numpy().tolist()
        for (obj,img),faceid in zip(queue,faceids):
            del img
            obj['face_id']=faceid

    def _after_precessing(self, final, videofile, jobfile):
        '''

        '''
        if os.path.exists(jobfile):
            os.remove(jobfile)

        outputpath = os.path.join(self.outputpath, os.path.basename(videofile))
        if os.path.exists(outputpath):
            os.remove(outputpath)
        shutil.move(videofile, outputpath)
        id = getId(videofile)

        # final['job3_video='] = outputpath
        with open(os.path.join(self.outputpath, id + '.json'), 'w') as fs:
            json.dump(final, fs, indent=1)

    def _process(self,filename):
        print('开始任务3:%s'%filename)

        stream=None
        final=self._defaultReturn(filename)
        queue=[]
        try:
            if final['status']=='success':
                videofile=final['job3_video']
                stream,videoinfo=readVideo(videofile)
                frames=videoinfo['frames']

                for t in tqdm(range(frames)):
                    retval, frame = stream.read()
                    if not retval: break
                    objs_at_t=final['track']['objs'][t]

                    for obj in objs_at_t:
                        if 'face_box' in obj:
                            queue.append(self._get_input(frame,obj))
                            if len(queue)==self.batchsize:
                                self._handle(queue)
                                del queue
                                queue=[]
                if len(queue)>0:
                    self._handle(queue)
                self._after_precessing(final,videofile,filename)
                print('任务3:%s完成'%videofile)
        except Exception as e:
            final['status']='fail'
            final['error']='At job3,'+str(e)
            if self.config['debug']:
                raise e
        finally:
            if stream is not None:
                stream.release()
        print('任务3:%s完成' % filename)

from threading import Thread,currentThread
class MyWorker(Thread):
    def __init__(self,service):
        super().__init__()
        self.service=service
    def run(self):
        self.service.start()

if __name__ == "__main__":
    threads=[]

    config=loadConfig()
    service1=ObjDetectionService(config)
    threads.append(MyWorker(service1))

    service2=FaceDetectionService(config)
    threads.append(MyWorker(service2))

    service3=FaceFeatureService(config)
    threads.append(MyWorker(service3))

    for t in threads:
        t.start()
    for t in threads:
        t.join()




