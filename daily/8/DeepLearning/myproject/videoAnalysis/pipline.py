from yolov3 import CCPD_YOLO_Detector
from time import sleep
import glob
from utils import readVideo
import os
import datetime
import uuid
import json
import shutil

from config import loadConfig

class ObjDetectionService():
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

    def start(self):
        while True:
            filelist=glob.glob(self.inputpath)
            if len(filelist)>0:
                for filename in filelist:
                    self._process(filename)
            sleep(self.refresh_interval)
    def _defaultReturn(self,id):
        now=datetime.datetime.now()
        final={
            'id':id,
            'status':'success',
            'timestamp':{
                'ts':[],
                'objs':[]
            },
            'time':str(now.strftime("%Y-%m-%d %H:%M:%S"))
        }
        return final
    
    @staticmethod
    def getId(filename):
        bs=os.path.basename(filename)
        ii=bs.index('.')
        return bs[:ii]

    def _after_precessing(self,final,videofile):
        '''
        
        '''
        id=self.getId(videofile)
        with open(os.path.join(self.outputpath,id+'.json'),'w') as fs:
            json.dump(final,fs,indent=1)
        if os.path.exists(os.path.join(self.outputpath,os.path.basename(videofile))):
            os.remove(os.path.join(self.outputpath,os.path.basename(videofile)))
        shutil.move(videofile,self.outputpath)

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
        final=self._defaultReturn(self.getId(videofile))
        try:
            stream,videoinfo=readVideo(videofile)

            frames=videoinfo['frames']
            for t in range(frames):
                retval, frame = stream.read()
                if not retval: break
                
                if t % self.skip==0:
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
                        final['timestamp']['ts'].append(t)                    
                        final['timestamp']['objs'].append(objs_at_t)

            print('任务1:%s完成'%videofile)
        except Exception as e:
            final['status']='fail'
            final['error']='At job1,'+str(e)
            if self.config['debug']:
                raise e
        finally:
            if stream is not None:
                stream.release()
        self._after_precessing(final,videofile)
if __name__ == "__main__":
    config=loadConfig()
    service=ObjDetectionService(config)
    service.start()