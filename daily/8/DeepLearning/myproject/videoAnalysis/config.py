from collections import OrderedDict
import json
def createDefaultConfig():
    d=OrderedDict()
    
    d['debug']=True
    d['device']='cpu'
    d['skipRate']=10
    #for object detection
    d['job1_input']='tmp/input/input_job1/*.avi'
    
    d['yolo_conf']='yolov3/cfg/yolov-obj.cfg'
    d['yolo_weight']='yolov3/cfg/yolov-obj_final.weights'
    d['yolo_objs_name']='yolov3/cfg/coco.names'
    d['yolo_img_size']=(416,416)
    d['yolo_refresh_interval']=10
    d['job1_output']='tmp/input/input_job2/'

    #for face detection
    d['face_detector']='mtcnn'

    d['mtcnn']={
        'min_face_size':20,
        'thresholds':[0.6, 0.7, 0.7],
        'factor':0.709,
        'prewhiten':True,
        'out_image_size':160
    }
    d['job2_output']='tmp/input/input_job3/'

    #for face align
    d['face_align_margin']=40
    d['face_align_type']=0


    #for face identify
    d['face_iditify']='facenet'
    d['facenet_pretained_model']='casia-webface'
    d['facenet_image_size']=160
    d['facenet_batch_size']=4


    d['arcface_arch']='resnet' #或者mobilefacenet
    d['arcface_modelpath']='arcface_pytorch/weights/model_ir_se50.pth'
    d['arcface_net_depth'] = 50
    d['arcface_drop_ratio']=0.6
    d['arcface_net_mode']='ir_se'
    d['arcface_image_size']=112


    d['job3_output']='tmp/input/final'
    with open('config.json','w') as fs:
        json.dump(d,fs,indent=1)
def loadConfig():
    import os

    path=os.path.join((os.path.dirname(__file__)),'config.json')

    with open(path,'r') as fs:
        config=json.load(fs)
        config['yolo_img_size']=tuple(config['yolo_img_size'])

        return config
if __name__ == "__main__":
    createDefaultConfig()
    # print(loadConfig())

