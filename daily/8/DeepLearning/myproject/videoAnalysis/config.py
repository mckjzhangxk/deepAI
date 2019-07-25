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
        'select_largest':True,
        'keep_all':True,
        'out_image_size':160
    }
    d['face_detector_output']='tmp/pipline2'

    with open('config.json','w') as fs:
        json.dump(d,fs,indent=1)
def loadConfig():
    with open('config.json','r') as fs:
        config=json.load(fs)
        config['yolo_img_size']=tuple(config['yolo_img_size'])

        return config
if __name__ == "__main__":
    createDefaultConfig()
    # print(loadConfig())

