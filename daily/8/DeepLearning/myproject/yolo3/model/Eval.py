from model.CocoData import COCODataset
from model.YoloNet import YoLoService
import os
import json

if __name__ == '__main__':
    gt_json_file= '../data/instances_val2017.json'
    predict_json_file = ''
    basepath=''
    model_path='weights/yolov3.ckpt'


    coco=COCODataset(gt_json_file)
    imagelist=coco.getImageList()#image_id,file_name
    filelist=[os.path.join(basepath,x['file_name']) for x in imagelist]
    service=YoLoService(model_path)
    result=service.predict_imagelist(filelist)

    ret=[]
    for predict,imageinfo in zip(result,imagelist):
        _id=imageinfo['image_id']

        for box,score,label in zip(predict['boxes'],predict['scores'],predict['labels']):
            _category_id=coco.classId2COCOId(label)
            box[2] -= box[0]
            box[3] -= box[1]

            _bbox=[round(float(x),3) for x in box]
            _score=round(float(score),3)
            ret.append({'image_id':_id,'category_id':_category_id,'bbox':_bbox,'score':_score})
    with open(predict_json_file, 'w') as f:
        json.dump(ret, f)

    coco.print_coco_metrics(predict_json_file)