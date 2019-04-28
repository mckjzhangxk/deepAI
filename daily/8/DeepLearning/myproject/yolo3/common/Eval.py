import json
import os

from data.CocoData import COCODataset

from model.yolo.YoloNet import YoLoService
from model.frcnn.Models import FRCnnService

def doEval(modelName,model_path):
    if modelName=='yolo':
        service=YoLoService(model_path=model_path)
    if modelName=='frcnn':
        service=FRCnnService(model_path=model_path)

    result=service.predict_imagelist(filelist,batchSize=32)

    return result
if __name__ == '__main__':
    gt_json_file = '/home/zxk/AI/coco/annotations/instances_val2017.json'
    # predict_json_file = '/home/zxk/AI/coco/bencemark/yolo_result.json'
    predict_json_file = '/home/zxk/AI/coco/bencemark/frcnnR50C4_result.json'

    basepath = '/home/zxk/AI/coco/val2017'
    # model_path = '/home/zxk/AI/tensorflow-yolov3/checkpoint/yolov3.ckpt'
    model_path = '/home/zxk/AI/tensorpack/FRCNN/COCO-R50C4-MaskRCNN-Standard.npz'

    # model_name='yolo'
    model_name='frcnn'

    coco=COCODataset(gt_json_file)
    imagelist=coco.getImageList()#image_id,file_name
    imagelist=imagelist[:10]
    filelist=[os.path.join(basepath,x['file_name']) for x in imagelist]
    result = doEval(model_name, model_path)

    ret=[]
    for predict,imageinfo in zip(result,imagelist):
        _id=imageinfo['id']

        for box,score,label in zip(predict['boxes'],predict['scores'],predict['labels']):
            if box is None:continue
            _category_id=coco.classId2COCOId(label)
            box[2] -= box[0]
            box[3] -= box[1]

            _bbox=[round(float(x),3) for x in box]
            _score=round(float(score),3)
            ret.append({'image_id':_id,'category_id':_category_id,'bbox':_bbox,'score':_score})
    with open(predict_json_file, 'w') as f:
        json.dump(ret, f)

    coco.print_coco_metrics(predict_json_file)