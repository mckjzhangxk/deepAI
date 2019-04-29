import json
from pycocotools.coco import COCO
from collections import namedtuple
import cv2
from pycocotools.cocoeval import COCOeval
from tensorpack.utils import viz


class PredictResult(namedtuple('_PredictResult',['score','bbox','label'])):pass

class Result_JSON():
    def __init__(self,path,prefix,cocodb):
        self.cocodb=cocodb
        self.prefix=prefix

        with open(path) as fp:
            objs=json.load(fp)
        self._d={}
        for obj in objs:
            score=obj['score']
            category_id=obj['category_id']
            image_id=obj['image_id']
            bbox=obj['bbox']

            if image_id in self._d:
                self._d[image_id].score.append(score)
                self._d[image_id].bbox.append(bbox)
                self._d[image_id].label.append(category_id)
            else:
                self._d[image_id]=PredictResult([score],[bbox],[category_id])
    def __len__(self):
        return len(self._d)
    def __getitem__(self, imageid):
        filename=self.cocodb.lookup[imageid]['file_name']
        I=cv2.imread(self.prefix+'/'+filename)

        obj=self._d[imageid]


        labels=['%d:%.2f'%(ll,ss) for ll,ss in zip(obj.label,obj.score)]
        print(obj.bbox)
        boxes=[[bb[0],bb[1],bb[0]+bb[2],bb[1]+bb[3]] for bb in obj.bbox]
        I = viz.draw_boxes(I,boxes,labels)
        print('total # boxes:%d'%len(boxes))
        return I
    def getImageIds(self):
        return [key for key in self._d.keys()]
class COCODB():
    def __init__(self,annofile):
        self.coco = COCO(annofile)
        imgIds = self.coco.getImgIds()
        self.lookup = {imgid:self.coco.loadImgs(imgid)[0] for imgid in imgIds}
    def doEval(self,json_file):
        cocoDt = self.coco.loadRes(json_file)
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
