import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

class COCODataset():

    COCO_id_to_category_id = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13,
                              15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                              27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35,
                              40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46,
                              52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57,
                              63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68,
                              78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79,
                              90: 80}  # noqa
    def __init__(self,annofile):
        self.annofile=annofile
        self.categoru_id_to_COCO_id={v:k for k,v in COCODataset.COCO_id_to_category_id.items()}
        self.coco=COCO(annofile)
    def getImageList(self):
        '''
        
        :return: a list of dict<file_name,id>
        '''
        imgIds = self.coco.getImgIds()
        ret=[self.coco.loadImgs(imgid)[0] for imgid in imgIds]
        return ret
    def classId2COCOId(self,id):
        return self.categoru_id_to_COCO_id[id+1]
    def print_coco_metrics(self, json_file):
        """
        Args:
            json_file (str): gt_json_file to the results json file in coco format
        Returns:
            dict: the evaluation metrics
        """

        ret = {}
        cocoDt = self.coco.loadRes(json_file)
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
        for k in range(6):
            ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

        json_obj = json.load(open(json_file))
        if len(json_obj) > 0 and 'segmentation' in json_obj[0]:
            cocoEval = COCOeval(self.coco, cocoDt, 'segm')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            for k in range(6):
                ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
        return ret
