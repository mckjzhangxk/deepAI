from pycocotools.coco import COCO
from tqdm import tqdm
import os



# coco=COCO('instances_val2017.json')
# catIds=coco.getCatIds()
# catIds=sorted(catIds)
# label_dict={cid:i for i,cid in enumerate(catIds)}
#
# imgIds=coco.getImgIds()
# imgId=imgIds[0]
#
# imginfo=coco.loadImgs(imgId)[0]
# annIds=coco.getAnnIds(imgIds=imgId,catIds=catIds)
#
# aa=parseAnn(imginfo,annIds,label_dict)
# print(aa)

MIN_SIZE=20
IMG_SIZE=416

def coco_dataset(cocopath,output_path,prefix=''):
    coco = COCO(cocopath)

    def parseAnn(imginfo, annIds, label_dict):
        anns = coco.loadAnns(annIds)

        ret = []
        for ann in anns:
            rs, x1, y1, x2, y2 = chioce_box(ann['bbox'], imginfo['width'], imginfo['height'])
            if not rs: continue
            label = label_dict[ann['category_id']]
            ret.extend([x1, y1, x2, y2, label])
        return ret

    def chioce_box(bbox, W, H):
        x1, y1, w, h = bbox
        if (w * IMG_SIZE / W < MIN_SIZE) or (h * IMG_SIZE / H < MIN_SIZE):
            return (False, 0, 0, 0, 0)
        x2, y2 = x1 + w, y1 + h
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        return (True, x1, y1, x2, y2)

    catIds = coco.getCatIds()
    catIds = sorted(catIds)
    label_dict = {cid: i for i, cid in enumerate(catIds)}
    imgIds = coco.getImgIds()

    with open(output_path,'w') as fs:
        for imgid in tqdm(imgIds):
            img_info=coco.loadImgs(imgid)[0]
            annIds=coco.getAnnIds(imgIds=imgid,catIds=catIds)

            annoinfo=parseAnn(img_info,annIds,label_dict)
            if len(annoinfo)==0:continue
            filename=os.path.join(prefix,img_info['file_name'])
            line_info=[filename]+[str(s) for s in annoinfo]
            fs.write(' '.join(line_info)+'\n')

if __name__ == '__main__':
    in_path='instances_val2017.json'
    out_path='r.txt'
    prefix='/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/yolo3'
    coco_dataset(in_path,out_path,prefix)