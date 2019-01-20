import cv2
import numpy as np
from detect.detect_face import generateBoundingBox,imresample,nms,bbreg,rerec,validbox,normalize,feedImage,drawDectectBox
import tensorflow as tf
from model import createPNet,createRNet
from scipy.misc import imsave

class Detector():
    '''
    
    threshold:用于三个网络的"通行"概率
    nms:[0,1]用于pnet的nms,一次是对一张scale的nms,一张是对所有scale处理后的nms
    model_path:三个模型的路径
    '''
    def __init__(self,
                 sess=None,
                 minsize=50,
                 scaleFactor=0.7,
                 nms=[0.5,0.7,None,None],
                 threshold=[0.6,0.3,None],
                 model_path=[None,None,None],
                 save_stage=False):
        self.m_session=sess
        self.m_minsize=minsize
        self.m_scalefactor = scaleFactor
        self.m_nms = nms
        self.m_threshold = threshold
        self.m_model_path=model_path
        self.m_save_stage=save_stage
        self.pnet=None
        self.rnet = None
        self.onet = None
        self.load()
    '''
        这个方法定义并加载了
        self.m_pnet,m_rnet,m_onet三个网络,
        分别从self.m_model_path加载
    '''
    def load(self):
        raise NotImplemented
    def preprocessing(self,img):
        raise NotImplemented
    '''
        对于:m_save_stage=False:
            给一个图片,文件名,返回MTCNN的检测结果
            total_box:shape[N',9],x1,y1,x2,y2,score,rx1,ry1,rx2,ry2
            如果没有检测出,返回[0,9]
        对于:m_save_stage=True:
            返回:[total_box]*3....
    '''
    def detect_face(self,filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        assert image is not None, '%s not exist' % filename
        total_box=np.empty((0,9),dtype=np.float32)
        stage_box=[]
        if self.pnet:
            total_box=self.p_stage(image)
            if self.m_save_stage:
                stage_box.append(total_box.copy())
        if self.rnet and len(total_box)>0:
            total_box = self.r_stage(image,total_box)
            if self.m_save_stage:
                stage_box.append(total_box.copy())
        if self.rnet and len(total_box) > 0:
            total_box = self.o_stage(image,total_box)
            if self.m_save_stage:
                stage_box.append(total_box.copy())
        if self.m_save_stage:
            return stage_box
        else:
            return total_box
    '''
        p_stage返回
        total_box:shape[N',5],x1,y1,x2,y2,score
        total_box是经过
        bbreg->rrect->valid->pad处理后的!
        
        pnet有2次调用NMS,参数分别是m_nms的[0],[1]
    '''
    def p_stage(self,img):
        h, w, _ = img.shape
        l = min(h, w)

        minsize=self.m_minsize
        factor=self.m_scalefactor
        t=self.m_threshold[0]
        nms1,nms2=self.m_nms[0],self.m_nms[1]
        pnet=self.pnet



        '''
        minsize is min size of a face,so scale origin image to by factor 12/minsize,
        then pass it to a pnet,as least get a output(1x1x4,1x1,2),because pnet is a
        detector with window size =12
        '''
        scale = 12 / minsize
        scales = []
        while int(l * scale) >= 12:
            scales.append(scale)
            scale *= factor
        total_boxes = np.empty((0, 9))


        # 不同的scale经过筛选后的total_box没有经过修正
        for i, scale in enumerate(scales):
            hs, ws = int(np.ceil(scale * h)), int(np.ceil(scale * w))

            im_data = imresample(img, (hs, ws))
            im_data = self.preprocessing(im_data)
            out = pnet(im_data)
            # out[0]是regressor,0表示第一张图片,shape[H',W',4]
            regressor = out[0][0]
            # out[1]是prosibility,out[1][0]是第一张图片,shape[H',W',2]
            score = out[1][0][:, :, 1]
            # bbox是原图的坐标,(N',9),N'是概率>t保留下来的人脸数量
            bbox, _ = generateBoundingBox(score.copy(), regressor.copy(), scale, t)
            if bbox.size > 1:
                pick = nms(bbox.copy(), nms1, 'Union')
                bbox = bbox[pick]
                total_boxes = np.append(total_boxes, bbox, axis=0)
        #不同的scale图片处理筛选外成,要经过
        #reg修正->正方形化->有效性验证
        if len(total_boxes) > 1:
            pick = nms(total_boxes.copy(), nms1, 'Union')
            total_boxes = total_boxes[pick]
            '''
            using regress to adjust
            '''
            total_boxes = bbreg(total_boxes, total_boxes[:, 5:9])
            total_boxes = rerec(total_boxes.copy())
            total_boxes=validbox(total_boxes)
        return total_boxes[:,:5]

    def r_stage(self,image,total_box):pass
    def o_stage(self):pass
class Detector_Caffe(Detector):pass
class Detector_tf(Detector):
    def load(self):
        sess=self.m_session
        p_path,r_path,o_path=self.m_model_path[0],None,None
        if len(self.m_model_path)>=2:
            r_path=self.m_model_path[1]
        if len(self.m_model_path) == 3:
            o_path = self.m_model_path[2]


        if p_path:
            data_p = tf.placeholder(tf.float32, (None,None,None,3))
            prob_tensor_p, regbox_tensor_p=createPNet(data_p, True)
            self.pnet=lambda img: sess.run((regbox_tensor_p,prob_tensor_p), feed_dict={data_p: img})

            varlist=tf.get_collection('PNet')
            saver = tf.train.Saver(var_list=varlist)
            saver.restore(sess, p_path)

        if r_path:
            data_r = tf.placeholder(tf.float32, (None, 24, 24, 3))
            prob_tensor_r, regbox_tensor_r =createRNet(data_r, True)
            self.rnet = lambda img: sess.run((regbox_tensor_r, prob_tensor_r), feed_dict={data_r: img})

            varlist=tf.get_collection('RNet')
            saver = tf.train.Saver(var_list=varlist)
            saver.restore(sess, r_path)

    def preprocessing(self,img):
        img=normalize(img)
        img=feedImage(img)
        return img


# if __name__ == '__main__':
#     sess = tf.Session()
#     imagepath='/home/zhangxk/AIProject/WIDER_train/images/0--Parade/0_Parade_Parade_0_904.jpg'
#     pnet_path='/home/zhangxk/AIProject/MTCNN_TRAIN/pnet/model/PNet-3'
#     rnet_path = '/home/zhangxk/AIProject/MTCNN_TRAIN/rnet/model/RNet-1'
#     # rnet_path=None
#     print('p_net---->totalbox and next input:')
#
#     df=Detector_tf(
#                 sess=sess,
#                 minsize=50,
#                 scaleFactor=0.7,
#                 nms=[0.5,0.7,0.5,None],
#                 threshold=[0.6,0.3,None],
#                      model_path=[pnet_path],
#                      save_stage=False
#                 )
#     totalbox=df.detect_face(imagepath)
#     print(totalbox.shape)
#     image=drawDectectBox(imagepath, totalbox, scores=None)
#     imsave('ssss.jpg',image)