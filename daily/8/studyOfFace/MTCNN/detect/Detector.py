import cv2
import numpy as np
from detect.detect_block import generateBoundingBox,imresample,nms,bbreg,rerec,validbox,normalize,feedImage,nextInput,bb_landmark,pickfilter,pad,drawDectectBox
import tensorflow as tf
from model import createPNet,createRNet,createONet,PNet,RNet,ONet
import os
import matplotlib.pyplot as plt
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
                 nms=[0.5,0.7,0.7,None],
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
        if self.onet and len(total_box) > 0:
            total_box,landmark = self.o_stage(image,total_box)
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

            # out[1]是prosibility,out[1][0]是第一张图片,shape[H',W',2]
            regressor = out[0][0]
            score = out[1][0][:, :, 1]

            # regressor = np.transpose(out[0], (0, 2, 1, 3))[0]
            # score = np.transpose(out[1], (0, 2, 1, 3))[0][:, :, 1]

            # bbox是原图的坐标,(N',9),N'是概率>t保留下来的人脸数量
            bbox, _ = generateBoundingBox(score.copy(), regressor.copy(), scale, t)

            if bbox.size > 1:
                pick = nms(bbox.copy(), nms1, 'Union')
                bbox = bbox[pick]
                total_boxes = np.append(total_boxes, bbox, axis=0)

        #不同的scale图片处理筛选外成,要经过
        #reg修正->正方形化->有效性验证
        if len(total_boxes) > 1:
            pick = nms(total_boxes.copy(), nms2, 'Union')
            total_boxes = total_boxes[pick]
            '''
            using regress to adjust
            '''
            total_boxes = bbreg(total_boxes, total_boxes[:, 5:9])
            total_boxes = rerec(total_boxes.copy())
            # total_boxes=validbox(total_boxes)
        return total_boxes[:,:5]
    '''
    total_box是N,4
    image:(H,W,3)
    '''
    def r_stage(self,orignImage,total_boxes):
        #先根据total_box的信息,裁剪出来给onet图片,缩放成24,24,3
        images=nextInput(orignImage, total_boxes, 24)

        if len(images)==0:return np.empty((0,5))
        #feed  into rnet
        images=self.preprocessing(images)
        out=self._predictBatch(images,self.rnet)


        regressor,score=out[0] ,out[1][:,1] #(N',4),#(N',)

        ipass=score>self.m_threshold[1]
        total_boxes, score, regressor=pickfilter(ipass,[total_boxes,score,regressor])

        if len(total_boxes)>0:
            pick=nms(total_boxes,self.m_nms[2],'Union')
            total_boxes, score, regressor = pickfilter(pick, [total_boxes, score, regressor])
            total_boxes=rerec(bbreg(total_boxes,regressor))
            total_boxes[:,4]=score
        return total_boxes

    def o_stage(self,orignImage, total_boxes):
        #先根据total_box的信息,裁剪出来给onet图片,缩放成48,48,3
        images=nextInput(orignImage, total_boxes, 48)

        if len(images)==0:return np.empty((0,5))
        #feed  into rnet
        images=self.preprocessing(images)
        out = self._predictBatch(images, self.onet)

        regressor,landmark,score=out[0] ,out[1],out[2][:,1] #(N',4),#(N',)

        ipass=score>self.m_threshold[2]

        total_boxes, score, regressor,landmark=pickfilter(ipass,[total_boxes,score,regressor,landmark])


        if len(total_boxes)>0:
            total_boxes = bbreg(total_boxes.copy(), regressor)
            total_boxes[:, 4] = score
            pick=nms(total_boxes,self.m_nms[3],'Min')
            total_boxes, score,landmark = pickfilter(pick, [total_boxes, score,landmark])

            '''
            这里没有使用rect
            '''
            w,h,_=orignImage.shape
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes, w, h)
            boxes = np.stack([x, y, ex, ey], axis=1)
            landmark = bb_landmark(boxes, landmark)
        return total_boxes,landmark

    '''
        images:(N,H,W,3)
        net:rnet,onet
        out:( array(N,2),array(N,4),array(N,10) )
    '''
    def _predictBatch(self,images,net,batch=256):
        cur=0
        N=len(images)

        out=None
        while cur<N:
            inp_images=images[cur:min(N,cur+batch)]
            _out=net(inp_images)
            if out is None:
                out=list(_out)
            else:
                outdim=len(out)
                for k in range(outdim):
                    out[k]=np.concatenate([out[k],_out[k]],axis=0)
            cur+=len(inp_images)
        return tuple(out)

class Detector_Caffe(Detector):
    def preprocessing(self,img):
        img=normalize(img)
        img=feedImage(img,True)
        return img
    def load(self):
        sess=self.m_session
        p_path, r_path, o_path = self.m_model_path[0], None, None
        if len(self.m_model_path)>=2:
            r_path=self.m_model_path[1]
        if len(self.m_model_path) == 3:
            o_path = self.m_model_path[2]
        if p_path:
            with tf.variable_scope('pnet'):
                data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
                pnet = PNet({'data': data})
                pnet.load(os.path.join(self.m_model_path[0], 'det1.npy'), sess)
            self.pnet = lambda img: sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0': img})

        if r_path:
            with tf.variable_scope('rnet'):
                data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
                rnet = RNet({'data': data})
                rnet.load(os.path.join(self.m_model_path[1], 'det2.npy'), sess)
            self.rnet = lambda img: sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'),
                                             feed_dict={'rnet/input:0': img})
        if o_path:
            with tf.variable_scope('onet'):
                data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
                onet = ONet({'data': data})
                onet.load(os.path.join(self.m_model_path[2], 'det3.npy'), sess)
            self.onet= lambda img: sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'),
                                        feed_dict={'onet/input:0': img})

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
            prob_tensor_p, regbox_tensor_p,_=createPNet(data_p, True)
            self.pnet=lambda img: sess.run((regbox_tensor_p,prob_tensor_p), feed_dict={data_p: img})

            varlist=tf.get_collection('PNet')
            saver = tf.train.Saver(var_list=varlist)
            saver.restore(sess, p_path)

        if r_path:
            data_r = tf.placeholder(tf.float32, (None, 24, 24, 3))
            prob_tensor_r, regbox_tensor_r,_ =createRNet(data_r, True)
            self.rnet = lambda img: sess.run((regbox_tensor_r, prob_tensor_r), feed_dict={data_r: img})

            varlist=tf.get_collection('RNet')
            saver = tf.train.Saver(var_list=varlist)
            saver.restore(sess, r_path)
        if o_path:
            data_o=tf.placeholder(tf.float32, (None, 48, 48, 3))
            prob_tensor_o, regbox_tensor_o, landmark_tensor_o = createONet(data_o, True)
            self.onet = lambda img: sess.run((regbox_tensor_o,landmark_tensor_o, prob_tensor_o), feed_dict={data_o: img})
            varlist=tf.get_collection('ONet')
            saver = tf.train.Saver(var_list=varlist)
            saver.restore(sess, o_path)
    def preprocessing(self,img):
        img=normalize(img)
        img=feedImage(img)
        return img

# from scipy.misc import imsave
# if __name__ == '__main__':
#     sess = tf.Session()
#     imagepath='/home/zhangxk/projects/deepAI/daily/8/studyOfFace/MTCNN/detect/0_Parade_marchingband_1_849.jpg'
#     # imagepath='/home/zhangxk/AIProject/MTCNN_TRAIN/pnet/dataset/part/1262.jpg'
#
#
#     pnet_path='/home/zhangxk/projects/deepAI/daily/8/studyOfFace/mylib/detect'
#     # pnet_path='/home/zhangxk/AIProject/MTCNN_TRAIN/model/PNet-29'
#
#     rnet_path = '/home/zhangxk/projects/deepAI/daily/8/studyOfFace/mylib/detect'
#     onet_path = '/home/zhangxk/projects/deepAI/daily/8/studyOfFace/mylib/detect'
#
#     rnet_path=None
#     onet_path=None
#     print('p_net---->totalbox and next input:')
#
#     df=Detector_Caffe(
#                 sess=sess,
#                 minsize=50,
#                 scaleFactor=0.709,
#                 nms=[0.5,0.6,0.7,0.7],
#                 threshold=[0.6,0.6,0.7],
#                 model_path=[pnet_path,rnet_path,onet_path],
#                 save_stage=False
#                 )
#     # df=Detector_tf(
#     #             sess=sess,
#     #             minsize=50,
#     #             scaleFactor=0.709,
#     #             nms=[0.5,0.6,0.7,0.7],
#     #             threshold=[0.6,0.6,0.7],
#     #             model_path=[pnet_path,rnet_path,onet_path],
#     #             save_stage=False
#     #             )
#
#     totalbox=df.detect_face(imagepath)
#     print(totalbox.shape)
#
#     image=drawDectectBox(imagepath, totalbox, scores=None)
#     imsave('ssss.jpg',image)