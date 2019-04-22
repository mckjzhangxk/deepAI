import tensorflow as tf
from tensorpack import regularize_cost,l2_regularizer,GlobalAvgPooling
from tensorpack.tfutils.tower import TowerContext
from tensorpack.dataflow import DataFromList,MapData
from model.frcnn.basemodel import image_preprocess,resnet_c4_backbone,resnet_conv5
from model.frcnn.model_rpn import rpn_head,RPNAnchors,generate_rpn_proposals,rpn_losses
from model.frcnn.model_roi import sample_fast_rcnn_targets,roi_align,fastrcnn_outputs,fastrcnn_losses,fastrcnn_predictions
from model.frcnn.model_box import encode_bbox_target,decode_bbox_target,clip_boxes
from model.frcnn.utils import get_all_anchors
from model.frcnn.config import config as cfg,finalize_configs
from common.utils import restore_from_npz
import cv2
from tqdm import tqdm

class DetectionModel():

    def preprocess(self, image):
        '''
        norm图片之后,返回(N,C,H,W),caffe格式
        :param image: 
        :return: 
        '''
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def _parseInputs(self,inputs):
        if not isinstance(inputs,list):
            self.X=inputs
        else:
            self.X,self.anchor_gt_boxes,self.anchor_gt_labels,self.gt_boxes,self.gt_labels=inputs

    def forward(self, inputs,is_training=False):
        self._parseInputs(inputs)
        self.image = self.preprocess(self.X)     # 1CHW
        with TowerContext('',is_training):
            self.features = self._backbone()

            self.rpn_box_logit,self.rpn_label_logit,self.proposals_boxes=self._rpn_head(is_training)
            self.roi_box_logit,self.roi_label_logit=self._roi_head(is_training)

    def _backbone(self):
        raise NotImplemented
    def _rpn_head(self,is_training):
        raise NotImplemented
    def _roi_head(self,is_training):
        raise NotImplemented
    def _rpn_loss(self):
        raise NotImplemented
    def _roi_loss(self):
        raise NotImplemented
    def total_loss(self,weight_decay=1e-4):
        rpn_loss=self._roi_loss()
        roi_loss=self._roi_loss()

        if weight_decay>0:
            weight_loss = [regularize_cost('.*/W', l2_regularizer(weight_decay), name='wd_cost')]
        else:
            weight_loss=[]
        total_loss = tf.add_n([rpn_loss] + [roi_loss] + weight_loss, 'total_cost')
        return total_loss,rpn_loss,roi_loss,weight_loss

    def predict(self):
        raise NotImplemented

class ResNetC4Model(DetectionModel):
    def __init__(self,cfg):
        self.blocknum=cfg.BACKBONE.RESNET_NUM_BLOCKS
        self.rpn_head_dim=cfg.RPN.HEAD_DIM
        self.num_anchors=cfg.RPN.NUM_ANCHOR

        self.TRAIN_PRE_NMS_TOPK=cfg.RPN.TRAIN_PRE_NMS_TOPK
        self.TRAIN_POST_NMS_TOPK=cfg.RPN.TRAIN_POST_NMS_TOPK

        self.TEST_PRE_NMS_TOPK=cfg.RPN.TEST_PRE_NMS_TOPK
        self.TEST_POST_NMS_TOPK=cfg.RPN.TEST_POST_NMS_TOPK

        self.anchor_stride=cfg.RPN.ANCHOR_STRIDE

        self.num_classes=cfg.DATA.NUM_CLASS

        self.bbox_regression_weights=tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS)
        self.maxsize=cfg.PREPROC.MAX_SIZE
    def _backbone(self):
        return [resnet_c4_backbone(self.image,self.blocknum[:3])][0]

    def _rpn_head(self,is_training):
        label_logits, box_logits=rpn_head('rpn',self.features, self.rpn_head_dim,self.num_anchors)
        anchors=RPNAnchors(get_all_anchors(),None,None).narrow_to(self.features)
        proposals=anchors.decode_logits(box_logits)
        image_shape2d=tf.shape(self.image)[2:]
        proposal_boxes, _ = generate_rpn_proposals(
            tf.reshape(proposals, [-1, 4]),
            tf.reshape(label_logits, [-1]),
            image_shape2d,
            self.TRAIN_PRE_NMS_TOPK if is_training else self.TEST_PRE_NMS_TOPK,
            self.TRAIN_POST_NMS_TOPK if is_training else self.TEST_POST_NMS_TOPK)
        return box_logits,label_logits,proposal_boxes
    def _rpn_loss(self):
        anchors=RPNAnchors(get_all_anchors(),self.anchor_gt_labels,self.anchor_gt_boxes).narrow_to(self.features)
        losses = rpn_losses(anchors.gt_labels, anchors.encoded_gt_boxes(), self.rpn_label_logit, self.rpn_box_logit)
        return losses
    def _roi_head(self, is_training):
        if is_training:
            # sample proposal boxes in training,对rpn给出的proposal精心挑选训练样本
            self.proposals_boxes, self.proposals_gt_labels, fg_inds_wrt_gt= \
                sample_fast_rcnn_targets(self.proposals_boxes, self.gt_boxes, self.gt_labels)

            self.proposals_fg_id=tf.reshape(tf.where(self.proposals_labels>0),[-1])
            self.proposals_fg_boxes=tf.gather(self.proposals_boxes, self.proposals_fg_id)

            self.proposals_fg_gt_label=tf.gather(self.proposals_gt_labels, self.proposals_fg_id)
            self.proposals_fg_gt_boxes=tf.gather(self.gt_boxes,fg_inds_wrt_gt)
        boxes_on_featuremap = self.proposals_boxes * (1.0 / self.anchor_stride)

        roi_resized = roi_align(self.features, boxes_on_featuremap, 14)
        # (n,2048,7,7)
        feature_fastrcnn = resnet_conv5(roi_resized,self.blocknum[3]) # (n,2048,7,7)
        # Keep C5 feature to be shared with mask branch,(n,2048)
        feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')

        fastrcnn_label_logits, fastrcnn_box_logits=fastrcnn_outputs('fastrcnn',feature_gap,self.num_classes)
        return fastrcnn_box_logits,fastrcnn_label_logits

    def _roi_loss(self):
        roi_label_logit=self.roi_label_logit

        roi_fg_box_logit=None
        _=tf.gather(self.roi_box_logit,self.proposals_fg_id)
        i1=tf.range(tf.shape(self.proposals_fg_id)[0])
        i2=self.proposals_fg_gt_label
        ii=tf.stack([i1,i2],axis=1)
        roi_fg_box_logit=tf.gather_nd(_,ii)

        #处理gt label

        roi_gt_label=self.proposals_gt_labels
        roi_fg_gt_box =encode_bbox_target(self.proposals_fg_gt_boxes, self.proposals_fg_boxes)


        return fastrcnn_losses(
            roi_gt_label, roi_label_logit,
            roi_fg_gt_box,roi_fg_box_logit
        )

    def predict(self):
        anchors = tf.tile(tf.expand_dims(self.proposals_boxes, 1),
                          [1,self.num_classes, 1])   # N x #class x 4
        decoded_boxes = decode_bbox_target(
            self.roi_box_logit / self.bbox_regression_weights,
            anchors,
            self.maxsize
        )
        decoded_boxes = clip_boxes(decoded_boxes,tf.shape(self.image)[2:],
                                   name='fastrcnn_all_boxes')
        label_scores=tf.nn.softmax(self.roi_label_logit)
        return fastrcnn_predictions(decoded_boxes, label_scores, name_scope='output')

class ResNet50_C4(ResNetC4Model):
    def __init__(self,cfg,istraining):
        cfg.BACKBONE.RESNET_NUM_BLOCKS=[3,4,6,3]
        finalize_configs(istraining)
        super().__init__(cfg)

class FRCnnService():
    def __init__(self,configure=None,model_path=None):
        if configure is None:
            configure=cfg

        model = ResNet50_C4(configure,False)
        self.X=tf.placeholder(tf.float32,shape=(None,None,3))
        model.forward(self.X, False)
        self.box, self.score, self.label = model.predict()
        self.session=tf.Session()
        if model_path:self.restore(model_path)
    def restore(self,path):
        restore_from_npz(self.session,path)

    def predict_imagelist(self,imagelist,**kwargs):
        ds =DataFromList(imagelist)
        def f(fname):
            im = cv2.imread(fname)
            assert im is not None, fname
            return im
        ds = MapData(ds, f)
        ds.reset_state()
        ret=[]
        for img in tqdm(ds,'Doing Predictions:'):
            _b,_s,_l=self.session.run([self.box,self.score,self.label],feed_dict={self.X:img})
            obj = {'boxes': _b, 'labels': _l, 'scores': _s}
            ret.append(obj)
        return ret
# import tensorpack.utils.viz as viz
# if __name__ == '__main__':
#
#     with open('/home/zxk/PycharmProjects/deepAI1/daily/8/DeepLearning/myproject/yolo3/data/coco.names') as fs:
#         names=fs.readlines()
#
#     istraining=False
#     model_path='/home/zxk/AI/tensorpack/FRCNN/COCO-R50C4-MaskRCNN-Standard.npz'
#     service=FRCnnService(cfg,model_path)
#     imagelist=['/home/zxk/PycharmProjects/deepAI1/daily/8/DeepLearning/myproject/yolo3/data/demo_data/car.jpg']
#     result=service.predict_imagelist(imagelist)
#
#     im=cv2.imread(imagelist[0])
#     for r in result:
#         # print(r['boxes'].shape,r['labels'].shape,r['scores'].shape)
#         # print(r['scores'])
#
#         labels=['%s:%.2f'%(names[ll-1],round(ss,2)) for ll,ss in zip(r['labels'],r['scores'])]
#         im=viz.draw_boxes(im,r['boxes'],labels)
#         viz.interactive_imshow(im)