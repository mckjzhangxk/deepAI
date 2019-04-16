import tensorflow as tf
from model.DarkNet import darknet53
import model.common as common
from model.utils import cpu_nms
from tqdm import tqdm
import numpy  as np



slim = tf.contrib.slim
class yolov3(object):
    def __init__(self,
                 num_classes=80,
                 batch_norm_decay=0.9,
                 leaky_relu=0.1,
                 anchors=[(10, 13), (16, 30), (33, 23),
                       (30, 61), (62, 45), (59, 119),
                       (116, 90), (156, 198), (373, 326)]
                 ):

        # self._ANCHORS = [[10 ,13], [16 , 30], [33 , 23],
        # [30 ,61], [62 , 45], [59 ,119],
        # [116,90], [156,198], [373,326]]
        self._ANCHORS = anchors
        self._BATCH_NORM_DECAY = batch_norm_decay
        self._LEAKY_RELU = leaky_relu
        self._NUM_CLASSES = num_classes
        self.feature_maps = []  # [[None, 13, 13, 255], [None, 26, 26, 255], [None, 52, 52, 255]]

    def _yolo_block(self, inputs, filters):
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = common._conv2d_fixed_padding(inputs, filters * 1, 1)
        route = inputs
        inputs = common._conv2d_fixed_padding(inputs, filters * 2, 3)
        return route, inputs

    def _detection_layer(self, inputs, anchors):
        num_anchors = len(anchors)
        feature_map = slim.conv2d(inputs, num_anchors * (5 + self._NUM_CLASSES), 1,
                                  stride=1, normalizer_fn=None,
                                  activation_fn=None,
                                  biases_initializer=tf.zeros_initializer())
        return feature_map

    def _reorg_layer(self, feature_map, anchors):
        '''
            feature_map是NN的输出,(gh,gw,3,2+2+1+C),要经过sigmoid,exp处理后前4维度转换成相对grid的长宽
            anchors:(3,2):还原boxes长宽使用,也是困惑的一点, rw=exp(feature_map(i,j,a,2)),rh=exp(feature_map(i,j,a,3))
            ,这里的rw,rh不再代表box长宽相对于网格单元长宽的比例,而是表示相对于anchors[a]的长宽比例
            返回:
            xy_offset:(gh,gw,1,2):网格系统坐标,xy_offset[i,j,0,]=(i,j)
            boxes:(N,gh,gw,3,4):box的中心,长宽,都是绝对坐标
            conf_logits:(N,gh,gw,3,1),存在检测目标的logit
            prob_logits:(N,gh,gw,3,C),每种类别出现的logit
        '''
        num_anchors = len(anchors)  # num_anchors=3
        grid_size = feature_map.shape.as_list()[1:3]
        # the downscale image in height and weight
        stride = tf.cast(self.img_size // grid_size, tf.float32)  # [h,w] -> [y,x]
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors, 5 + self._NUM_CLASSES])

        box_centers, box_sizes, conf_logits, prob_logits = tf.split(
            feature_map, [2, 2, 1, self._NUM_CLASSES], axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        # I think x_y_offset=tf.stack([x_offset,y_offset],axis=1)
        a, b = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
        x_y_offset = tf.cast(x_y_offset, tf.float32)

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * stride[::-1]

        box_sizes = tf.exp(box_sizes) * anchors  # anchors -> [w, h]
        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, prob_logits

    @staticmethod
    def _upsample(inputs, out_shape):

        new_height, new_width = out_shape[1], out_shape[2]
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
        inputs = tf.identity(inputs, name='upsampled')

        return inputs

    def forward(self, inputs, is_training=False, reuse=False):
        """
        Creates YOLO v3 model.

        :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
               Dimension batch_size may be undefined. The channel order is RGB.
        :param is_training: whether is training or not.
        :param reuse: whether or not the network and its variables should be reused.
        :return:
        """
        # it will be needed later on
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self._BATCH_NORM_DECAY,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        # Set activation_fn and parameters for conv2d, batch_norm.
        with slim.arg_scope([slim.conv2d, slim.batch_norm, common._fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self._LEAKY_RELU)):
                route_1, route_2, inputs = darknet53(inputs).outputs
                with tf.variable_scope('yolo-v3'):
                    route, inputs = self._yolo_block(inputs, 512)
                    feature_map_1 = self._detection_layer(inputs, self._ANCHORS[6:9])
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inputs = common._conv2d_fixed_padding(route, 256, 1)
                    upsample_size = route_2.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_2], axis=3)

                    route, inputs = self._yolo_block(inputs, 256)
                    feature_map_2 = self._detection_layer(inputs, self._ANCHORS[3:6])
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inputs = common._conv2d_fixed_padding(route, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = self._upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, route_1], axis=3)

                    route, inputs = self._yolo_block(inputs, 128)
                    feature_map_3 = self._detection_layer(inputs, self._ANCHORS[0:3])
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    # reshape reduce spatial dim
    def _reshape(self, x_y_offset, boxes, confs, probs):

        grid_size = x_y_offset.shape.as_list()[:2]
        boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
        confs = tf.reshape(confs, [-1, grid_size[0] * grid_size[1] * 3, 1])
        probs = tf.reshape(probs, [-1, grid_size[0] * grid_size[1] * 3, self._NUM_CLASSES])

        return boxes, confs, probs

    def decode(self, feature_maps):
        """
        Note: given by feature_maps, compute the receptive field
              and get boxes, confs and class_probs
        input_argument: feature_maps -> [None, 13, 13, 255],
                                        [None, 26, 26, 255],
                                        [None, 52, 52, 255],
        输入NN的输出

        返回 boxes, confs, probs
        #(N,-1,4),#(N,-1,1),#(N,-1,C),-1=10647!
        不要被方法名误导,这里相当于slide window,3张缩放图的每个可能的区域都检测一遍,然后把检测结果返回,nms不再本方法进行!
        """
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        feature_map_anchors = [(feature_map_1, self._ANCHORS[6:9]),
                               (feature_map_2, self._ANCHORS[3:6]),
                               (feature_map_3, self._ANCHORS[0:3]), ]

        results = [self._reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]
        boxes_list, confs_list, probs_list = [], [], []

        for result in results:
            boxes, conf_logits, prob_logits = self._reshape(*result)

            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)

            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        boxes = tf.concat(boxes_list, axis=1)  # (N,-1,4)
        confs = tf.concat(confs_list, axis=1)  # (N,-1,1)
        probs = tf.concat(probs_list, axis=1)  # (N,-1,C),-1表示所有可能的区域索引的数量,一共3*(13^2+26^2+52^2)=10647个

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x0 = center_x - width / 2.
        y0 = center_y - height / 2.
        x1 = center_x + width / 2.
        y1 = center_y + height / 2.

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        return boxes, confs, probs


    def compute_loss(self, pred_feature_map, y_true, ignore_thresh=0.5, max_box_per_image=8):
        """
        Note: compute the loss
        Arguments: y_pred, list -> [feature_map_1, feature_map_2, feature_map_3]
                                        the shape of [None, 13, 13, 3*85]. etc
        """
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        total_loss = 0.
        # total_loss, rec_50, rec_75,  avg_iou    = 0., 0., 0., 0.
        _ANCHORS = [self._ANCHORS[6:9], self._ANCHORS[3:6], self._ANCHORS[0:3]]

        for i in range(len(pred_feature_map)):
            result = self.loss_layer(pred_feature_map[i], y_true[i], _ANCHORS[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]

        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    '''
      y_true:(N,gh,gw,3,85)图片的绝对标注
      feature_map_i:(N,gh,gw,3,85) NN的输出

      1.对于标注的区域:计算如下4种loss
      conf_loss:判定区域有检测目标,使用sigmoid,计算binary entropy loss
      class_loss:对于标注的类型,使用sigmoid,计算 binary entropy loss
      xy_loss:把ytrue[...,0:2]转化为相对于cell的坐标%,求l2 loss

      wh_loss:w,h转换成相对于各自的anchor box的比例后,取log,计算l2 loss


      2.对于非标注的区域:
       如果区域与所有标注区域不相交(all iou<0.5)
       判定这个区域没有检测目标,计算conf_loss,
      返回:
            xy_loss:中心loss
            wh_loss:长宽loss
            conf_loss:objectness loss
            class_loss:分类loss
      1.所有loss最终都sum()/batch_size
      2.更加关注小box,weight=2-(gt box area)/(image area)
    '''

    def loss_layer(self, feature_map_i, y_true, anchors):
        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        grid_size_ = feature_map_i.shape.as_list()[1:3]

        y_true = tf.reshape(y_true, [-1, grid_size_[0], grid_size_[1], 3, 5 + self._NUM_CLASSES])

        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self._reorg_layer(feature_map_i, anchors)
        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]
        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))

        # shape: [V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]
        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou
        # shape: [N, 13, 13, 3, V]
        iou = self._broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # shape: [N, 13, 13, 3]
        best_iou = tf.reduce_max(iou, axis=-1)
        # get_ignore_mask,只有当某个坐标点和所有GT_OBJ都不想交(很小的IOU,all IOU<0.5),才能肯定说这个点确实没有要检测物体,你应该把-log(1-prob_logit)作为
        # objectness loss的一部分
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)
        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th, numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)

        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1], 2-boxes_arae/image_area,
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (
        y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N

        # shape: [N, 13, 13, 3, 1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:],
                                                                           logits=pred_prob_logits)
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def _broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''
        maintain an efficient way to calculate the iou matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match

        true_box_xy, true_box_wh:表示标注的boxes中心和长宽,都是绝对坐标,(V,2),(V,2),V是一个batch中标注的总数>=batch_size
        pred_box_xy, pred_box_wh:表示NN输出的boxes中心和长宽,也是绝对坐标,(N,gh,gw,3,2),(N,gh,gw,3,2)

        返回:iou=[N, 13, 13, 3, V],例如
          iou[1000,2,5,3,:]表示NN对于example[1000]的(2,5,3)这个slot输出的boxes与所有标注的boxes的覆盖率(IOU值)
        '''
        # shape:
        # true_box_??: [V, 2]
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

        return iou

class YoLoService():
    def __init__(self,model_path):
        self.backend=yolov3()
        self._build_()
        self.restore_model(model_path)
    def _build_(self):
        with tf.variable_scope('yolov3'):
            self.X = tf.placeholder(tf.float32, shape=(None, 416, 416, 3))
            feature_maps = self.backend.forward(self.X,False)
            self.boxes, conf, prob = self.backend.decode(feature_maps)  # (N,-1,4),(N,-1,1),(N,-1,C)
            self.score_per_class = conf * prob  # (N,-1,C)

    def restore_model(self,path):
        self.sess = tf.Session()
        print('Recover Model From Path:%s'%path)
        saver=tf.train.Saver()
        saver.restore(self.sess,path)
    def predict_416(self,x,batchSize=128 ,score_thresh=0.3, iou_thresh=0.5):
        ret=[]
        for k in tqdm(range(int(np.ceil(len(x)/batchSize))),'Doing Predictions:'):
            xs=x[k*batchSize:k*batchSize+batchSize]
            _boxes,_scores=self.sess.run([self.boxes,self.score_per_class],feed_dict={self.X:xs})
            for _box,_score in zip(_boxes,_scores):
                _b,_s,_l=cpu_nms(_box,_score,
                                 self.backend._NUM_CLASSES,
                                 score_thresh=score_thresh,
                                 iou_thresh=iou_thresh)
                obj={'boxes':_b,'labels':_l,'scores':_s}
                ret.append(obj)
        return ret

    def predict_imagelist(self,imagelist,batchSize=128,score_thresh=0.3, iou_thresh=0.5):
        images_and_shape=list(map(common.processImage,imagelist))
        images=np.array([im for im,_  in images_and_shape])
        orgin_shape=[s for _,s in images_and_shape]


        result=self.predict_416(images,batchSize,score_thresh,iou_thresh)
        for r,orgin in zip(result,orgin_shape):
            w,h=orgin
            if len(r['boxes'])==0 :continue
            bb=r['boxes']*np.array([w,h,w,h])/416
            bb[:, [0, 2]] = np.maximum(bb[:, [0, 2]], 0)
            bb[:, [1, 3]] = np.maximum(bb[:, [1, 3]], 0)
            bb[:, [0, 2]] = np.minimum(bb[:, [0, 2]], w)
            bb[:, [1, 3]] = np.minimum(bb[:, [1, 3]], h)
            r['boxes']=bb
        return result