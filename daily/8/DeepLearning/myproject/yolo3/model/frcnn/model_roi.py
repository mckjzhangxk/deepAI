# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf

from tensorpack.models import Conv2D, FullyConnected, layer_register
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized_method
from model.frcnn.config import config as cfg
from model.frcnn.model_box import decode_bbox_target, encode_bbox_target,pairwise_iou



from .model_box import roi_align as roi_align



@under_name_scope()
def proposal_metrics(iou):
    """
    Add summaries for RPN proposals_boxes.

    Args:
        iou: nxm, #proposal x #gt
    """
    # find best roi for each gt, for summary only
    best_iou = tf.reduce_max(iou, axis=0)
    mean_best_iou = tf.reduce_mean(best_iou, name='best_iou_per_gt')
    summaries = [mean_best_iou]
    with tf.device('/cpu:0'):
        for th in [0.3, 0.5]:
            recall = tf.truediv(
                tf.count_nonzero(best_iou >= th),
                tf.size(best_iou, out_type=tf.int64),
                name='recall_iou{}'.format(th))
            summaries.append(recall)
    add_moving_summary(*summaries)


@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels):
    """
    Sample some boxes from all proposals_boxes for training.
    #fg is guaranteed to be > 0, because ground truth boxes will be added as proposals_boxes.
    
    boxes是所有proposal的区域,采集1/4的前景色,3/4的背景色,
    然后返回(N',4)boxes -->boxes
           (N',)labels-->boxes标注的类别
           (N',)fg_inds_wrt_gt-->前景与gt_box的关系,例如
                            fg_idx_wrt_gt[10]=22,表示第10个前景和第22标注的box的iou大,所以
                            把boxes[10]对于的labels[10]赋值 gt[10]的类别
    Args:
        boxes: nx4 region proposals_boxes, floatbox
        gt_boxes: mx4, floatbox
        gt_labels: m, int32

    Returns:
        A BoxProposals instance.
        sampled_boxes: tx4 floatbox, the rois
        sampled_labels: t int64 labels, in [0, #class). Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
            It contains the matching GT of each foreground roi.
    """
    iou = pairwise_iou(boxes, gt_boxes)     # nxm
    proposal_metrics(iou)

    # add ground truth as proposals_boxes as well
    boxes = tf.concat([boxes, gt_boxes], axis=0)    # (n+m) x 4
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)   # (n+m) x m
    # #proposal=n+m from now on
    #iou是(n,m),iou>0.5的区域认为是前景色,否则是背景色,前景里面抽取FG_RATIO比例的,背景色抽取1-FG_RATIO
    # 返回fg_idx,bg_idx
    def sample_fg_bg(iou):
        fg_mask = tf.reduce_max(iou, axis=1) >= cfg.FRCNN.FG_THRESH

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(
            cfg.FRCNN.BATCH_PER_IM * cfg.FRCNN.FG_RATIO),
            tf.size(fg_inds), name='num_fg')
        fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            cfg.FRCNN.BATCH_PER_IM - num_fg,
            tf.size(bg_inds), name='num_bg')
        bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

        add_moving_summary(num_fg, num_bg)
        return fg_inds, bg_inds

    fg_inds, bg_inds = sample_fg_bg(iou)
    # fg,bg indices w.r.t proposals_boxes

    best_iou_ind = tf.argmax(iou, axis=1)   # #proposal, each in 0~m-1
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)   # num_fg,收集到前景色与gt_box对应的关系

    all_indices = tf.concat([fg_inds, bg_inds], axis=0)   # indices w.r.t all n+m proposal boxes
    ret_boxes = tf.gather(boxes, all_indices)

    ret_labels = tf.concat(
        [tf.gather(gt_labels, fg_inds_wrt_gt),
         tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)
    # stop the gradient -- they are meant to be training targets

    ret_boxes=tf.stop_gradient(ret_boxes, name='sampled_proposal_boxes'),
    ret_labels=tf.stop_gradient(ret_labels, name='sampled_labels'),
    fg_inds_wrt_gt=tf.stop_gradient(fg_inds_wrt_gt)
    return ret_boxes,ret_labels,fg_inds_wrt_gt


@layer_register(log_shape=True)
def fastrcnn_outputs(feature, num_classes, class_agnostic_regression=False):
    """
    fastrnn的输出 label(n,C),boxes(n,c,4),n是proposal boxes的数量
    features:(n,C)
    Args:
        feature (any shape):
        num_classes(int): num_category + 1
        class_agnostic_regression (bool): if True, regression to N x 1 x 4

    Returns:
        cls_logits: N x num_class classification logits
        reg_logits: N x num_classx4 or Nx2x4 if class agnostic
    """
    #(N,C)
    classification = FullyConnected(
        'class', feature, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    num_classes_for_box = 1 if class_agnostic_regression else num_classes
    #(N,4xC) or (N,4)
    box_regression = FullyConnected(
        'box', feature, num_classes_for_box * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    #box_regression:(N,C,4)
    box_regression = tf.reshape(box_regression, (-1, num_classes_for_box, 4), name='output_box')
    return classification, box_regression


@under_name_scope()
def fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits):
    """
    Args:
        labels: n, 所有proposal的label
        label_logits: nxC,所有proposal的label_logits
        fg_boxes: nfgx4, encoded,只有前景的boxes
        fg_box_logits: nfgxCx4 or nfgx1x4 if class agnostic,只有前景的boxes
        
        X:label_logits:(n,C)
          fg_box_logits:(#fg,C,4)
        Y:labels:(n,)
          fg_boxes:(#gt,4)  
        label_loss=binary_entropy(label_logits,labels).sum()/n
        box_loss=huger(fg_box_logits[fg_label],fg_boxes).sum(n)
        
    Returns:
        label_loss, box_loss
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')

    fg_inds = tf.where(labels > 0)[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds, out_type=tf.int64)
    empty_fg = tf.equal(num_fg, 0)
    if int(fg_box_logits.shape[1]) > 1:
        indices = tf.stack(
            [tf.range(num_fg), fg_labels], axis=1)  # #fgx2
        fg_box_logits = tf.gather_nd(fg_box_logits, indices)# (#fg,4)
    else:
        fg_box_logits = tf.reshape(fg_box_logits, [-1, 4])

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
        correct = tf.cast(tf.equal(prediction, labels), tf.float32)  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy') #判断的准确率
        #收集了标注为fg的boxes的预测输出标签
        fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
        #fg_label_pred标注的都是>0的,这里计算预测为0的,也就是FN
        num_zero = tf.reduce_sum(tf.cast(tf.equal(fg_label_pred, 0), tf.int64), name='num_zero')
        # fn/#fg 所有前景色中有多少被判断为背景,recall指标
        false_negative = tf.where(
            empty_fg, 0., tf.cast(tf.truediv(num_zero, num_fg), tf.float32), name='false_negative')
        # tp/#fg 所有前景色中判断为正确的
        fg_accuracy = tf.where(
            empty_fg, 0., tf.reduce_mean(tf.gather(correct, fg_inds)), name='fg_accuracy')

    box_loss = tf.losses.huber_loss(
        fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)
    box_loss = tf.truediv(
        box_loss, tf.cast(tf.shape(labels)[0], tf.float32), name='box_loss')

    add_moving_summary(label_loss, box_loss, accuracy,
                       fg_accuracy, false_negative, tf.cast(num_fg, tf.float32, name='num_fg_label'))
    return [label_loss, box_loss]


@under_name_scope()
def fastrcnn_predictions(boxes, scores):
    """
    roi输出scores:(N,C),经过还原后的boxes(N,C,4),本方法执行NMS
    最终输出final_label:(K,) final_label[i]表述输出i的类别>=1
           final_box:(K,4)
           final_score:(K,)  
    Generate final results from predictions of all proposals_boxes.

    Args:
        boxes: n#classx4 floatbox in float32
        scores: nx#class

    Returns:
        boxes: Kx4
        scores: K
        labels: K
    """
    assert boxes.shape[1] == cfg.DATA.NUM_CLASS
    assert scores.shape[1] == cfg.DATA.NUM_CLASS
    boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # #catxnx4
    scores = tf.transpose(scores[:, 1:], [1, 0])  # #catxn

    def f(X):
        """
        这个方法输入n个box返回对这n 个box的判断
        prob: n probabilities
        box: nx4 boxes

        Returns: n boolean, the selection
        """
        prob, box = X
        output_shape = tf.shape(prob, out_type=tf.int64)
        # filter by score threshold
        ids = tf.reshape(tf.where(prob > cfg.TEST.RESULT_SCORE_THRESH), [-1])
        prob = tf.gather(prob, ids)
        box = tf.gather(box, ids)
        # NMS within each class
        selection = tf.image.non_max_suppression(
            box, prob, cfg.TEST.RESULTS_PER_IM, cfg.TEST.FRCNN_NMS_THRESH)
        selection = tf.gather(ids, selection)

        if get_tf_version_tuple() >= (1, 13):
            sorted_selection = tf.sort(selection, direction='ASCENDING')
            mask = tf.sparse.SparseTensor(indices=tf.expand_dims(sorted_selection, 1),
                                          values=tf.ones_like(sorted_selection, dtype=tf.bool),
                                          dense_shape=output_shape)
            mask = tf.sparse.to_dense(mask, default_value=False)
        else:
            # this function is deprecated by TF,对selection按照升序排列
            sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
            mask = tf.sparse_to_dense(
                sparse_indices=sorted_selection,
                output_shape=output_shape,
                sparse_values=True,
                default_value=False)
        return mask

    # TF bug in version 1.11, 1.12: https://github.com/tensorflow/tensorflow/issues/22750
    buggy_tf = get_tf_version_tuple() in [(1, 11), (1, 12)]
    masks = tf.map_fn(f, (scores, boxes), dtype=tf.bool,
                      parallel_iterations=1 if buggy_tf else 10)     # #cat x N
    selected_indices = tf.where(masks)  # #selection x 2, each is (cat_id, box_id)
    scores = tf.boolean_mask(scores, masks)

    # filter again by sorting scores
    topk_scores, topk_indices = tf.nn.top_k(
        scores,
        tf.minimum(cfg.TEST.RESULTS_PER_IM, tf.size(scores)),
        sorted=False)
    #(K,2) 2=[C,n]
    filtered_selection = tf.gather(selected_indices, topk_indices)
    #保留下来的定位,cat_ids[i]表示第i个保留的box的类别,box_ids[i]表示第i个保留下来的box坐标
    cat_ids, box_ids = tf.unstack(filtered_selection, axis=1)

    final_scores = tf.identity(topk_scores, name='scores')
    final_labels = tf.add(cat_ids, 1, name='labels')
    final_ids = tf.stack([cat_ids, box_ids], axis=1, name='all_ids')
    final_boxes = tf.gather_nd(boxes, final_ids, name='boxes')
    return final_boxes, final_scores, final_labels


class BoxProposals(object):
    """
    A structure to manage box proposals_boxes and their relations with ground truth.
    """
    def __init__(self, boxes, labels=None, fg_inds_wrt_gt=None):
        """
        Args:
            boxes: Nx4
            labels: N, each in [0, #class), the true label for each input box
            fg_inds_wrt_gt: #fg, each in [0, M)

        The last four arguments could be None when not training.
        """
        for k, v in locals().items():
            if k != 'self' and v is not None:
                setattr(self, k, v)

    @memoized_method
    def fg_inds(self):
        """ Returns: #fg indices in [0, N-1] """
        return tf.reshape(tf.where(self.labels > 0), [-1], name='fg_inds')

    @memoized_method
    def fg_boxes(self):
        """ Returns: #fg x4"""
        return tf.gather(self.boxes, self.fg_inds(), name='fg_boxes')

    @memoized_method
    def fg_labels(self):
        """ Returns: #fg"""
        return tf.gather(self.labels, self.fg_inds(), name='fg_labels')

