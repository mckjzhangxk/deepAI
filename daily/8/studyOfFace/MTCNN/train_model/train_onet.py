import train_model.train as train
from train_model.solver import onet_solver
from Configure import ONET_DATASET_PATH,ONET_DATASET_VALID_PATH
from  utils.tf_utils import readTFRecord
from model.mtcnn_model import createONet
from train_model.losses import mtcnn_loss_acc,calAccuracy
import tensorflow as tf
import os

'''
从3个tf_file读取一次训练的batch,pos:part:right=1:1:2
因为pos:part:neg数据存在不平衡现象,right>>part>pos
所有读取分成了三个
'''
def getInput():
    image_batch, label_batch, roi_batch,landmark_batch=[],[],[],[]
    ratio=[1/6,1/6,3/6,1/3]
    fnames=['ONet_pos_shuffle','ONet_part_shuffle','ONet_neg_shuffle','ONet_landmark_shuffle']
    for r,fname in zip(ratio,fnames):
        tf_filename=os.path.join(ONET_DATASET_PATH,fname)
        assert os.path.exists(tf_filename) ,'%s TFRecord does not exist'%fname
        #每次都有比例
        bz=int(r*onet_solver.BATCH_SIZE)
        imgs, labs, rois,landmarks=readTFRecord(tf_filename,bz,onet_solver.IMG_SIZE)
        image_batch.append(imgs)
        label_batch.append(labs)
        roi_batch.append(rois)
        landmark_batch.append(landmarks)

    image_batch=tf.concat(image_batch,axis=0)
    label_batch = tf.concat(label_batch, axis=0)
    roi_batch = tf.concat(roi_batch, axis=0)
    landmark_batch=tf.concat(landmark_batch,axis=0)
    return image_batch,label_batch,roi_batch,landmark_batch

def buildModel(input_images):
    p_prob, p_regbox,p_landmark = createONet(input_images, trainable=True)
    return p_prob,p_regbox,p_landmark

def buildLoss(prob, regbox, landmark, label, gt_roi, gt_landmark):
    return mtcnn_loss_acc(prob, regbox, landmark, label, gt_roi,gt_landmark, cls_ratio=1.0, reg_ratio=0.5, landmark_ratio=1.0)


def validateInput():
    tf_filename=os.path.join(ONET_DATASET_VALID_PATH,'ONet_shuffle')
    assert os.path.exists(tf_filename) ,'ONet validateInput TFRecord does not exist'
    image_batch,label_batch,roi_batch,landmark_batch=readTFRecord(tf_filename, onet_solver.BATCH_SIZE, onet_solver.IMG_SIZE)
    return image_batch,label_batch

def buildValidModel(input_images):
    createONet(input_images, trainable=True)
    g = tf.get_default_graph()
    prob_tensor=g.get_tensor_by_name('onet_1/prob1:0')
    return prob_tensor
def validateAccuracy(prob,label):
    acc=calAccuracy(prob, label)
    tf.summary.scalar('validate_accuracy', acc)
    return acc

if __name__ == '__main__':
    train.svConf=onet_solver
    train.getInput = getInput
    train.buildModel = buildModel
    train.buildLoss = buildLoss


    train.validate=True
    train.validateInput=validateInput
    train.validModel=buildValidModel
    train.validateAccuracy=validateAccuracy

    train.start_train()


