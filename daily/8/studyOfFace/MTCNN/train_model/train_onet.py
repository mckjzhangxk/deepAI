import train_model.train as train
from train_model.solver import onet_solver
from Configure import ONET_DATASET_PATH
from  utils.tf_utils import readTFRecord
from model.mtcnn_model import createRNet
from train_model.losses import mtcnn_loss_acc
import tensorflow as tf
import os

'''
从3个tf_file读取一次训练的batch,pos:part:neg=1:1:2
因为pos:part:neg数据存在不平衡现象,neg>>part>pos
所有读取分成了三个
'''
def getInput():
    image_batch, label_batch, roi_batch=[],[],[]
    ratio=[1/4,1/4,2/4]
    fnames=['ONet_pos_shuffle','ONet_part_shuffle','ONet_neg_shuffle']
    for r,fname in zip(ratio,fnames):
        tf_filename=os.path.join(ONET_DATASET_PATH,fname)
        assert os.path.exists(tf_filename) ,'%s TFRecord does not exist'%fname
        #每次都有比例
        bz=int(r*onet_solver.BATCH_SIZE)
        imgs, labs, rois=readTFRecord(tf_filename,bz,onet_solver.IMG_SIZE)
        image_batch.append(imgs)
        label_batch.append(labs)
        roi_batch.append(rois)
    image_batch=tf.concat(image_batch,axis=0)
    label_batch = tf.concat(label_batch, axis=0)
    roi_batch = tf.concat(roi_batch, axis=0)
    return image_batch,label_batch,roi_batch

def buildModel(input_images):
    p_prob, p_regbox,p_landmark = createRNet(input_images, trainable=True)
    return p_prob,p_regbox,p_landmark

def buildLoss(prob,regbox,landmark,label,roi):
    return mtcnn_loss_acc(prob,regbox,landmark,label,roi,cls_ratio=1.0,reg_ratio=0.5,landmark_ratio=1.0)


if __name__ == '__main__':
    train.svConf=onet_solver
    train.getInput = getInput
    train.buildModel = buildModel
    train.buildLoss = buildLoss
    train.start_train()


