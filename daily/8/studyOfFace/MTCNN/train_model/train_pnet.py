import train_model.train as train
from train_model.solver import pnet_solver
from Configure import PNET_DATASET_PATH
from  utils.tf_utils import readTFRecord
from model.mtcnn_model import createPNet
from train_model.losses import mtcnn_loss_acc
import os

def getInput():
    tf_filename=os.path.join(PNET_DATASET_PATH,'PNet_shuffle')
    assert os.path.exists(tf_filename) ,'PNet TFRecord does not exist'
    image_batch,label_batch,roi_batch,landmark_batch=readTFRecord(tf_filename, pnet_solver.BATCH_SIZE, pnet_solver.IMG_SIZE)
    return image_batch,label_batch,roi_batch,landmark_batch

def buildModel(input_images):
    p_prob, p_regbox,p_landmark = createPNet(input_images, trainable=True)
    return p_prob,p_regbox,p_landmark

def buildLoss(prob, regbox, landmark, label, gt_roi, gt_landmark):
    return mtcnn_loss_acc(prob, regbox, landmark, label, gt_roi,gt_landmark, cls_ratio=1.0, reg_ratio=0.5, landmark_ratio=0.5)

if __name__ == '__main__':
    train.svConf=pnet_solver
    train.getInput = getInput
    train.buildModel = buildModel
    train.buildLoss = buildLoss
    train.start_train()


