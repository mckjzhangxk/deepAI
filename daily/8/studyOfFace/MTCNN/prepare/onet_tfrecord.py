from Configure import ONET_DATASET_PATH
from utils.tf_utils import cvtTxt2TF



if __name__ == '__main__':
    shuffle=True
    cvtTxt2TF(ONET_DATASET_PATH,'pos.txt', 'ONet_pos_shuffle',shuffle)
    cvtTxt2TF(ONET_DATASET_PATH,'part.txt','ONet_part_shuffle',shuffle)
    cvtTxt2TF(ONET_DATASET_PATH, 'right.txt','ONet_neg_shuffle', shuffle)
    cvtTxt2TF(ONET_DATASET_PATH, 'landmark.txt','ONet_landmark_shuffle', shuffle)
