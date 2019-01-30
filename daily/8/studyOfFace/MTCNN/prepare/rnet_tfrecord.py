from Configure import RNET_DATASET_PATH
from utils.tf_utils import cvtTxt2TF



if __name__ == '__main__':
    shuffle=True
    cvtTxt2TF(RNET_DATASET_PATH,'pos.txt', 'RNet_pos_shuffle',shuffle)
    cvtTxt2TF(RNET_DATASET_PATH,'part.txt','RNet_part_shuffle',shuffle)
    cvtTxt2TF(RNET_DATASET_PATH,'right.txt','RNet_neg_shuffle', shuffle)
    cvtTxt2TF(RNET_DATASET_PATH, 'landmark.txt', 'RNet_landmark_shuffle', shuffle)
