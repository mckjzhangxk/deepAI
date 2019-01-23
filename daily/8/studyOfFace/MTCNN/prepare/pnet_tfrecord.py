from Configure import PNET_DATASET_PATH
from utils.tf_utils import cvtTxt2TF





if __name__ == '__main__':
    cvtTxt2TF(PNET_DATASET_PATH,'PNet.txt','PNet_shuffle',True)
