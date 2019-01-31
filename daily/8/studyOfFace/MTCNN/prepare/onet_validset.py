from Configure import ONET_DATASET_VALID_PATH
from utils.tf_utils import cvtTxt2TF
from prepare.valiset import gen_valid_data,merge_dataset


if __name__ == '__main__':
    fname='ONet.txt'
    outputPath=ONET_DATASET_VALID_PATH
    gen_valid_data(outputPath,posCopys=1,negCopys=1,negNum=0,SIZE=48)
    merge_dataset(outputPath,fname)
    cvtTxt2TF(outputPath, fname, 'ONet_shuffle', False)