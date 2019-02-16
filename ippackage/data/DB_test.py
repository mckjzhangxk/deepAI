from data.utils import load_data
from data.analysisUtils import pca,standardizeData
import numpy as np

path='/home/zhangxk/AIProject/ippack/ip_capture/out1/'
# features=['upcount','upsize','up_rate','downcount','downsize','down_rate']
features=['upcount','up_rate','downcount','down_rate']
db=load_data(path,features)

np_db=standardizeData(db.db,'STD')

print(np.min(np_db))
print(np.max(np_db))

image=pca(np_db)
#
print(np.min(image))
print(np.max(image))