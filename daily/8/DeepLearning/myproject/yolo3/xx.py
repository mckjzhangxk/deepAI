import glob
import os
filename=glob.iglob('/home/zhangxk/projects/deepAI/daily/8/DeepLearning/myproject/yolo3/data/*/*.jpg',recursive=True)

for f in filename:
    f1,e=os.path.splitext(f)
    print(f)