from tensorpack.dataflow import DataFromGenerator,DataFlow,BatchData
import numpy as np
# def f():
#     for i in range(10):
#         imgs=np.random.randint(0,256,(32,64,64,3))
#         lables=np.random.randint(0,100,(32,))
#         yield [imgs,lables]
class MyDataFlow(DataFlow):
    def __iter__(self):
        for i in range(10):
            imgs = np.random.randint(0, 256, (64, 64, 3))
            lables = np.random.randint(0, 100,)
            yield [imgs, lables]
    def __len__(self):return 10
# df=DataFromGenerator(f)
# for f in df:
#     print(f[0].shape,f[1].shape)
ds=MyDataFlow()
print(len(ds))
ds=BatchData(ds,3,True)
for d in ds:
    print(d[0].shape)
# print('xx')
# # df.reset_state()
# for d in df:
#     print(d[0].shape)