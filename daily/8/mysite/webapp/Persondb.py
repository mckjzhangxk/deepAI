#import recognition as face_recognition
import recognition_facenet.recognition as face_recognition
import numpy as np
import os
from scipy.misc import imread,imresize,imshow
import pickle
from recognition_facenet.recognition import modify_autoContract,equalization
class FaceService():
    def __init__(self,N=100,D=128):
        self.N=N
        self.D=D
        self._db=np.zeros((N,D))
        self._num=0
        self.faceid=[]

    def getNumOfFace(self):
        return self._num
    def loaddb(self,dir,topath):
        N = 0
        for f in os.listdir(dir):
            N += 1

        for f in os.listdir(dir):
            self.addFace2db(os.path.join(dir,f))
            if self._num%50==0:
                print('load progress %f'%(self._num/N*100))
        with open(topath, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('load face database successful,load #%d faces!'%(self._num))
    def addFace2db(self,filename):
        characters = face_recognition.face_encodings(filename,prefunc=modify_autoContract)
        if len(characters)==0:return False
        self._appendFace(characters[0])
        self.faceid.append(filename[filename.rfind('/')+1:])
        self._num += 1
        return True
    def _appendFace(self,c):
        N=self._db.shape[0]
        if self._num>=N:
            _db = np.zeros((self.N, self.D))
            self._db=np.concatenate((self._db,_db),0)
        self._db[self._num]=c
    def deleteFile(self,id):
        if id in self.faceid:
            index=self.faceid.index(id)
            self.faceid.pop(index)
            if index!= self._num-1:
                self._db[index:self._num-1]=self._db[index+1:self._num]
            self._num-=1
            return True
        else:
            return False
    def detectFace(self,filename):
        
        cf=face_recognition.face_encodings(filename,prefunc=modify_autoContract)
        if(len(cf)==0):return (None,1000)

        distance=np.linalg.norm(self._db[:self._num]-cf[0],axis=1)
        chioce=np.argmin(distance)
        # distance=self._db[:self._num].dot(cf[0])
        # norma=np.linalg.norm(self._db[:self._num],axis=1)
        # normb=np.linalg.norm(cf[0])
        # distance=distance/norma/normb
        # chioce = np.argmax(distance)

        return (self.faceid[chioce],distance[chioce])

# if __name__ == '__main__':
    # pc=FaceService()
    # path='facedb'
    # for f in os.listdir(path):
    #     result=pc.addFace2db(os.path.join(path,f))
    #     print(f,' add ',result)
    # print(pc.getNumOfFace())
    # with open('faces.pickle', 'wb') as handle:
    #     pickle.dump(pc, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # with open('faces.pickle', 'rb') as handle:
    #     pc = pickle.load(handle)
    # dectPath='/home/zxk/mysite/webapp/detect'
    # for f in os.listdir(dectPath):
    #     id,dist=pc.detectFace(os.path.join(dectPath,f))
    #     if dist==1000:
    #         print('can not find %s'%f)
    #     else:
    #         print('%s ----> %s, dist %.3f'%(f,id,dist))
