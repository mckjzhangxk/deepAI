import os
import face_recognition
import numpy as np
import pickle

def loadDataset(path='/home/zhangxk/projects/daily/8/facedb'):
    ret={}
    for name in os.listdir(path):
        ret[name]=[]
        db=ret[name]

        path_name=os.path.join(path,name)
        for filename in os.listdir(path_name):
            if filename.rfind('.jpg')==-1 \
                    and filename.rfind('.png')==-1 \
                    and filename.rfind('.jpeg')==-1:continue
            samples_path=os.path.join(path_name, filename)
            img = face_recognition.load_image_file(samples_path)
            encodes=face_recognition.face_encodings(img)
            if len(encodes)>0:
                encode=encodes[0]
                db.append(encode)
            else:
                print(samples_path,',fail to recognize')
        db=np.array(db)
        ret[name]=db
        print('finish %s,data size %d'%(name,len(db)))
        print('------------------------------------------------------')
    return ret
if __name__ == '__main__':
    db=loadDataset()
    with open('faces.pickle', 'wb') as handle:
        pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)