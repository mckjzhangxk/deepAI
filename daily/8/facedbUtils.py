import os
import face_recognition
import numpy as np
import pickle

def _loadDataset(path):
    '''
    
    
    :param 
    :return:return a dict with name(key),encoder code for names(a numpy array) 
    '''
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
def createValidationDataset(db,p_samples=100,n_samples=100):
    names=[name for name in db]
    n_classes = len(names)
    classesIdx = list(range(n_classes))
    Y = [1] * p_samples + [0] * n_samples
    Y=np.array(Y)


    positive_sample_classes=np.random.randint(0,n_classes,p_samples)
    #using Xa compare Xb
    Xa,Xb=[],[]
    for c in positive_sample_classes:
        name=names[c]
        db_c=db[name]
        chioceIdx=np.random.choice(list(range(len(db_c))),2,False)
        Xa.append(db_c[chioceIdx[0]])
        Xb.append(db_c[chioceIdx[1]])

    def _chioceFromDB(dset):
        id=np.random.choice(list(range(len(dset))),1)[0]
        return dset[id]

    for n in range(n_samples):
        chioceClasses=np.random.choice(classesIdx,2,False)
        c1,c2=chioceClasses[0],chioceClasses[1]

        db_c1,db_c2=db[names[c1]],db[names[c2]]
        Xa.append(_chioceFromDB(db_c1))
        Xb.append(_chioceFromDB(db_c2))
    Xa=np.array(Xa)
    Xb=np.array(Xb)

    return Xa,Xb,Y
def getFaceDB(path_encode='FaceRecognizationDataset/faces_db_test.pickle',
               path_image='/home/zhangxk/projects/daily/8/facedb'):
    db = _loadDataset(path_image)
    with open(path_encode, 'wb') as handle:
        pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return db

def getValidateSet(path='FaceRecognizationDataset/face_validate_set.pickle',path_encode='FaceRecognizationDataset/faces_db_test.pickle'):
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            validset = pickle.load(handle)
    else:
        with open(path_encode, 'rb') as handle:
            db = pickle.load(handle)
        Xa, Xb, Y = createValidationDataset(db)
        validset={'Xa':Xa,'Xb':Xb,'Y':Y}
        with open(path,'wb') as  handle:
                pickle.dump(validset,handle,protocol=pickle.HIGHEST_PROTOCOL)
    return validset
if __name__ == '__main__':
    vs=getValidateSet()
    Xa,Xb,Y=vs['Xa'],vs['Xb'],vs['Y']
    print(Xa.shape)
    print(Xb.shape)
    print(len(Y))