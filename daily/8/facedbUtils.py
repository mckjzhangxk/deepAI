import os
import face_recognition
import numpy as np
import pickle
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate

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
               path_image='facedb'):
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
################################Copy From FaceNet###########################################


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

if __name__ == '__main__':
    # getFaceDB()
    vs=getValidateSet()
    Xa,Xb,Y=vs['Xa'],vs['Xb'],vs['Y']
    print(Xa.shape)
    print(Xb.shape)
    print(len(Y))
