import numpy as np
from iou import general_iou
import os
import cv2
import tensorflow as tf

def kmeans(boxes, k, dist=np.median,seed=1):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    distances     = np.empty((rows, k)) ## N row x N cluster
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    # initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # Step 1: allocate each item to the closest cluster centers
##        for icluster in range(k): # I made change to lars76's code here to make the code faster
##            distances[:,icluster] = 1 - iou(clusters[icluster], boxes)
        distances=1-general_iou(boxes,clusters)
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances

def get_boxes(path):
    basedir=os.path.dirname(path)
    def _getHW(imagepath):
        I=cv2.imread(imagepath)
        H,W=I.shape[0],I.shape[1]
        return W,H
    anchor_box=[]
    with tf.gfile.GFile(path) as reader:
        lines=reader.readlines()
        for line in lines:
            sps=line.strip().split(' ')
            W,H=_getHW(os.path.join(basedir,sps[0]))
            for j in range(1,len(sps),5):
                x1,y1,x2,y2=float(sps[j]),float(sps[j+1]),float(sps[j+2]),float(sps[j+3])
                anchor_box.append([(x2-x1)/W,(y2-y1)/H])
    anchor_box=np.array(anchor_box)
    return anchor_box

def get_anchor_boxes(anofile,num_anchors):
    boxes=get_boxes(anofile)
    clusters,nearst_cluster,distances=kmeans(boxes,k=num_anchors,seed=None)
    clusters=sorted(clusters,key=lambda x:x[0]*x[1])
    clusters=np.array(clusters)
    return clusters,nearst_cluster,distances
def create_anchor_boxes(anofile,outputfile,num_anchors=9):
    clusters,_,_=get_anchor_boxes(anofile,num_anchors)
    clusters=clusters.ravel().tolist()
    with open(outputfile,'w') as fs:
        fs.write(' '.join(map(str,clusters)))
# if __name__ == '__main__':
#     anofile='data/train.txt'
#     outputfile='data/raccoon_my_anchors.txt'
#     create_anchor_boxes(anofile, outputfile, num_anchors=9)