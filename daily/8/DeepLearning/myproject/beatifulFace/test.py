import cv2
import numpy as np


if __name__ == "__main__":
    Z=np.zeros((800,600,3))

    Z[100:300,100:300]=255

    K=np.float32([[0,1,0],[1,-4,1],[0,1,0]])
    Z1=cv2.filter2D(Z,-1,K)
    print(Z1.dtype)
    Z2=cv2.filter2D(Z,-1,-K)
    print(np.min(Z2))
    cv2.imshow('Z',Z)
    cv2.imshow('laplacian',Z1)
    cv2.imshow('inv',Z2-Z1)
    print(np.allclose(Z1,Z2))

    cv2.waitKey(0)