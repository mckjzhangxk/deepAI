from mainDetector import CCPD_Detector


def test_main():
    import glob
    import os
    import cv2
    from yolov3.utils.utils import plot_one_box
    from  PIL import Image


    cn=CCPD_Detector()


    imglist = glob.glob('data/detector/*')

    for imgfile in imglist:
        I = cv2.imread(imgfile)
        I1 = Image.open(imgfile, mode='r').convert("L")


        result = cn.predict(I)
        basename=None
        for x1, y1, x2, y2, conf, cls,label in result:
            plot_one_box((x1, y1, x2, y2), I, label='%.2f' % conf, color=(0, 255, 0))
            basename=label+'.jpg'
        basename=basename if basename else os.path.basename(imgfile)
        outfile = os.path.join('data/output', basename)

        cv2.imwrite(outfile, I)

