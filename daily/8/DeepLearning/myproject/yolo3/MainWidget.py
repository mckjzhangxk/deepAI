from PyQt5 import QtWidgets,QtGui
import sys
import numpy  as np
from dbutils import ImageBrower
import tensorflow as tf

def createLabel(parent,text):
    l=QtWidgets.QLabel(parent) if parent else QtWidgets.QLabel()
    l.setText(text)
    return l

def createButton(parent,text,handler=None):
    b = QtWidgets.QPushButton(parent) if parent else QtWidgets.QPushButton()
    b.setText(text)
    if handler:
        b.clicked.connect(handler)
    return b
def wrap(p,mode='H'):
    if mode=='H':
        layout=QtWidgets.QHBoxLayout()
    else:
        layout = QtWidgets.QVBoxLayout()
    layout.addStretch()
    if not isinstance(p, list):p=[p]
    for w in p:
        if isinstance(w,QtWidgets.QBoxLayout):
            layout.addLayout(w)
        else:
            layout.addWidget(w)
    layout.addStretch()
    return layout

def wrapGrid(p,shape):
    layout = QtWidgets.QGridLayout()
    r,c=shape
    for i in range(r):
        for j in range(c):
            layout.addWidget(p[i*c+j],i,j)
    return layout
def numpy2pixel(content):
    img = QtGui.QImage(content, content.shape[1], content.shape[0], 3 * content.shape[1], QtGui.QImage.Format_RGB888)
    pixels = QtGui.QPixmap(img)
    return pixels
def createImage(parent,content):
    if isinstance(content,str):
        pixels=QtGui.QPixmap(content)
    if isinstance(content,np.ndarray):
        # img=QtGui.QImage(content,content.shape[1],content.shape[0],3*content.shape[1],QtGui.QImage.Format_RGB888)
        pixels=numpy2pixel(content)
    l = QtWidgets.QLabel(parent) if parent else QtWidgets.QLabel()
    l.setPixmap(pixels)
    return l

class MyWidge(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Yolov3 demo')

        '''
        set full screen
        '''
        screen=QtWidgets.QApplication.primaryScreen()
        self.setGeometry(screen.availableGeometry())

        left  =self.left_frame()
        right =self.right_frame()

        container = QtWidgets.QVBoxLayout()
        self.anchorbox_img=createImage(None,np.zeros((1,1,3),np.uint8))
        frame=QtWidgets.QHBoxLayout()
        frame.addLayout(left)
        frame.addLayout(right)
        container.addLayout(frame)
        # container.addLayout(right)
        self.setLayout(container)
        self.show()
    def left_frame(self):
        img=np.zeros((416,416,3),np.uint8)
        img1 = createImage(None, img)
        img = np.random.randint(0,255,(416,416,3),np.uint8)
        img2 = createImage(None, img)
        img = np.random.randint(0, 255, (416, 416, 3), np.uint8)
        img3 = createImage(None, img)
        img = np.random.randint(0, 255, (416, 416, 3), np.uint8)
        img4 = createImage(None, img)

        self.img1=  img1
        self.img2 = img2
        self.img3 = img3
        self.img4 = img4

        imglist=wrapGrid([img1,img2,img3,img4],shape=(2,2))
        return imglist
    def right_frame(self):
        def loadBtn_click():

            self.imagebrower=ImageBrower(self.filepath.text(),self.anchorbox.text(),self.classname.text())
            self.refleshImage()
            I=self.imagebrower.anchorBox2Img()
            self.anchorbox_img.setPixmap(numpy2pixel(I))
            self.anchorbox_img.show()
        ediaLanel=QtWidgets.QLabel('coco file gt_json_file:')
        self.filepath  = QtWidgets.QLineEdit('/home/zxk/AI/coco/annotations/dev.txt')
        p1=wrap([ediaLanel,self.filepath],mode='H')


        ediaLanel=QtWidgets.QLabel('anchorbox gt_json_file:')
        self.anchorbox= QtWidgets.QLineEdit('/home/zxk/AI/tensorflow-yolov3/anchors.txt')
        p2=wrap([ediaLanel,self.anchorbox],mode='H')

        ediaLanel = QtWidgets.QLabel('class name:')
        self.classname= QtWidgets.QLineEdit('/home/zxk/PycharmProjects/deepAI1/daily/8/DeepLearning/myproject/yolo3/data/coco.names')
        p3 = wrap([ediaLanel, self.classname], mode='H')


        loadBtn = createButton(None, 'load', loadBtn_click)
        bn1 = createButton(None, '<<',lambda :self.refleshImage(False))
        bn2 = createButton(None, '>>',lambda :self.refleshImage(True))
        p4=wrap([loadBtn,bn1,bn2],'H')


        panel=wrap([p1,p2,p3,p4],mode='V')
        return panel

    def refleshImage(self,forward=True):
        if forward:
            image, image13, image26, image52=self.imagebrower.next()
        else:
            image, image13, image26, image52 = self.imagebrower.prev()
        self.img1.setPixmap(numpy2pixel(image))
        self.img2.setPixmap(numpy2pixel(image13))
        self.img3.setPixmap(numpy2pixel(image26))
        self.img4.setPixmap(numpy2pixel(image52))
tf.enable_eager_execution()
tf.executing_eagerly()
app=QtWidgets.QApplication(sys.argv)
w=MyWidge()
sys.exit(app.exec())
