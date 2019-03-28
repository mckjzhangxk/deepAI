from PyQt5 import QtWidgets,QtGui
import sys
import numpy as np
import cv2

def clickfunc():
    print('oooooooooooo')
def window():
    app=QtWidgets.QApplication(sys.argv)
    w=QtWidgets.QWidget()
    w.setWindowTitle('YOLO V3')
    w.setGeometry(100,100,100+1024,100+768)
    title=createLabel(None,'Demo')
    title=wrapHorizion(title)


    container=QtWidgets.QVBoxLayout()
    container.addLayout(title)

    w.setLayout(container)
    w_left=createImage(None, cv2.imread('data/raccoon_dataset/images/raccoon-33.jpg'))
    bn1=  createButton(None, 'forward',clickfunc)
    edit1=QtWidgets.QLineEdit()
    bn2 = createButton(None, 'backward')
    bn3 = createButton(None, 'exit')
    w_right=wrapVertical([bn1,edit1,bn2,bn3])

    frame=wrapHorizion([w_left,w_right])
    container.addLayout(frame)

    w.show()

    sys.exit(app.exec())

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
def wrapHorizion(p):
    hlayout=QtWidgets.QHBoxLayout()
    hlayout.addStretch()
    if isinstance(p,list):
        for w in p:
            if isinstance(w,QtWidgets.QBoxLayout):
                hlayout.addLayout(w)
            elif isinstance(w,QtWidgets.QFrame):
                hlayout.addWidget(w)
    else:
        if isinstance(p, QtWidgets.QBoxLayout):
            hlayout.addLayout(p)
        elif isinstance(p, QtWidgets.QFrame):
            hlayout.addWidget(p)
    hlayout.addStretch()
    return hlayout

def wrapVertical(p):
    vlayout=QtWidgets.QVBoxLayout()
    vlayout.addStretch()
    if isinstance(p,list):
        for w in p:
            vlayout.addWidget(w)
    else:
        vlayout.addWidget(p)
    vlayout.addStretch()
    return vlayout


def createImage(parent,content):
    if isinstance(content,str):
        pixels=QtGui.QPixmap(content)
    if isinstance(content,np.ndarray):
        img=QtGui.QImage(content,content.shape[1],content.shape[0],3*content.shape[1],QtGui.QImage.Format_RGB888)
        pixels=QtGui.QPixmap(img)
    l = QtWidgets.QLabel(parent) if parent else QtWidgets.QLabel()
    l.setPixmap(pixels)
    return l
window()