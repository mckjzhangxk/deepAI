from PyQt5 import QtWidgets,QtGui
import sys
import numpy  as np

def numpy2pixel(content):
    '''
    convert a numpy to pixel object(QT)
    :param content: ndarray
    :return: 
    '''
    img = QtGui.QImage(content, content.shape[1], content.shape[0], 3 * content.shape[1], QtGui.QImage.Format_RGB888)
    pixels = QtGui.QPixmap(img)
    return pixels
def ImageComponent(parent,content):
    '''
    :param parent: parent widgets
    :param content: nbarray
    :return: a image component 
    '''
    if isinstance(content,str):
        pixels=QtGui.QPixmap(content)
    if isinstance(content,np.ndarray):
        # img=QtGui.QImage(content,content.shape[1],content.shape[0],3*content.shape[1],QtGui.QImage.Format_RGB888)
        pixels=numpy2pixel(content)
    l = QtWidgets.QLabel(parent) if parent else QtWidgets.QLabel()
    l.setPixmap(pixels)
    return l
def ButtonComponent(parent,text,handler=None):
    b = QtWidgets.QPushButton(parent) if parent else QtWidgets.QPushButton()
    b.setText(text)
    if handler:
        b.clicked.connect(handler)
    return b
def LabelComponent(parent,text):
    l=QtWidgets.QLabel(parent) if parent else QtWidgets.QLabel()
    l.setText(text)
    return l

def wrapGrid(p,shape,parent=None):
    '''
    create a grid layout to hold all of p
    :param p: 
    :param shape: 
    :return: 
    '''
    if parent is None:
        layout = QtWidgets.QGridLayout()
    else:
        for child in parent.children():
            parent.removeWidget(child)
        layout=parent

    r,c=shape
    for i in range(r):
        for j in range(c):
            if(i*c+j<len(p)):
                layout.addWidget(p[i*c+j],i,j)
            else:break

    return layout
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

class MyWidge(QtWidgets.QWidget):
    def __init__(self,dataModel,rows=8,cols=8):
        super().__init__()
        self.dataModel=dataModel
        dataModel.setPageSize(rows * cols)


        self.rows=rows
        self.cols=cols

        self.setWindowTitle('Yolov3 demo')
        self.container = QtWidgets.QVBoxLayout()
        self.loadImage()
        self.setToolKits()
        self.setLayout(self.container)
    def loadImage(self):
        contents=self.dataModel.get_content()

        p=[ImageComponent(None,ct) for ct in contents]
        ui_imglist = wrapGrid(p, (self.rows, self.cols))

        if self.container.count()==0:
            self.container.addLayout(ui_imglist)
        else:
            oldui=self.container.itemAt(0)
            self.container.removeItem(oldui)
            self.container.insertLayout(0, ui_imglist, 0)
    def setToolKits(self):
        previous_=ButtonComponent(None,"<<",lambda :self.dataModel.pageChange(self,'left'))
        label_=LabelComponent(None,'page:1/4')
        next_=ButtonComponent(None,">>",lambda :self.dataModel.pageChange(self,'right'))

        toolpanel=wrap([previous_,label_,next_],'H')
        self.container.addLayout(toolpanel)
class DataModel():
    def __init__(self):
        self.content=[np.random.randint(0,255,(100,100,3),np.uint8) for i in range(16)]
        self.content += [np.zeros((100,100,3),np.uint8) for i in range(16)]
        self.cursor=0
    def get_content(self):
        return self.content[self.cursor*self.pageSize:self.cursor*self.pageSize+self.pageSize]
    def setPageSize(self,pagesize):
        self.pageSize=pagesize
        self.totalPage=np.ceil(len(self.content)/self.pageSize)
    def pageChange(self,ui,flag):
        if flag=='right':
            self.cursor+=1
        if flag=='left':
            self.cursor-=1
        if self.cursor==0:
            print('disable <<')
        if self.cursor==self.totalPage-1:
            print('disable >>')
        ui.loadImage()

app=QtWidgets.QApplication(sys.argv)
dbModel=DataModel()

w=MyWidge(dbModel,4,4)
w.show()
sys.exit(app.exec())