from PyQt5 import QtWidgets,QtGui
import sys
import numpy  as np
from search.runTest import getFeature,LSH

def numpy2pixel(content):
    '''
    convert a numpy to pixel object(QT)
    :param content: ndarray
    :return: 
    '''

    content_show=np.zeros_like(content,np.uint8)
    content_show[:]=content

    img = QtGui.QImage(content_show, content.shape[1], content.shape[0], 3 * content.shape[1], QtGui.QImage.Format_RGB888)
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

        layout=QtWidgets.QHBoxLayout()

        self.container = QtWidgets.QVBoxLayout()
        layout.insertLayout(0,self.container)
        layout.insertLayout(1,self.searchPannel())
        self.loadImage()
        self.setToolKits()
        self.setLayout(layout)

        self.setButtonState(True,False)
    def loadImage(self):
        contents=self.dataModel.get_content()


        if self.container.count()==0:
            p = [ImageComponent(None, ct) for ct in contents]
            p+=[ImageComponent(None,np.zeros((112,112,3),np.uint8)) for _ in range(self.rows*self.cols-len(contents))]
            ui_imglist = wrapGrid(p, (self.rows, self.cols))
            self.container.addLayout(ui_imglist)
            self.Img_grid=p
        else:
            for i in range(len(self.Img_grid)):
                qimg=self.Img_grid[i]

                img=contents[i] if i<len(contents) else np.zeros((112,112,3),np.uint8)
                qpixel = numpy2pixel(img)
                qpixel = qpixel.scaledToWidth(112).scaledToHeight(112)
                qimg.setPixmap(qpixel)

        if hasattr(self,'lb_pageinfo'):
            self.lb_pageinfo.setText('page:%d/%d'%(self.dataModel.cursor+1,self.dataModel.totalPage))
    def setToolKits(self):
        previous_=ButtonComponent(None,"<<",lambda :self.dataModel.pageChange(self,'left'))
        label_=LabelComponent(None,'page:%d/%d'%(self.dataModel.cursor+1,self.dataModel.totalPage))

        next_=ButtonComponent(None,">>",lambda :self.dataModel.pageChange(self,'right'))

        toolpanel=wrap([previous_,label_,next_],'H')
        self.container.addLayout(toolpanel)

        self.bn_previous=previous_
        self.bn_next=next_
        self.lb_pageinfo=label_

    def clickHome(self):
        self.dataModel = self.dataModel.root()
        self.loadImage()

    def clickSearch(self):
        import cv2
        filename=QtWidgets.QFileDialog().getOpenFileName()[0]
        if filename==None or filename=="":return
        qimg=QtGui.QPixmap()
        qimg.load(filename)
        qimg=qimg.scaledToWidth(112).scaledToHeight(112)
        self.Img_search.setPixmap(qimg)

        self.dataModel=self.dataModel.search(filename)

        self.loadImage()
    def searchPannel(self):
        vlayout=QtWidgets.QVBoxLayout()

        imgcmp=ImageComponent(None,np.random.randint(0,255,(112,112,3),np.uint8))
        bn=ButtonComponent(None,"search",self.clickSearch)
        bn1 = ButtonComponent(None, "Home", self.clickHome)

        vlayout.addStretch()

        vlayout.addWidget(imgcmp)

        vlayout.addWidget(bn)
        vlayout.addWidget(bn1)
        vlayout.addStretch()
        vlayout.addStretch()

        self.Img_search=imgcmp
        return vlayout

    def setButtonState(self,pre_state,next_state):
        self.bn_previous.setDisabled(pre_state)
        self.bn_next.setDisabled(next_state)

lsh=LSH()
class DataModel():
    def __init__(self):
        self.parent=None

    def setDataSource(self,source):
        self.cursor = 0
        self.dbSource=source
        self.indexes=np.array([s[1] for s in source])

        if hasattr(self,'pageSize'):
            self._loadContent_()

    def _loadContent_(self):
        import cv2
        ret=[]
        for i in range(self.cursor*self.pageSize,min(self.cursor*self.pageSize+self.pageSize,len(self.dbSource))):
            imgfile=self.dbSource[i][0]
            I=cv2.imread(imgfile)[:,:,::-1]
            ret.append(I)
        self._imgs=ret

    def get_content(self):
        return self._imgs

    def setPageSize(self,pagesize):
        self.pageSize=pagesize
        self.totalPage=np.ceil(len(self.dbSource)/self.pageSize)
        self._loadContent_()

    def pageChange(self,ui,flag):
        state_1=False
        state_2=False
        if flag=='right':
            self.cursor+=1
        if flag=='left':
            self.cursor-=1
        if self.cursor==0:
            state_1=True
        if self.cursor==self.totalPage-1:
            state_2=True
        ui.setButtonState(state_1,state_2)
        self._loadContent_()
        ui.loadImage()
    def root(self):
        obj=self
        while obj.parent is not None:
            obj=obj.parent
        return obj
    def search(self,filename):

        obj=self
        while obj.parent is not None:
            obj=obj.parent

        emb=getFeature(filename)
        index = np.array(lsh.localHash(emb))

        # isSame=obj.indexes == index
        prob = (obj.indexes == index).sum(axis=1)

        # isSame = np.any(isSame, axis=1)
        subsetIndex = np.where(prob>=8)[0]
        prob=prob[subsetIndex]





        db=DataModel()
        subdb=[obj.dbSource[ii] for ii in subsetIndex]
        #sort the result base on prob

        subsetIndex = np.argsort(-prob)
        prob = prob[subsetIndex]
        print(prob)
        dbnew=[subdb[ii] for ii in subsetIndex]

        db.setDataSource(dbnew)
        db.setPageSize(self.pageSize)
        db.parent=obj

        return db
def defaultSource():
    dbSource=[]
    with open('record.sql') as fs:
        for line in fs:
            filename,indexes=line.split()
            dbSource.append((filename,eval(indexes)))
    return dbSource

app=QtWidgets.QApplication(sys.argv)
dbSource=defaultSource()
dbModel=DataModel()
dbModel.setDataSource(dbSource)
w=MyWidge(dbModel,4,4)
w.show()
sys.exit(app.exec())
