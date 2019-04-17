from  modelzoo.caffe.BaseNet import BaseModel


class alexNet(BaseModel):
    def __init__(self,model_path):
        super(alexNet,self).__init__(model_path,(227,227,3))
    def get_weighs_name(self):
        return 'bvlc_alexnet.caffemodel'