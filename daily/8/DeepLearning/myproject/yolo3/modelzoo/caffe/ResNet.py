from  modelzoo.caffe.BaseNet import BaseModel

class resNet50(BaseModel):
    def __init__(self,model_path):
        super(resNet50,self).__init__(model_path,(224,224,3))
    def get_weighs_name(self):
        return 'ResNet-50-model.caffemodel'
class resNet101(BaseModel):
    def __init__(self,model_path):
        super(resNet101,self).__init__(model_path,(224,224,3))
    def get_weighs_name(self):
        return 'ResNet-101-model.caffemodel'
class resNet152(BaseModel):
    def __init__(self,model_path):
        super(resNet152,self).__init__(model_path,(224,224,3))
    def get_weighs_name(self):
        return 'ResNet-152-model.caffemodel'
