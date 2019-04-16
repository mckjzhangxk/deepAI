from  modelzoo.BaseNet import BaseModel


class vgg16(BaseModel):
    def __init__(self,model_path):
        super(vgg16,self).__init__(model_path,(224,224,3))
    def get_weighs_name(self):
        return 'VGG_ILSVRC_16_layers.caffemodel'