from  modelzoo.BaseNet import BaseModel


class googleNet(BaseModel):
    def __init__(self,model_path):
        super(googleNet,self).__init__(model_path,(224,224,3))
    def get_weighs_name(self):
        return 'bvlc_googlenet.caffemodel'