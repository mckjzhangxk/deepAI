import json
import tensorflow.contrib as tfcontrib
import tensorflow.logging as tflogging
class TPU_Solver():
    def __init__(self,config_file=None,config=None):
        '''
        本类 是 为了解决TPU使用繁琐的步骤,通过提供的config_file或config,
        无需再去定义诸如tpu_config,run_config一类的东西,你就可以使用TPU
        :param config_file:
        :param config:
        '''
        assert (config is not None) or (config_file is not None),'必须提供:配置文件或配置json对象'

        if config_file is not None:
            with open(config_file, encoding='utf-8') as fp:
                config = json.load(fp)
        self._get_necessary_property(config)
        self._basic_setup(config)

    def _get_necessary_property(self,config):
        _propertys=['use_tpu','train_batch_size','eval_batch_size','predict_batch_size']

        for p in _propertys:
            if p not in config:
                raise ValueError('%s 属性必须设置'%p)
            self.__setattr__(p,config[p])
    def _basic_setup(self,config):
        '''
            基本的配置TPU通信协议,如果use_tpu=False,或略一切关于tpu的配置
        :return:
        '''
        if self.use_tpu:
            tpu_name=config.get('tpu_name') if 'tpu_name' in config else None
            tpu_zone = config.get('tpu_zone') if 'tpu_zone' in config else None

            try:
                self.tpu=tfcontrib.cluster_resolver.TPUClusterResolver(tpu=tpu_name,zone=tpu_zone)
                info=self.tpu.cluster_spec().as_dict()
                tflogging.info('Run on TPU:%s'%(','.join(info['worker'])))
            except:
                raise ValueError('当前环境不支持TPU')
            tpu_iterations=config.get('tpu_iterations') if 'tpu_iterations' in config else None
            self.tpu_run_config=tfcontrib.tpu.TPUConfig(tpu_iterations)
            tpu_master=config.get('tpu_master') if 'tpu_master' in config else None
            self.run_config=tfcontrib.tpu.RunConfig(tpu_config=config,master=tpu_master,cluster=self.tpu)
        else:
            self.run_config=tfcontrib.tpu.RunConfig()

if __name__ == '__main__':
    tflogging.set_verbosity(tflogging.INFO)
    TPU_Solver('config.json')