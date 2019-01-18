import train_model.train as train
import Solver_Configure as pconf

if __name__ == '__main__':
    train.svConf=pconf
    train.start_train('PNet')


