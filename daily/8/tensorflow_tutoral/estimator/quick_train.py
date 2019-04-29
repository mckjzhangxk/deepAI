import tensorflow as tf
from censut_dataset import define_feature_names,getDataSet
from functools import partial
import argparse


def createParser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_dir',type=str,default='/tmp',help='location of model')
    parser.add_argument('--train_file', type=str, help='location of train input')
    parser.add_argument('--eval_file', type=str, help='location of eval input')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.1)
    return parser
def train(args):
    train_inp = partial(getDataSet,
                        filename=args.train_file,
                        batch_size=args.batch_size,
                        epoch=args.epoch,
                        shuffle=args.shuffle)
    eval_inp = partial(getDataSet,
                       filename=args.eval_file,
                       batch_size=args.batch_size,
                       epoch=1,
                       shuffle=False)

    model = tf.estimator.LinearClassifier(
        model_dir=args.model_dir,
        feature_columns=define_feature_names(),
        optimizer=tf.train.FtrlOptimizer(learning_rate=args.lr))

    model.train(train_inp)
    results=model.evaluate(eval_inp)
    for key, value in sorted(results.items()):
        print('%s: %0.2f' % (key, value))
# quick_train --model_dir=models --train_file=data/test.txt --eval_file=data/test.txt --batch_size=4 --epoch=10 --shuffle=True
if __name__ == '__main__':
    parser=createParser()
    args=parser.parse_args()
    train(args)
