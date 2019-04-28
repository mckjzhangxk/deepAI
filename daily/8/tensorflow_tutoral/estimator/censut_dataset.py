import tensorflow as tf
import tensorflow.feature_column as fc
import urllib
import argparse
import os
from functools import partial

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_default =[[0],[''],[0],[''],[0],[''],[''],[''],[''],[''],[0], [0], [0], [''], ['']]
def _download_and_clean_file(filename, url):
  """Downloads data from url, and makes changes to match the CSV format."""
  temp_file, _ = urllib.request.urlretrieve(url)
  with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
    with tf.gfile.Open(filename, 'w') as eval_file:
      for line in temp_eval_file:
        line = line.strip()
        line = line.replace(', ', ',')
        if not line or ',' not in line:
          continue
        if line[-1] == '.':
          line = line[:-1]
        line += '\n'
        eval_file.write(line)
  tf.gfile.Remove(temp_file)


def download(data_dir):
  """Download census data if it is not already present."""
  tf.gfile.MakeDirs(data_dir)

  training_file_path = os.path.join(data_dir, TRAINING_FILE)
  if not tf.gfile.Exists(training_file_path):
    _download_and_clean_file(training_file_path, TRAINING_URL)

  eval_file_path = os.path.join(data_dir, EVAL_FILE)
  if not tf.gfile.Exists(eval_file_path):
    _download_and_clean_file(eval_file_path, EVAL_URL)

def createParser():
  parser=argparse.ArgumentParser()
  parser.add_argument('--data_dir',default='/tmp',type=str,help='The path of datafile')
  return parser

def getDataSet(filename,batch_size=2,epoch=2,shuffle=False):
  def convert(line):
    tensors=tf.decode_csv(line,_CSV_default)
    r={colname:tensor for colname,tensor in zip(_CSV_COLUMNS[:-1],tensors[:-1])}
    label=tf.equal(tensors[-1],'>50K')
    return r,label

  dataset=tf.data.TextLineDataset(filename)
  dataset=dataset.repeat(epoch)

  if shuffle:
    dataset=dataset.shuffle(batch_size*4)

  dataset=dataset.map(convert,4)
  dataset=dataset.batch(batch_size)
  dataset = dataset.prefetch(batch_size)
  return dataset

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]


def define_feature_names():
  def _base():
    education_num=fc.numeric_column('education_num')
    capital_gain=fc.numeric_column('capital_gain')
    capital_loss=fc.numeric_column('capital_loss')
    hours_per_week=fc.numeric_column('hours_per_week')

    #categorical,embedding_column
    relationship=fc.categorical_column_with_vocabulary_file('relationship',vocabulary_file='data/relationship')
    relationship=fc.indicator_column(relationship)

    education=fc.categorical_column_with_vocabulary_file('education',vocabulary_file='data/education')
    education=fc.indicator_column(education)

    race=fc.categorical_column_with_vocabulary_file('race',vocabulary_file='data/race')
    race=fc.indicator_column(race)

    occupation=fc.indicator_column(fc.categorical_column_with_hash_bucket('occupation',20))
    return [education_num,capital_gain,capital_loss,hours_per_week,relationship,education,race,occupation]
  def _combination():
    education_occupation=fc.crossed_column(['education','occupation'],300)
    education_occupation=fc.indicator_column(education_occupation)
    return [education_occupation]
  def _bucket():
    age=fc.bucketized_column(fc.numeric_column('age'),[18,30,40,50,60])
    return [age]
  return _base()+_combination()+_bucket()

if __name__ == '__main__':
    parser=createParser()
    args=parser.parse_args()
    download(args.data_dir)

