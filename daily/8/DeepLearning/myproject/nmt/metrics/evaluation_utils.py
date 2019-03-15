import tensorflow as tf
import codecs
import re
from metrics.bleu import compute_bleu


def blue(trans,refs,subword_option=None):
    '''
    要求trans,ref必须是list,长度一样，一对一
    :param trans: list(str),shape(N,)
    :param ref:list(list(str)),shape(R,N)，R是参考的数目
    :return: 
    '''
    if isinstance(refs,str):
        refs=load_file(refs,subword_option)

    transs=[_clean(line.strip(),subword_option).split(' ') for line in trans]

    refss=[]
    for reference in refs:
        refss.append([reference.strip().split(' ')])
    blue_score,_,_,_,_,_=compute_bleu(refss,transs,max_order=4,smooth=False)
    return 100*blue_score
def _clean(sentence,subword_option):
    sentence=sentence.strip()

    if subword_option=='bpe':
        sentence=re.sub('@@ ','',sentence)
    return sentence

def load_file(file_path,subword_option=False):
    '''
    返回list of string
    :param file_path: 
    :param subword_option: 
    :return: 
    '''
    with codecs.getreader('utf-8')(tf.gfile.GFile(file_path,mode='rb')) as fs:
        lines=fs.readlines()
        lines=[l.strip() for l in lines]
    if subword_option:
        return [_clean(l,subword_option) for l in lines]
    else:
        return lines