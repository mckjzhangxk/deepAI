import re
def _clean(sentence,subword_option):
    sentence=sentence.strip()

    if subword_option=='bpe':
        sentence=re.sub('@@ ','',sentence)
    return sentence

def get_translation(nmt_output,sen_id,eos,subword_option):
    '''
    把nmt_output转成一个字符串（utf-8编码的byte数组）
    :param nmt_output: nmt的输出,shape[N,?],type=byte,numpy类型
    :param sen_id: sentence id
    :param eos: 
    :return: a string without padding
    '''
    eos=eos.encode('utf-8')
    sentence=list(nmt_output[sen_id])
    if eos in sentence:
        sentence=sentence[:sentence.index(eos)]
    translation=b' '.join(sentence)

    return _clean(translation.decode('utf-8'),subword_option)
