import tensorflow as tf
import collections
import os

def _read_words(filename):
    f=tf.gfile.GFile(filename)
    data=f.read().replace('\n','<eos>').split()
    return data
def _build_vocab(filename):
    data=_read_words(filename)
    counter=collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id
def _file_to_word_idxs(filepath,word_to_idx):
    '''
        Load PTB raw data from data directory "data_path".
        Reads PTB text files, converts strings to integer ids,
        and performs mini-batching of the inputs.

    :param filepath:the path of input corpus
    :param word_to_idx: dict,
    :return:
    '''
    corpus=_read_words(filepath)
    return [word_to_idx[x] for x in corpus if x in word_to_idx]
def ptb_raw_data(data_path=None):
    '''

    :param data_path:base dir of ptb set
    :return:
    '''
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id=_build_vocab(train_path)
    train_data=_file_to_word_idxs(train_path,word_to_id)
    valid_data=_file_to_word_idxs(valid_path,word_to_id)
    test_data=_file_to_word_idxs(test_path,word_to_id)
    vacabs=len(word_to_id)

    return train_data,valid_data,test_data,vacabs

def ptb_producer(raw_data, batch_size, num_steps, name=None):
    '''

    :param raw_data: return from ptb_raw_data
    :param batch_size:s
    :param num_steps:
    :param name:
    :return:
    '''
    with tf.name_scope(name,'PTB_SCOPE',values=[raw_data,batch_size,num_steps]):
        batch_length=len(raw_data)//batch_size
        raw_data=tf.convert_to_tensor(raw_data)

        data=tf.reshape(raw_data[0:batch_size*batch_length],(batch_size,batch_length))
        epoch_size=batch_length//num_steps -1

        i=tf.train.range_input_producer(epoch_size,shuffle=False).dequeue()
        x=data[0:batch_size,i*num_steps:i*num_steps+num_steps]
        y=data[0:batch_size, i * num_steps+1 : i * num_steps + num_steps+1]

        return x,y
if __name__ == '__main__':
    data_path='/home/zxk/AI/data/simple-examples/data'
    word_idx=_build_vocab(os.path.join(data_path,'ptb.train.txt'))

    idx_word={v:k for k,v in word_idx.items()}

    train_data, valid_data, test_data, vacabs=ptb_raw_data(data_path)

    trainX,trainY=ptb_producer(train_data,4,7)


    def array2str(x):
        strs=[]
        for k in range(len(x)):
            words=[idx_word[idx] for idx in x[k]]
            strs.append(' '.join(words))
        return strs

    with tf.Session() as sess:
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess,coord)
        try:
            for i in range(2):
                x,y=sess.run([trainX,trainY])
                print(array2str(x))
                print(array2str(y))
                print('------------------------')
        finally:
            coord.request_stop()
        coord.join(threads)