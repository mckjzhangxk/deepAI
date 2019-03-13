from data.rnnInput import get_input
import tensorflow as tf
import numpy as np
def randomFile(path, N, T, D):
    with open(path, 'w') as f:
        for n in range(N):


            s=','.join(map(str,np.random.randint(0,500,T*D,dtype=np.int32)))
            label=str(n%2)

            slen=str(np.random.randint(1, T, dtype=np.int32))
            f.write(label+','+slen+','+s+'\n')

def output(batch, x, y, xlen, fs):
    '''

    :param batch: 
    :param x: (batch,Tmax,D)
    :param y: (batch)
    :param xlen: (batch)
    :param label: (batch)
    :param fs: 
    :return: 
    '''

    for i in range(batch):
        _x = list(x[i].ravel())
        _y = y[i]
        _xlen = xlen[i]
        record = [_y, _xlen] + _x
        record = map(int, record)
        record = map(str, record)

        _s = ','.join(record)
        fs.write(_s + '\n')





filename = 'resource/data'


BATCH, Tmax, D = 32, 300, 6
perodic = 30


randomFile(filename, 10000, Tmax, D)

def xx():
    trainInput = get_input(filename, BATCH, Tmax, D, perodic)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(trainInput.Iterator.initializer)

    steps = Tmax // perodic

    output_filenames=[
    'resource/data_output1',
    'resource/data_output2',
    'resource/data_output3',
    ]
    for i in range(len(output_filenames)):
        with open(output_filenames[i], 'w') as ofs:
            try:

                while True:
                    sess.run(trainInput.feed_Source)

                    _xx = None
                    for s in range(steps):
                        _bs, _x, _y, _xl, _cursor = sess.run([
                            trainInput.batch_size,
                            trainInput.X,
                            trainInput.Y,
                            trainInput.X_len,
                            trainInput.Cursor
                        ])
                        if _xx is not None:
                            _xx = np.append(_xx, _x, 1)
                        else:
                            _xx = _x
                    output(_bs, _xx, _y, _xl, ofs)
            except tf.errors.OutOfRangeError:
                sess.run(trainInput.Iterator.initializer)
    sess.close()

# xx()