import tensorflow as tf
import tensorflow.layers as layers
import numpy as np
import functools
import tensorflow as tf
from tensorflow.python.client import timeline

x = tf.random_normal([1000, 1000])
y = tf.random_normal([1000, 1000])
z=tf.random_normal([1000, 1000])
res = tf.matmul(x, y)
b=x+y

# Run the graph with full trace option
# with tf.Session() as sess:
#     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#     run_metadata = tf.RunMetadata()
#     for i in range(100):
#         sess.run([res,b], options=run_options, run_metadata=run_metadata)
#
#     # Create the Timeline object, and write it to a json
#     tl = timeline.Timeline(run_metadata.step_stats)
#     ctf = tl.generate_chrome_trace_format()
#     with open('timeline.json', 'w') as f:
#         f.write(ctf)

# # functools.partial(lambda :layers.Conv2D(32,(3,3),(1,1),'same',use_bias=False))
with tf.device('/gpu:0'):
    N=512
    X=tf.random_normal((N,256,256,3))
    conv=functools.partial(lambda f:layers.Conv2D(f,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=False))
    # print(conv.kernel)

    def showMemory(d):
        print('%fM'%(4*d/1024/1024))
    memory=0
    for i in range(6):
        X=conv((i+1)*32)(X)
        size=np.prod(X.get_shape()).value

        showMemory(size)
        memory+=size
    showMemory(memory)
    loss=tf.reduce_mean(tf.reduce_sum(X**2,-1))
    solver=tf.train.GradientDescentOptimizer(0.0)
    op=solver.minimize(loss)

    run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction =1.0

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer(),options=run_opts,run_metadata=run_metadata)
        with open('timeline.json', 'w') as f:
            for i in range(10):
                print('xxx')
                sess.run(op)
                print('ss')
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                f.write(ctf)