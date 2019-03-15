import tensorflow as tf
from collections import defaultdict

a=tf.get_variable('nmt/nmt/embeding/decoder/Adam',shape=(17191, 128))

ckt_path='/home/zxk/projects/deepAI/daily/8/DeepLearning/myproject/nmt/result/baseModel'
ckt_state=tf.train.get_checkpoint_state(ckt_path)
if ckt_state:
    model_paths=ckt_state.all_model_checkpoint_paths



    vars,num=defaultdict(float),len(model_paths)
    for p in model_paths:
        vars_info=tf.contrib.framework.list_variables(p)
        reader = tf.contrib.framework.load_checkpoint(p)

        for varname,varshape in vars_info:
            vars[varname]+=reader.get_tensor(varname)/num
    assign_op=[]
    for name,value in vars.items():
        try:
            with tf.variable_scope('',reuse=True):
                x=tf.get_variable(name)
                print(value.shape)
                assert x.shape.as_list()==list(value.shape),'variable shape must compatiable'
                assign_op.append(tf.assign(x,value))

        except ValueError:
            pass
    print(assign_op)

    with tf.Session() as sess:
        sess.run(assign_op)
