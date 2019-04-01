import tensorflow as tf
g=tf.Graph()
with g.as_default():
    graph_def=tf.GraphDef()
    fs=tf.gfile.GFile('g1.pb','rb')
    graph_def.ParseFromString(fs.read())
    xx=tf.import_graph_def(graph_def,return_elements=['Placeholder:0','scope1/mul:0'])
    print(xx)

    graph_def=tf.GraphDef()
    fs=tf.gfile.GFile('g2.pb','rb')
    graph_def.ParseFromString(fs.read())
    tf.import_graph_def(graph_def)

    # ops=g.get_operations()
    # for op in ops:
    #     print(op.name)
    # print(g.get_tensor_by_name('import/Placeholder:0'))
    # print(g.get_tensor_by_name('import/scope1/mul:0'))
    # print(g.get_tensor_by_name('import_1/scope2/mul:0'))