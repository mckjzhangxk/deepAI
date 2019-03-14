import tensorflow as tf

vocab_path='data/vocab.en'
vocab_tb=tf.contrib.lookup.index_table_from_file(vocab_path,default_value=0)
reverse_vocab_tb=tf.contrib.lookup.index_to_string_table_from_file(vocab_path,default_value='<unk>')


Y=reverse_vocab_tb.lookup(tf.constant([0,1,2,200000],dtype=tf.int64))


input_file='data/train.en'

dbset=tf.data.TextLineDataset(input_file)

dbset=dbset.map(lambda line:tf.string_split([line]).values)

dbset=dbset.map(lambda x:vocab_tb.lookup(x))
dbset=dbset.padded_batch(128,
                     padded_shapes=(tf.TensorShape([None])),
                     padding_values=(tf.constant(0,dtype=tf.int64)))


iterator=dbset.make_initializable_iterator()
X=iterator.get_next()
YY=reverse_vocab_tb.lookup(X)
sess=tf.Session()
sess.run(tf.tables_initializer())
sess.run(iterator.initializer)

for i in range(1):
    # print(sess.run(X))
    print(sess.run(YY))