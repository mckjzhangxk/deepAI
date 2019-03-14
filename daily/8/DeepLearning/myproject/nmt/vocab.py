import tensorflow as tf


src_vocab_path='vocab.en'

tb1=tf.contrib.lookup.index_table_from_file(src_vocab_path)
ts=tb1.lookup(tf.constant(['climate','glimpse']))
print(dir(tb1))
sess=tf.Session()
print(tf.tables_initializer())
sess.run(tf.tables_initializer())
print(sess.run(ts))