import numpy as np
import tensorflow as tf
from tensorpack.utils import logger

def restore_from_npz(sess,filename):
    def get_op_tensor_name(name):
        """
        Will automatically determine if ``name`` is a tensor name (ends with ':x')
        or a op name.
        If it is an op name, the corresponding tensor name is assumed to be ``op_name + ':0'``.

        Args:
            name(str): name of an op or a tensor
        Returns:
            tuple: (op_name, tensor_name)
        """
        if len(name) >= 3 and name[-2] == ':':
            return name[:-2], name
        else:
            return name, name + ':0'

    def is_training_name(name):
        """
        **Guess** if this variable is only used in training.
        Only used internally to avoid too many logging. Do not use it.
        """
        # TODO: maybe simply check against TRAINABLE_VARIABLES and MODEL_VARIABLES?
        # TODO or use get_slot_names()
        name = get_op_tensor_name(name)[0]
        if name.endswith('/Adam') or name.endswith('/Adam_1'):
            return True
        if name.endswith('/Momentum'):
            return True
        if name.endswith('/Adadelta') or name.endswith('/Adadelta_1'):
            return True
        if name.endswith('/RMSProp') or name.endswith('/RMSProp_1'):
            return True
        if name.endswith('/Adagrad'):
            return True
        if name.startswith('EMA/') or '/EMA/' in name:  # all the moving average summaries
            return True
        if name.startswith('AccumGrad') or name.endswith('/AccumGrad'):
            return True
        if name.startswith('apply_gradients'):
            return True
        return False


    dict_param = dict(np.load(filename))
    param_names = set([get_op_tensor_name(n)[1] for n in dict_param.keys()])

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    variable_names = set([k.name for k in variables])
    variables_dict={v.name:v for v in variables}


    intersect = variable_names & param_names


    mismatch = []
    for k in sorted(variable_names - param_names):
        if not is_training_name(k):
            mismatch.append(k)
    if len(mismatch)>0:
        logger.warn('Miss follow variables from dict:{}'.format(', '.join(mismatch)))

    logger.info("Restoring {} variables from dict ...".format(len(intersect)))
    #now start initial var
    prms={name:(variables_dict[name],dict_param[name]) for name in dict_param if name in  intersect}
    fetches = []
    feeds = {}
    for name, (var,value) in prms.items():
        fetches.append(var.initializer)
        # This is the implementation of `var.load`
        feeds[var.initializer.inputs[1]] = value
    sess.run(fetches, feed_dict=feeds)

