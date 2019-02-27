import tensorflow as tf
class RNNState():
    def __init__(self,cell,Batch,batch_size):
        base_state=cell.zero_state(batch_size,tf.float32)

        if isinstance(base_state,tuple):
            for state in base_state:
                if isinstance(state,tf.nn.rnn_cell.LSTMStateTuple):
                    ndim=state.c.get_shape()[1]
                    

def _singleRNNCell(type,ndims,dropout,forget_bias,residual):
    cell=None
    if type=='lstm':
        cell=tf.contrib.rnn.BasicLSTMCell(ndims,forget_bias)
    elif type=='gru':
        cell=tf.contrib.rnn.GRUCell(ndims)
    else:raise ValueError('unknown cell type %s'%type)
    if dropout>0.0:
        cell=tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=1-dropout)
    if residual:
        cell=tf.contrib.rnn.ResidualWrapper(cell,None)
    return cell
def _cells(type,ndims,number_layers,dropout,forget_bias,residual):
    cells=[_singleRNNCell(type,ndims,dropout,forget_bias,residual)
                        for n in range(number_layers)]
    return cells
def buildRNNCell(type,ndims,number_layers,dropout,forget_bias,residual):
    cells=_cells(type,ndims,number_layers,dropout,forget_bias,residual)
    if len(cells)==1:return cells[0]
    return tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)


def _initial_state(cell_type,BATCH,ndims,num_layers,batch):
    states=[]
    if cell_type=='lstm':
        for n in range(num_layers):
            c0=tf.get_variable('%s%d_c0'%(cell_type,n),shape=[BATCH,ndims]
                               ,trainable=False,initializer=tf.zeros_initializer)
            h0=tf.get_variable('%s%d_h0' % (cell_type,n), shape=[BATCH, ndims],
                               trainable=False,initializer=tf.zeros_initializer)
            state=tf.nn.rnn_cell.LSTMStateTuple(c0[:batch],h0[:batch])
            states.append(state)
    return tuple(states)

def _clone_state():pass

BATCH,T,D=64,60,4
N=None
X=tf.placeholder(dtype=tf.float32,shape=[N,T,D])

cell=buildRNNCell('lstm',128,2,0.2,1.0,False)
RNNState(cell,2)
init_state=cell.zero_state(tf.shape(X)[0],dtype=tf.float32)




myinit_state=_initial_state('lstm',BATCH,128,2,tf.shape(X)[0])
init_state=myinit_state

for t in range(2):
    output,init_state=cell(X[:,t],init_state)
    print(init_state)
print(myinit_state)
final_state=init_state
