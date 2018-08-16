import numpy as np
import tensorflow as tf
from scipy.misc import imread,imresize
from nst_utils import *

def _compute_content_cost(G_C, A_C):
    '''

    :param A_C:(H,W,C) tenspr,generated feature
    :param G_C:(H,W,C) tensor,content
    :return:  l2 distance between A_S,G_S
    '''
    H,W,C=G_C.get_shape().as_list()
    return tf.reduce_sum((G_C-A_C)**2)/(2*H*W*C)
def __gramMatrix(A):
    '''

    :param A:[C,H*W]
    :return:
    '''
    return tf.matmul(A,A,transpose_b=True)
def _compute_one_style_cost(G_S,A_S):
    '''

    :param G_S:(H,W,C) tensor,,generated style feature
    :param A_S: (H,W,C) one layer style tensor(constant)
    :return: l2 distance between Gram(G_s) and Gram (A_s)
    '''
    H,W,C=G_S.get_shape().as_list()

    #(H,W,C)-->(C,H,W)-->(C,HxW)
    G_S=tf.reshape(tf.transpose(G_S,perm=[2,0,1]),[C,H*W])
    A_S=tf.reshape(tf.transpose(A_S,perm=[2,0,1]),[C,H*W])

    Gram_G_S=__gramMatrix(G_S)
    Gram_A_S=__gramMatrix(A_S)

    return tf.reduce_sum((Gram_G_S - Gram_A_S) ** 2) / ((2 * H * W * C)**2)
def _compute_style_cost(layers):
    '''

    :param layers: a list, layers[i]=(G_S,A_S,coff_i)
    :return: loss=coff_1*L1(G_S,A_S_1)+coff_2*L1(G_S,A_S_2)+...coff_n*L1(G_S,A_S_n)
    '''

    loss=0

    for G_S,A_S,coff in layers:
        loss+=coff*_compute_one_style_cost(G_S,A_S)
    return loss
def compute_loss(G_C,A_C,layers,alpha,beta):
    '''

    :param G_C: (H,W,C) a generated content tensor
    :param A_C: (H,W,C) a content tensor(constant)
    :param layers: list entry is (G_S,A_S,coff)
    :return: total loss
    '''
    J_content=_compute_content_cost(G_C,A_C)
    J_style = _compute_style_cost(layers)
    J=alpha*J_content+beta*J_style
    return J_content,J_style,J

def _initial(model):
    '''

    :param model:VGG network,model[layer_name] get a activation!
    :return: G_C,A_C,layers list of all (G_S,A_S,coff)
    '''


    G_CONTENT=model[CONFIG.CONTENT_LAYER][0] #content layer,(1,H,W,C)-->(H,W,C)
    G_STYLE=[] #style layer
    for layername,coff in CONFIG.STYLE_LAYERS:
        G_STYLE.append(model[layername][0])
    '''
    now compute the feature representation of content Image,
    1.feed content Image into network
    2.eval the content feature
    then eval the all feature representation of style Image:
    1.feed style Image into network
    2.eval the style feature
    '''

    img_content=reshape_and_normalize_image(imread(CONFIG.CONTENT_IMAGE))
    img_style = reshape_and_normalize_image(imread(CONFIG.STYLE_IMAGE))

    with tf.Session() as sess:
        # content image eval
        sess.run(model['input'].assign(img_content))
        A_CONTENT=sess.run(G_CONTENT)
        #style image eval
        sess.run(model['input'].assign(img_style))
        A_STYLE=sess.run(G_STYLE)

    style_layers=[]
    for i in range(len(CONFIG.STYLE_LAYERS)):
        c=(G_STYLE[i],A_STYLE[i],CONFIG.STYLE_LAYERS[i][1])
        style_layers.append(c)

    return G_CONTENT,A_CONTENT,style_layers
def model_nn(model,numItera=2000):
    '''

    :param model:VGG model
    :return:
    '''

    input_img=generate_noise_image(imread(CONFIG.CONTENT_IMAGE))

    #define style network
    G_CONTENT, A_CONTENT, style_layers=_initial(model)
    J_content, J_style, J=compute_loss(G_CONTENT,A_CONTENT,style_layers, CONFIG.alpha,CONFIG.beta)
    train_step=tf.train.AdamOptimizer(5.0).minimize(J)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        gen_image=model['input']  #input tensor(like a placeholder)
        sess.run(gen_image.assign(input_img))
        for i in range(numItera):
            _,_jc,_js,_j=sess.run([train_step,J_content,J_style,J])
            if i %100==0:
                print('content loss %.2f,style loss %.2f,loss %.2f'%(_jc,_js,_j))
                _gen_image=sess.run(gen_image)
                save_image(CONFIG.OUTPUT_DIR + str(i) + ".png", _gen_image)


if __name__ == '__main__':
    tf.reset_default_graph()
    model = load_vgg_model(CONFIG.VGG_MODEL)
    model_nn(model)