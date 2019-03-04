a=[None,None,None,None,None,3,None,None,None,None,None]


def get_valid_input(series):
    '''
    series是一个list,表示时间序列,但是这个序列头尾
    可能会被大量的None填充,这些视为无效输入,因为可能
    这段实际根本就不存在链接通信.
    
    返回:l=serise[a,b],l[0]和l[-1]不为None,
    l的长度是有效序列的长度
    :param series: 
    :return: 
    '''
    s=0
    for k in series:
        if k is None:
            s+=1
        else:
            break
    e=len(a)
    for k in reversed(series):
        if k is None:
            e-=1
        else:break
    assert e>s,'at least one element is not None'
    return series[s:e]