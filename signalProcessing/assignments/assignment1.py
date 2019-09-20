import numpy as np

def HannWindow(N):
    '''
    w[n]= 0.5*c*(1-cos(2pin/N-1) )
    c is normlize factor factor,such w[n].T*w[n]=N-1
    :param N: 
    :return: 
    '''

    n=np.arange(0,N)
    w=0.5*(1-np.cos(2*np.pi*n/(N-1)))
    normsquare=w.dot(w)
    c=np.sqrt((N-1)/normsquare)
    w=w*c

    return w
def scaled_fft_db(x):
    """ ASSIGNMENT 1:
        a) Compute a 512-point Hann window and use it to weigh the input data.
        b) Compute the DFT of the weighed input, take the magnitude in dBs and
        normalize so that the maximum value is 96dB.
        c) Return the first 257 values of the normalized spectrum

        Arguments:
        x: 512-point input buffer.

        Returns:
        first 257 points of the normalized spectrum, in dBs
        
        
        
    """

    y=HannWindow(512)*x

    Y=np.fft.fft(y)/len(y)

    Ymag=np.abs(Y)[0:257]

    db_mag=np.where(Ymag>1e-4,20*np.log10(Ymag),-100)

    '''
    I make a mistake as this normal operation,
    
    I simple multi db_mag by 96/db_mag.max(),max is normal to 
    96db,but something wrong,I need to figure it out!
    '''
    normalFactor=96-20*np.log10(Ymag.max())
    ret=db_mag+normalFactor

    return ret[0:257]
if __name__ == '__main__':
    a=np.arange(10)
    # a[3]=0
    # c=np.where(a>0,a*2,100)
    # print(c)
    # xx= np.hanning(512)
    # print(xx.dot(xx))
    # yy=HannWindow(512)
    # print(yy.dot(yy))
    # print(np.log10(10))