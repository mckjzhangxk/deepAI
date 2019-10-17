def getBit(value,index):
    return (value&(1<<index))>>index
def setBit(value,index):
    value=value|(1<<index)
    return value
def clearBit(value,index):
    value=value&(~(1<<index))
    return value

def subSet_K(items,k):
    assert len(items)>=k
    m={}
    m[tuple()]=items      
            
    for _ in range(k):
        newKeys,newValues=[],[]
        for key,value in m.items():
            newKeys.extend([key+(v,) for v in value])
            newValues.extend([value[j+1:] for j in range(0,len(value))])
        m.clear()                
        for newK,newV in zip(newKeys,newValues):
            m[newK]=newV
    return list(m.keys())
class Bitmap():
    def __init__(self,maxSize):
        self._value=0
        self._size=maxSize
        self._ones=set()

    def getBit(self,index):
        assert index>=0 and index<self._size
        return getBit(self._value,index)
    def setBit(self,index):
        assert index>=0 and index<self._size
        if index not in self._ones:
            self._value=setBit(self._value,index)
            self._ones.add(index)

    def clearBit(self,index):
        assert index>=0 and index<self._size
        if index in self._ones:
            self._value=clearBit(self._value,index)
            self._ones.remove(index)
    def __iter__(self):
        for i in range(self._size):
            yield self.getBit(i)
    
    def __str__(self):
        ret='#total bits %d'%self._size

        for v in self._ones:
            ret+='\n%d:%d'%(v,self.getBit(v))
        return ret
    def __repr__(self):
        return self.__str__()


# print(hash((4,2)))
# B=Bitmap(30)
# B.setBit(22)

# B.clearBit(22)
# B.setBit(23)
# B.setBit(29)

# for i,c in enumerate(B):
#     print(i,c)
# # B.setBit(30000)
# # B.clearBit(30000)
# print(B)