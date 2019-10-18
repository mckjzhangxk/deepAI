from utils import Bitmap,subSet_K

class Bucket():
    '''
    usage:
    
    A).
        bucket=Bucket(buckSize=71,K=2)
        ...
        buck.add(item)
        ...
        bucket.finalize()

    B)
        bucket.isFrequence(queryItem)
    '''
    def __init__(self,buckSize,K):
        '''
            K mean this Bucket only handle K-itemset
            support:if a block's count greate than support,this a frequence bucket
        '''
        self._bitmap=Bitmap(buckSize)
        self._K=K
        self._buckSize=buckSize
        self._mbuckItems=[0]*buckSize
        
        
    def add(self,item):
        '''
            map item to bucket, update bucket count,if your have call finalize method before,
            then also update summary bitmap(online statis)
        '''
        assert isinstance(item,tuple) or isinstance(item,tuple)
        assert len(item)==self._K
        
        
        index=self.__encode__(item)
        if self._bitmap.getBit(index)==0:
            self._mbuckItems[index]+=1
            if hasattr(self,'_support') and self._mbuckItems[index]>=self._support:
                self._bitmap.setBit(index)
    def finalize(self,support):
        '''
            this method summary buckets' information to a bitmap,
            should be called after you hash all item's to its bucket
        '''
        self._support=support
        for i,count in enumerate(self._mbuckItems):
            if count>=self._support:
                self._bitmap.setBit(i)
    def isFrequence(self,item):
        '''
            return true if item's bucket have support greater than support
        '''
        assert isinstance(item,tuple) or isinstance(item,tuple)
        assert len(item)==self._K
        index=self.__encode__(item)
        
        return self._bitmap.getBit(index)==1
    def __encode__(self,item):
        item=tuple(sorted(item))
        return hash(item)%self._buckSize

    def __str__(self):
        ret='#total buckets %d'%self._buckSize
        ret+='\n frequence set'
        ret+='{'
        for i,c in enumerate(self._bitmap):
            if c==1:
                ret+=str(i)+','
        ret+='}'
        return ret
    def __repr__(self):
        return self.__str__()

class TransActionSystem():
    def __init__(self,buckSizes):
        '''
            buckSizes:list of integer,indicate the size of k-itemset buckets

            usage:
                system=TransActionSystem(buckSizes=[11,21,31,17])
                system.build_summary_system(ds,support=232)
        '''

        self._bucket_set={}

        for k in range(1,len(buckSizes)+1):
            self._bucket_set[k]=Bucket(buckSizes[k-1],k)
    def _subSet_(self,items):
        '''
            items is N-tuple
            return a dict with key=1,2,3,value is list of k-itemset
        '''
        
        
        assert isinstance(items,list) or isinstance(items,tuple)    
        ret={}
        for k in range(1,len(self._bucket_set)+1):
            if k<=len(items):
                ret[k]=subSet_K(items,k)
            else:
                ret[k]=[]
        return ret
        
    def build_summary_system(self,dataSource,support):
        '''
            dataSource is a iterator,every time when call it,will
            return a tuple represent a particular transaction

            Update all buckset count  by go through all dataSource,
            
            note:don't forget call bucket.finalize to convert count information
            to bitmap!
        '''
        import tqdm
        for i,data in tqdm.tqdm(enumerate(dataSource)):
            subset=self._subSet_(data)
            for k in range(1,len(self._bucket_set)+1):
                k_items=subset[k]
                k_bucket=self._bucket_set[k]
                for item in k_items:
                    k_bucket.add(item)

        for buk in self._bucket_set.values():
            buk.finalize(support)

    def isFrequence(self,item):
        '''
        check whether item is frequence,
        
        item is frequence iff all its subset is frequence,
        this method check all it's bucket
        '''

        assert isinstance(item,tuple) or isinstance(item,list)
        if len(item)>len(self._bucket_set):
            return False
        subsets=self._subSet_(item)

        for k in range(len(item),0,-1):
            k_items=subsets[k]
            for subitem in k_items:
                if not self._bucket_set[k].isFrequence(subitem):
                    return False
        return True
    def maxK(self):
        return len(self._bucket_set)
    def print(self):
        for k,v in self._bucket_set.items():
            print(k,v)
if __name__ == "__main__":
    bucket=Bucket(71,2)
    for i in range(5):
        bucket.add((1,2))
    for i in range((3)):
        bucket.add((20,9))
    for i in range((3)):
        bucket.add((2,9))

    bucket.finalize(support=3)
    print(bucket)


    print(bucket.isFrequence((2,9)))
    dataSource=[
        ('a','b','c'),
        ('a','b','c'),
        ('a','b','c'),

        ('a','b','c','d'),
        ('a','b','c','d'),
        ('a','b','c','d'),
        ('a','b','d'),

        ('c','d'),
        ('c','d'),
        ('d',)
    ]
    # dataSource=[
    #     ('a',),('a',),('a',)
    # ]
    system=TransActionSystem([71,71,19,22])
    system.build_summary_system(dataSource,support=3)
    system.print()
    print(system.isFrequence(('a','b','c','d')))