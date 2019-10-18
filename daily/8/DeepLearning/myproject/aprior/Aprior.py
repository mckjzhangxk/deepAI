from Bucket import TransActionSystem

class Result:
    def __init__(self):
        self.result={}
    def put(self,someset):
        assert len(someset)>0
        currentK=len(someset[0])
        self.result[currentK]=someset
    def print(self):
        for k,v in self.result.items():
            print(k,":",v)
class Aprior():
    '''
    usage:
        model=Aprior()
        model.set_transaction_system(system)
        model.fit(ds)
    '''
    def __init__(self):
        pass

    def set_transaction_system(self,system):
        self._query_system=system

    def _filter(self,candidate_set):
        '''
            flter the item in candidata_set that is not frequence by using member query_system,
            and return the filtered result

        '''
        assert isinstance(candidate_set,tuple) or isinstance(candidate_set,list)
        L=[item for item in candidate_set if self._query_system.isFrequence(item)]
        return L
    def _construction(self,frequence_set):
        assert isinstance(frequence_set,tuple) or isinstance(frequence_set,list)
        
        candidate_set=set()
        for i in range(0,len(frequence_set)-1):
            for j in range(i+1,len(frequence_set)):
                a=set(frequence_set[i])
                b=set(frequence_set[j])
                c=b-a
                if(len(c)==1):
                    candidate_set.add(tuple(sorted([*a,*c])))
        return list(candidate_set)

    def fit(self,dataSource):
        '''
            dataSource is a iter,every time yield a tutple that
            represent a transaction!
            also I assume dataSource have a get_uniform_item,method
            
        '''

        assert hasattr(dataSource,'get_uniform_item'),'dataSource should have method get_uniform_item'
        assert hasattr(self,'_query_system'),'transaction system must be set!'

        self.result=Result()

        C=dataSource.get_uniform_item()
        L=[]

        maxK=self._query_system.maxK()
        for _ in range(maxK):
            L=self._filter(C)
     
            if len(L)==0:break
            self.result.put(L)
            C=self._construction(L)
            

class CustomDataSource():
    def __init__(self):
        self.db=[
            ('a','b','c'),
            ('a','b','c'),
            ('a','b'),

            # ('a','b','c','d'),
            # ('a','b','c','d'),
            # ('a','b','c','d'),
            # ('a','b','d'),

            # ('c','d'),
            # ('c','d'),
            # ('d',)
    ]
    def get_uniform_item(self):
        ret=set()
        for record in self.db:
            for item in record:
                ret.add((item,))
        return list(ret)
    def __iter__(self):
        for record in self.db:
            yield record
if __name__ == "__main__":
    db=CustomDataSource()
    system=TransActionSystem([31,53,71,47])
    system.build_summary_system(db,support=3)
    # system.print()
    # print(system.isFrequence(['a','b','c']))
    model=Aprior()
    model.set_transaction_system(system)
    model.fit(db)
    model.result.print()