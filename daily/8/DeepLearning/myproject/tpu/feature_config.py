import json
import collections

if __name__ == '__main__':
    f=collections.OrderedDict()
    f['name']=''
    f['type']='int'

    obj=[f,f]

    with open('feature.json','w') as fp:
        json.dump(obj,fp,indent=1)
