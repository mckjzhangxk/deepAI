class Package():
    def __init__(self, srcip, srcport, destip, destport, type, upcount, upsize, ts):
        self.srcip=srcip
        self.srcport=srcport
        self.destip=destip
        self.destport=destport
        self.type=type

        self.upcount=int(upcount)
        self.upsize=int(upsize)

        self.downcount=0
        self.downsize=0

        self.label=-1
        self.ts=ts
    @property
    def up_rate(self):
        return 1.0*self.upsize/self.upcount if self.upcount>0 else 0.0
    @property
    def down_rate(self):
        return 1.0*self.downsize / self.downcount if self.downcount > 0 else 0.0
    @property
    def signature(self):
        return self.srcip+':'+self.srcport+'->'+self.destip+':'+self.destport+'_'+self.type
    @property
    def signature_connection(self):
        if (self.srcip+':'+self.srcport)<(self.destip+':'+self.destport):
            return self.srcip+':'+self.srcport+'<->'+self.destip+':'+self.destport+'_'+self.type
        else:
            return self.destip+':'+self.destport+'<->'+self.srcip+':'+self.srcport+'_'+self.type
    def __str__(self):
        _ret='****************************************************************\n'
        _ret+=self.srcip+':'+self.srcport+':'+self.destip+':'+self.destport+',type:'+self.type+'\n'
        _ret+='upcount:'+str(self.upcount)+',upsize:'+str(self.upsize)+',uprate:'+str(self.up_rate)+'\n'
        _ret+='downcount:' + str(self.downcount) + ',downsize:' + str(self.downsize) + ',downrate:' + str(self.down_rate)+'\n'
        return _ret
    def set_downloadinfo(self, other):
        assert isinstance(other,Package),'other must be Package object'
        assert self.signature_connection == other.signature_connection, 'not same connect'
        self.downcount=other.upcount
        self.downsize=other.upsize

class Package_FreeGate(Package):
    @property
    def signature(self):
        return self.srcip+'->'+self.destip+'_'+self.type
    @property
    def signature_connection(self):
        if (self.srcip)<(self.destip):
            return self.srcip+'<->'+self.destip+'_'+self.type
        else:
            return self.destip+'<->'+self.srcip+'_'+self.type