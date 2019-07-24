class Meta(type):
    '''
    https://lotabout.me/2018/Understanding-Python-MetaClass/
    https://jakevdp.github.io/blog/2012/12/01/a-primer-on-python-metaclasses/
        type是 元,用于定义类的 
        type(classname,bases,namespace,**kwargs)
        其中namespace包含属性 和方法

        继承的type,修改__new__,就可以 监控 创建 类的过程,
        这时候的类 还没有 实例,你可以做任何 操作,

        1.比如传入Base的类名,为可以把它改成Base1
        2.可以检查namespace有没有你想要的 某个 方法
            当然你也可以 检测 bases里面 是不是有
    '''
    def __new__(cls, name, bases, namespace, **kwargs):
        if name !='Base' and 'bar' not in namespace:
            flag=True
            for base in bases:
                if hasattr(base,'bar'):
                    flag=False
                    break
            if flag:
                raise('bad user method')
        if not hasattr(cls,'children'):
            cls.children={}
        else:
            cls.children[name]=bases
        return super().__new__(cls, name, bases, namespace, **kwargs)
class Base(metaclass=Meta):
    def foo(self):
        return self.bar()
class Derived1(Base):
    def bar(self):
        print('run bar')
class Derived2(Base):
    def bar(self):
        print('run bar')
class Derived3(Derived2):
    pass
class Derived4(Derived3):
    pass
if __name__ == "__main__":
    d=Derived3()
    d.foo()
    print(Meta.children)
    
    print('实例Dervided3的 class 是Dervided3',d.__class__.__name__)


    print('类实例Dervided3的 class 是Meta',Derived3.__class__.__name__)
    