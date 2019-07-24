
class A():
    def instant_method(self):
        print('instant_method')
    @classmethod
    def class_method(cls):
        print('class method')
    @staticmethod
    def static_method():
        print('static method')
if __name__ == "__main__":
    print('instant 方法')
    print(A.instant_method)
    a=A()
    print(a.instant_method)

    print('class,把方法与 类对象 绑定')
    print(A.class_method)
    print(a.class_method)
    #与一般函数没区别
    print('static与一般函数没区别')
    print(A.static_method)
    print(a.static_method)