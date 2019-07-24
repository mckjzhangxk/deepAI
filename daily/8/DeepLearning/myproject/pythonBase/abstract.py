from abc import ABCMeta,abstractmethod
# https://blog.louie.lu/2017/07/28/%E4%BD%A0%E6%89%80%E4%B8%8D%E7%9F%A5%E9%81%93%E7%9A%84-python-%E6%A8%99%E6%BA%96%E5%87%BD%E5%BC%8F%E5%BA%AB%E7%94%A8%E6%B3%95-03-abc/

class Animal(metaclass=ABCMeta):
    def __init__(self):
        self._successes='myzxk'
    @abstractmethod
    def walk():pass
class Dog(Animal):
    def walk(self):
        print(self._successes,'walk')

if __name__ == "__main__":
    #不能被实例化
    Dog().walk()

    print('Dog is Animal',issubclass(Dog,Animal))

    print('list is Animal',issubclass(list,Animal))
    Animal.register(list)
    print('After register,list is Animal',issubclass(list,Animal))