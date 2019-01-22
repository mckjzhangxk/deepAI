package main.scala
/**
  * 实现了类构造参数的getter方法
  *，它将帮你实现setter和getter方法。
  样例类默认帮你实现了toString,equals，copy和hashCode等方法。
  样例类可以new, 也可以不用new
  */
case class Person(name:String,age:Int)
object HelloCaseClass {
  def main(args: Array[String]): Unit = {
    var p=new Person("zxk",99)
    println(p.age)
    println(p.name)
  }
}