package main.scala
/**
  * Created by zhangxk on 19-1-14.
  */
object HelloList {
  def  testArray():Unit={
    var xx: Array[Nothing] =new Array(3)

    var arr=Array(10,20,30,"sdsasd")
    arr.foreach((x)=>{
      println(x)
    })
  }
  def  testList():Unit={
    var ls=List("hello world","hello zhangxk","hello zhuxin")

    var rsmap=ls.map((x)=>{
      x.split(" ")
    })

    var rsflatmap=ls.flatMap((x)=>{
      x.split(" ")
    })

    rsmap.foreach(x=>{
      println("ok")
      x.foreach(println)
    })

    rsflatmap.foreach(println)
  }
  def testMap():Unit={
    var map=Map("key1"->10,"key2"->3)
    var keys=map.keys
    var vals=map.values

    keys.foreach(println)
    vals.foreach(println)

    map.foreach(println)
  }
  def  testTuple():Unit={
    var b=(20,3,10,"xx")
    println(b._1)
    println(b._2)
    println(b._3)
    println(b._4)
  }
  def main(args: Array[String]): Unit = {
    testTuple()
  }
}
