package main.scala

/**
  * Created by zhangxk on 19-1-14.
  */
object HelloString {
  def main(args: Array[String]): Unit = {
    var sb=new StringBuffer()
    sb.append("beijing\n")
    sb.append("shangxuetang")
    println(sb)

    var a="ABC";
    var b="abc";
    println(a.compareToIgnoreCase(b))
    println(a.indexOf("C"))
  }
}
