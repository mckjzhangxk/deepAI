/**
  * Created by zhangxk on 19-1-14.
  */
object HelloList {
  def main(args: Array[String]): Unit = {
//    var rs=new Array[Int](3);
//    rs(0)=1
//    rs(1)=2
//
//    rs.foreach(println)

    var ls=List("hello world","hello zhangxk","hello zhuxin");

//    var rs=ls.map(x=>{
//      x.split(" ");
//    })
//
//    rs.foreach(s=>{
//      println("xxxxxxxxxxxxxxxx")
//      s.foreach(println)
//    })

    var rs=ls.flatMap(x=>{
      x.split(" ")
    })
    rs.foreach(println)
  }
}
