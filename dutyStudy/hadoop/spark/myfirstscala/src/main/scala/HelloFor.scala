/**
  * Created by zhangxk on 19-1-14.
  */
object HelloFor {
  def main(args: Array[String]): Unit = {

    for(i <-1 to 3)
      for (j <- 2 to 4){
        println(i,j)
      }
//    for (i<- 1 to 10){
//      println(i);
//    }


    //生成集合
    var mylist=for (i<- 1 to 1000 if i%10==0) yield i;
    println(mylist);

    //箭头函数(x)=>{}
    mylist.foreach((x)=>{
      println("item:",x)
    })

  }
}
