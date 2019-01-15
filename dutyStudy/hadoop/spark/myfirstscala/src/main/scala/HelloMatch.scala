/**
  * Created by zhangxk on 19-1-15.
  */
object HelloMatch {
  def main(args: Array[String]): Unit = {
    val input=(10,22,2.2,"Hello")
    val iter:Iterator[Any]=input.productIterator
    while(iter.hasNext){
      matchInput(iter.next())
    }
  }
  def matchInput(p:Any)={
    p match {
      case "Hello"=>println("say Hello")
      case i:Double=>println("input is double")
      case 10=>println("age is 10")
      case _=>println("default")
    }

  }
}
