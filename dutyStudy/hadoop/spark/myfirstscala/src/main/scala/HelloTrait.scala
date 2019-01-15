/**
  * Created by zhangxk on 19-1-15.
  */

trait IsEqual{
  def isEqual(p:Any):Boolean;
  def isNotEqual(p:Any): Boolean ={
    return !isEqual(p)
  }
}
class Point(xx:Int,yy:Int) extends IsEqual{
  var x=xx
  var y=yy

  override def isEqual(p: Any): Boolean = {
    return p.isInstanceOf[Point] && p.asInstanceOf[Point].x==x
  }
}
object HelloTrait {
  def main(args: Array[String]): Unit = {
    var p1=new Point(10,20);
    var p2=new Point(11,30);
    println(p1.isNotEqual(p2))
  }

}
