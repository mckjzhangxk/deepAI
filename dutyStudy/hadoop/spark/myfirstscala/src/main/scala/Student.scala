/**
  * Created by zhangxk on 19-1-14.
  */
class Student (xname:String,xage:Int){
  var name=xname;
  var age=xage;
  var sex="M";

  def this(xname:String,xage:Int,xsex:String)={
    this(xname,xage);
    this.sex=xsex;
  }
  override def toString():String={
    return this.name+"-"+this.age+"-"+this.sex;
  }
}
