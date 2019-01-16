package main.scala
/**
  * Created by zhangxk on 19-1-14.
  */
object HelloFunction {
  def fft(v:Double*):Double={

    var ret=0.0;
    for(s<-v){
      ret+=s
    }
    return ret;
  }
  def conv2d(out:Int,stride:Int=1,filtersize:Int=3,padding:String="SAME"): () => Unit ={
    return ()=>{
      println("outsize="+out+",fileter size="+filtersize+"  stride="+stride+",padding="+padding)
    }

  }
  def fun1(v:Double=5,power:Int=2):Double={
      return v*power;
  }
  def f(v:Int):Int={
    return v*2
  }
  def main(args: Array[String]): Unit = {
//    println(fft(1.0,2.0,3.0,4.0));
//    println(fun1(power = 3))

//    conv2d(10,2,5,"Valid")()

    var a=1 to 10
    var b=a.map((x:Int)=>{
      x*2
    })

    b.foreach(println)
  }

}
