import scala.actors.Actor

/**
  * Created by zhangxk on 19-1-15.
  */
case class Message(act:Actor,msg:String );

class  MyActor1 extends Actor{
  override def act(): Unit = {
    while (true){
      receive{
        case x:Message=>{
          println("get from actor2,msg:"+x.msg)
          x.act ! Message(this,"hello")
        }

        case _=>println("default")
      }
    }
  }
}
class  MyActor2 extends Actor{
  override def act(): Unit = {
    while (true){
      receive{
        case x:Message=>{
          println("get from actor1,msg:"+x.msg)
          x.act ! Message(this,"yes")
        }

        case _=>println("default")
      }
    }
  }
}

object HelloActor {
  def main(args: Array[String]): Unit = {
    var actor1=new MyActor1
    var actor2=new MyActor2
    actor1.start()
    actor2.start()

    actor1 !  Message(actor2,"hello")
  }
}
