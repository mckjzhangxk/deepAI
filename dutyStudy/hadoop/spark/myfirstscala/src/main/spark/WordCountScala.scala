import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangxk on 19-1-15.
  */
object WordCountScala {
  def main(args: Array[String]): Unit = {
    /*
    * configiure 保存
    * 1.运行模式:
    *   local,standalone,yarn,...
    * 2.应用名称
    * 3.资源分配...conf.set
    * */
    var conf=new SparkConf()
    conf.setMaster("local");
    conf.setAppName("scala_WC");


    var sc=new SparkContext(conf)
    //读取本机文件,然后返回一个RDD
    var lines: RDD[String] =sc.textFile("./wc.txt")

    //flatmap 是一对多映射
    var words: RDD[String] =lines.flatMap((l:String)=>{
      l.split(" ")
    })
    //map是一对一,返回单词,1元祖
    var wc: RDD[(String, Int)] =words.map({(_,1)})

    //返回单词的统计,但是没有排序
    var reduce: RDD[(String, Int)] =wc.reduceByKey({_+_})

    reduce=reduce.sortBy((x:Tuple2[String,Int])=>{
      x._2
    })
    reduce.foreach(println)

//    var rdd1: RDD[(Int, String)] =reduce.map((x)=>{
//        x.swap
//    })
//
//    rdd1=rdd1.sortByKey()
//    var rdd2=rdd1.map({_.swap})
//
//
//    rdd2.foreach(println)

  }
}
