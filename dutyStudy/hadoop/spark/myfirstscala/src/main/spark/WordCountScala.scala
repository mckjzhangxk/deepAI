package main.spark

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
    * 避免手生,重写wc,第30行的时候忘记赋值sort结果导致 浪费时间
    * */
    var conf=new SparkConf()
    conf.setAppName("wc").setMaster("local")
    var sc=new SparkContext(conf)

    var line_rdd=sc.textFile("./wc.txt")
    var word_rdd=line_rdd.flatMap(_.split(" "))
    var wr_rdd=word_rdd.map((_,1))

    var rd_rdd=wr_rdd.reduceByKey(_+_)

//    var rd_rdd1: RDD[(Int, String)] =rd_rdd.map(_.swap)
//    rd_rdd1=rd_rdd1.sortByKey()
//    rd_rdd=rd_rdd1.map(_.swap)
    rd_rdd=rd_rdd.sortBy(_._2,false)
    rd_rdd.foreach(println)
    sc.stop()

  }
}
