package main.spark.operator

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangxk on 19-1-16.
  */

object SparkOperator {
  /*
  * 算子是将一个RDD转化为其他RDD的操作,分为
  * 1.transform 算子,延迟操作
  * 2.Action 算子,触发算子
  *
  * transform:map,flatmap,sortby,sortbykey,filter,reduceByKey,sample
  * */
  def test_filter(rdd:RDD[String]):Unit={
    var filter_rdd: RDD[String] =rdd.filter(_.contains("zxk"))
    filter_rdd.foreach(println)
  }
  def test_sample(rdd: RDD[String]):Unit={
    var sample_rdd:RDD[String]=rdd.sample(false,0.5)
    sample_rdd.foreach(println)
  }
  def test_sortBy(rdd: RDD[String]):Unit={
    var sortByRDD: RDD[String] =rdd.sortBy((line)=>{
      var s: Array[String] =line.split(" ")
      s(1)
    },true)
    sortByRDD.foreach(println)
  }

  /*
  * action算子:count,collect,first,take
  * */
  def test_count(rdd: RDD[String]): Unit ={
    var cnt: Long =rdd.count()
    println(cnt)
  }

  def test_take(rdd: RDD[String]): Unit ={
    var arr: Array[String] =rdd.take(6)
    arr.foreach(println)
  }

  /*
  * 调用collection后,把worker的数据收集回送给driver
  *
  * */
  def test_collect(rdd: RDD[String])={
    var arr: Array[String] =rdd.collect()
    arr.foreach(println)
  }
  def main(args: Array[String]): Unit = {
      var conf:SparkConf=new SparkConf()
      conf.setMaster("local").setAppName("myspark")
      var sc:SparkContext=new SparkContext(conf)

      var rdd: RDD[String] =sc.textFile("./wc.txt")
//      test_filter(rdd)
//      test_sample(rdd)
//      test_sortBy(rdd)

//     test_count(rdd)
//      test_take(rdd)
      test_collect(rdd)
      sc.stop()
  }

}
