package main.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangxk on 19-1-28.
  */
object HomeWork {
  def main(args: Array[String]): Unit = {
    var conf=new SparkConf()
    conf.setAppName("spark").setMaster("local")
    var sc=new SparkContext(conf)
    var rdd: RDD[String] =sc.textFile("/home/zhangxk/pvuvdata")
    var rdd_record: RDD[(String, Int)] =rdd.map((line)=>{
      var sps=line.split("\t")
      (sps(5),1)
    })
    var rdd_reduce: RDD[(String, Int)] =rdd_record.reduceByKey((a, b)=>{a+b})
    var rdd_reduce_sort: RDD[(String, Int)] =rdd_reduce.sortBy(_._2,false)

    rdd_reduce_sort.foreach(println)
    sc.stop()
  }
}
