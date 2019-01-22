package main.spark.app

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangxk on 19-1-16.
  */
object SparkPI {
  def main(args: Array[String]): Unit = {
    var conf=new SparkConf()
    conf.setMaster("local").setAppName("MySparkPI")
    var sc=new SparkContext(conf)

    var N=1000000
    var partitions=6

    var trails=1 to N
    var tasks: RDD[Int] =sc.parallelize(trails,partitions)
    var rdd: RDD[Int] =tasks.map((_)=>{
      var x=math.random
      var y=math.random

      if( math.pow((x-0.5),2)+math.pow((y-0.5),2) <1) 1
      else 0
    })

    var result: Int =rdd.reduce({_+_})

    var pi=1.0*result/N
    println(pi)
  }
}
