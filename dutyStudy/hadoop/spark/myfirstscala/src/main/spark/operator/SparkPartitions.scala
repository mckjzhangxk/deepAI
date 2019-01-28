package main.spark.operator

import java.util.ArrayList

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ListBuffer

/**
  * Created by zhangxk on 19-1-28.
  */
object SparkPartitions {
  def mapPartitonWithIndex(sc:SparkContext):Unit={
    var data=Array("A","B","C","D","E","F","G","H","I","J","K","L","M","N")
    var rdd1: RDD[String] =sc.parallelize(data,3)

    var rdd2: RDD[String] =rdd1.mapPartitionsWithIndex((idx, iter)=>{
      var ret=new ListBuffer[String]()
      while (iter.hasNext){
        var aa=iter.next()
        ret.+=:(idx+":"+aa)
      }
      ret.iterator
    })
    rdd2.foreach(println)
  }
  def reducePartition(sc:SparkContext)={
    var data=Array("A","B","C","D","E","F","G","H","I","J","K","L","M","N")
    var rdd1: RDD[String] =sc.parallelize(data,5)

    var rdd1_withindex=rdd1.mapPartitionsWithIndex((idx,iter)=>{
      var ret=new ListBuffer[String]()
      while(iter.hasNext){
        ret.+=:(idx+":"+iter.next())
      }
      ret.iterator
    })

    var rdd2=rdd1_withindex.coalesce(3,false)
    var rdd2_withindex: RDD[String] =rdd2.mapPartitionsWithIndex((idx, iter)=>{
      var ret=new ListBuffer[String]()
      while (iter.hasNext){
        ret.+=:(iter.next()+"--->"+idx)
      }
      ret.iterator
    })

    rdd2_withindex.foreach(println)
  }

  def groupByKey(sc:SparkContext)={
    var data=Array(
      ("zxk","day1"),("zxk","day2"),("zxk","day3"),
      ("zhuxin","day2"),("zhuxin","day3")
    )
    var rdd1: RDD[(String, String)] =sc.parallelize(data)
    var rdd2: RDD[(String, Iterable[String])] =rdd1.groupByKey()

    rdd2.foreach(println)
  }

  def  countByKey(sc:SparkContext)={
    var data=Array(
      ("zxk","day1"),("zxk","day2"),("zxk","day3"),
      ("zhuxin","day2"),("zhuxin","day3")
    )
    var rdd1: RDD[(String, String)] =sc.parallelize(data)

    var rdd2: collection.Map[String, Long] =rdd1.countByKey()
    rdd2.foreach((a)=>{
        println(a._1+"--->"+a._2)
    })
  }

  def reduce(sc:SparkContext)={
    var data=Array(
      ("zxk","day1"),("zxk","day2"),("zxk","day3"),
      ("zhuxin","day2"),("zhuxin","day3")
    )
    var rdd1: RDD[(String, String)] =sc.parallelize(data)

    var result: (String, String) =rdd1.reduce((a, b)=>{
      (a._1+"-"+b._1,a._2+"-"+b._2)
    })
    println(result._1)
    println(result._2)
  }

  def zip(sc:SparkContext)={
    var data1=Array(
      ("zxk","day1"),("zxk","day2"),("zxk","day3"),
      ("zhuxin","day2"),("zhuxin","day3")
    )
    var rdd1: RDD[(String, String)] =sc.parallelize(data1)

    var data2=Array(
      11,22,33,44,55
    )
    var rdd2: RDD[Int] =sc.parallelize(data2)


    var data3=Array(
      "lll","kkk","mmm","nnn","ppp"
    )
    var rdd3: RDD[String] =sc.parallelize(data3)

    var rdd4: RDD[(((String, String), Int), String)] =rdd1.zip(rdd2).zip(rdd3)

    rdd4.foreach(println)
  }
  def main(args: Array[String]): Unit = {
    var conf:SparkConf=new SparkConf()
    conf.setMaster("local").setAppName("spark")
    var sc:SparkContext=new SparkContext(conf)
//    mapPartitonWithIndex(sc)
//    reducePartition(sc)
//    groupByKey(sc)
//    countByKey(sc)
//    reduce(sc)
    zip(sc)
  }
}
