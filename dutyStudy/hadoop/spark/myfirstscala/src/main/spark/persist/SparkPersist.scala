package main.spark.persist

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangxk on 19-1-16.
  */
object SparkPersist {
  /*
  计算流程
  fs->rdd->count
         ->count
   所以计算count要回溯回读取fs
  * */
  def testCache(rdd:RDD[String]): Unit ={
    var s1=System.currentTimeMillis()
    rdd.count();
    var e1=System.currentTimeMillis()
    println((e1-s1)+"ms")

    s1=System.currentTimeMillis()
    rdd.count()
    e1=System.currentTimeMillis()

    println((e1-s1)+"ms")
  }
  def main(args: Array[String]): Unit = {
    var conf=new SparkConf()
    conf.setMaster("local").setAppName("mysparkPersist")
    var sc=new SparkContext(conf)
    sc.setCheckpointDir("./ck")
    var rdd=sc.textFile("/home/zhangxk/NASA_access_log_Aug95")
    rdd.checkpoint()
//    rdd=rdd.persist(StorageLevel.NONE)
//    rdd=rdd.persist(StorageLevel.MEMORY_ONLY)
//    rdd=rdd.persist(StorageLevel.MEMORY_AND_DISK)
//
    testCache(rdd)
    sc.stop()
  }
}
