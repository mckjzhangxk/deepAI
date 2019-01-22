package main.spark.persist

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangxk on 19-1-16.
  */
object SparkPersist {
  def run(task:()=>Unit): Unit ={
    var start=System.currentTimeMillis()
    task()
    var end=System.currentTimeMillis()
    println((end-start)+"ms")
  }
  /*
  计算流程
  fs->rdd->count
         ->count
   所以计算count要回溯回读取fs
  * */
  def testCache(rdd:RDD[String]): Unit ={
   var rdd_cache=rdd.cache();
    for (i <-1 to 5){
      run(()=>{
        print(i)
        rdd_cache.count()
      })
    }
  }
  def getContent(): SparkContext ={
    var conf=new SparkConf()
    conf.setAppName("cache").setMaster("local")
    var sc=new SparkContext(conf)
    return sc
  }
  def main(args: Array[String]): Unit = {
    var sc=getContent()
////    sc.setCheckpointDir("./ck")
    var rdd=sc.textFile("/home/zhangxk/NASA_access_log_Aug95")
////    rdd.checkpoint()
////    rdd=rdd.persist(StorageLevel.NONE)
////    rdd=rdd.persist(StorageLevel.MEMORY_ONLY)
////    rdd=rdd.persist(StorageLevel.MEMORY_AND_DISK)
////
    testCache(rdd)
    sc.stop()
  }
}
