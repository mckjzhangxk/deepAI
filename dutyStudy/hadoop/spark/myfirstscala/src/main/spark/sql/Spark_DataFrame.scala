package main.spark.sql

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

/**
  * Created by zhangxk on 19-1-28.
  */
object Spark_DataFrame {
  def getDataSource(sqlcxt:SQLContext):DataFrame={
//    var ret=sqlcxt.read.("/home/zhangxk/projects/deepAI/dutyStudy/hadoop/spark/myfirstscala/src/main/spark/sql/datasource/person.txt")
    var ret: DataFrame =sqlcxt.read.json("/home/zhangxk/projects/deepAI/dutyStudy/hadoop/spark/myfirstscala/src/main/spark/sql/datasource/json")
    return ret
  }

  def main(args: Array[String]): Unit = {
    var conf=new SparkConf()
    conf.setMaster("local").setAppName("sql")
    var sc=new SparkContext(conf)
    var sql=new SQLContext(sc)

    var df=getDataSource(sql)
    //打印schema
    df.printSchema()
    println("===========SELECT * FROM ...================")
    df.show()
    //注册临时表后可以使用sql
    df.registerTempTable("t1")
    df=sql.sql("select name from t1 where age=20")
    df.show()
    println("=========RDD==================")
    //df本质操作RDD
    var rowrdd: RDD[Row] =df.rdd
    rowrdd.foreach((row)=>{
      println(row.get(0))
    })
    sc.stop()
  }
}
