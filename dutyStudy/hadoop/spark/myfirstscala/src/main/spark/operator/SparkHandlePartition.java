package main.spark.operator;

import org.apache.spark.SparkConf;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.api.java.function.VoidFunction;
import scala.Tuple3;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Created by zhangxk on 19-1-24.
 */
public class SparkHandlePartition {
    /*
    * 比较map 与mappartions的不同,map针对单个元素操作
    * map针对一组操作
    * */
    public static void testMapPartitions(JavaSparkContext sc){

        //一个人员数据源(name,age,address)
        List<Tuple3<String,Integer,String>> data = Arrays.asList(new Tuple3[]{
                new Tuple3<String,Integer,String>("zxk",30,"lixia"),
                new Tuple3<String,Integer,String>("wangmk",31,"gaoxin"),
                new Tuple3<String,Integer,String>("wanb",32,"gaoxin"),
                new Tuple3<String,Integer,String>("zhuliang",38,"shizhong"),

                new Tuple3<String,Integer,String>("lirun",44,"lixia"),
                new Tuple3<String,Integer,String>("songyuxin",55,"gaoxin"),
                new Tuple3<String,Integer,String>("lishuhui",32,"shizhong"),
                new Tuple3<String,Integer,String>("mengsicong",29,"huaiyin"),
        } );
        JavaRDD<Tuple3<String,Integer,String>> rdd = sc.parallelize(data,4);

        System.out.println("Have partition num:"+rdd.getNumPartitions());

        JavaRDD<String> name_maprdd= rdd.map(new Function<Tuple3<String,Integer,String>, String>() {

            @Override
            public String call(Tuple3<String, Integer, String> people) throws Exception {
                System.out.println("handle with each element!");
                return people._1();
            }
        });

//        //Iterator<String>是一个partition里面元素的迭代,返回的应该也是迭代器,类型是String
        JavaRDD<String> name_mappartitionrdd = rdd.mapPartitions(new FlatMapFunction<Iterator<Tuple3<String, Integer, String>>, String>() {
            @Override
            public Iterable<String> call(Iterator<Tuple3<String, Integer, String>> peoples) throws Exception {
                List<String> ret = new ArrayList<>();
                System.out.println("handle with partitions!");
                while (peoples.hasNext()) {
                    ret.add(peoples.next()._1());
                }
                return ret;
            }
        });

        List<String> collect = name_maprdd.collect();
        System.out.println("result:"+collect);
        List<String> collect1 = name_mappartitionrdd.collect();
        System.out.println("result:"+collect1);

    }
    public static void testMapForEach(JavaSparkContext sc){
        //一个人员数据源(name,age,address)
        List<Tuple3<String,Integer,String>> data = Arrays.asList(new Tuple3[]{
                new Tuple3<String,Integer,String>("zxk",30,"lixia"),
                new Tuple3<String,Integer,String>("wangmk",31,"gaoxin"),
                new Tuple3<String,Integer,String>("wanb",32,"gaoxin"),
                new Tuple3<String,Integer,String>("zhuliang",38,"shizhong"),

                new Tuple3<String,Integer,String>("lirun",44,"lixia"),
                new Tuple3<String,Integer,String>("songyuxin",55,"gaoxin"),
                new Tuple3<String,Integer,String>("lishuhui",32,"shizhong"),
                new Tuple3<String,Integer,String>("mengsicong",29,"huaiyin"),
        } );
        JavaRDD<Tuple3<String,Integer,String>> rdd = sc.parallelize(data,4);

        //一个一个元素处理
        rdd.foreach(new VoidFunction<Tuple3<String, Integer, String>>() {
            @Override
            public void call(Tuple3<String, Integer, String> people) throws Exception {
                System.out.println("call with ForEach");
                System.out.println(people);
            }
        });

        //一个一个分区处理
        rdd.foreachPartition(new VoidFunction<Iterator<Tuple3<String, Integer, String>>>() {
            @Override
            public void call(Iterator<Tuple3<String, Integer, String>> peoples) throws Exception {
                System.out.println("call with ForEachPartition");
                while (peoples.hasNext()){
                    System.out.println(peoples.next());
                }
            }
        });
    }

    public static void main(String[] args){
        SparkConf conf=new SparkConf();
        conf.setMaster("local").setAppName("spark_with_javaapi");
        JavaSparkContext sc=new JavaSparkContext(conf);
//        testMapPartitions(sc);
        testMapForEach(sc);
        sc.stop();
    }
}
