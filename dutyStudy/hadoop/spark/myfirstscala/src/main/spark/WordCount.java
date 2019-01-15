import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import scala.Tuple2;

import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * Created by zhangxk on 19-1-15.
 */
public class WordCount {
    public static void main(String[] args){
        SparkConf conf=new SparkConf();
        conf.setMaster("local");
        conf.setAppName("Java_WC");

        JavaSparkContext sc=new JavaSparkContext(conf);


        JavaRDD<String> lines = sc.textFile("./wc.txt");

        //一行变成多行,单词变元祖,合并相同单词
        JavaRDD<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public Iterable<String> call(String l) throws Exception {
                return Arrays.asList(l.split(" "));
            }
        });
        JavaPairRDD<String, Integer> wc = words.mapToPair(new PairFunction<String, String, Integer>() {
            @Override
            public Tuple2<String, Integer> call(String word) throws Exception {
                return new Tuple2(word, 1);
            }
        });
        JavaPairRDD<String, Integer> reduce = wc.reduceByKey(new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer v1, Integer v2) throws Exception {
                return v1 + v2;
            }
        });

        //key,value 交换顺序,然后排序,再交换回来
        JavaPairRDD<Integer, String> rdd1 = reduce.mapToPair(new PairFunction<Tuple2<String, Integer>, Integer, String>() {

            @Override
            public Tuple2<Integer, String> call(Tuple2<String, Integer> v) throws Exception {
                return v.swap();
            }
        });
        rdd1=rdd1.sortByKey();
        reduce=rdd1.mapToPair(new PairFunction<Tuple2<Integer, String>, String, Integer>() {
            @Override
            public Tuple2<String, Integer> call(Tuple2<Integer, String> v) throws Exception {
                return v.swap();
            }
        });

        reduce.foreach(new VoidFunction<Tuple2<String, Integer>>() {
            @Override
            public void call(Tuple2<String, Integer> v) throws Exception {
                System.out.println(v);
            }
        });
        sc.close();
    }
}
