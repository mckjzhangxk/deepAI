package main.spark.operator;

import com.google.common.base.Optional;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

/**
 * Created by zhangxk on 19-1-24.
 */
public class SparkPairOperator {
    public static void testJoin(JavaSparkContext sc){
        JavaPairRDD[] dataset = getDataset(sc);
        JavaPairRDD<String,Integer> em_rdd=dataset[0];
        JavaPairRDD<String,String> edu_rdd=dataset[1];

        //第一个STring表示key,Tuple._1是rdd1的value,Tuple._2是rdd2的value
        JavaPairRDD<String, Tuple2<Integer,String>> join = em_rdd.join(edu_rdd);
        List<Tuple2<String, Tuple2<Integer,String>>> ret_join = join.collect();
        System.out.println("joint result");
        System.out.println(ret_join);

        JavaPairRDD<String, Tuple2<Integer, Optional<String>>> leftjoin = em_rdd.leftOuterJoin(edu_rdd);
        List<Tuple2<String, Tuple2<Integer, Optional<String>>>> ret_leftjoin = leftjoin.collect();
        System.out.println("leftjoin result");
        System.out.println(ret_leftjoin);

        JavaPairRDD<String, Tuple2<Optional<Integer>, String>> right_join = em_rdd.rightOuterJoin(edu_rdd);
        List<Tuple2<String,Tuple2<Optional<Integer>, String>>> ret_rightjoin=right_join.collect();
        System.out.println("rightjoin result");
        System.out.println(ret_rightjoin);

        JavaPairRDD<String, Tuple2<Optional<Integer>, Optional<String>>> full_join = em_rdd.fullOuterJoin(edu_rdd);
        List<Tuple2<String, Tuple2<Optional<Integer>, Optional<String>>>> ret_full_join = full_join.collect();
        System.out.println("full join result");
        System.out.println(ret_full_join);
    }
    public static void testUnionAndSubstractAndDistinct(JavaSparkContext sc){
        JavaPairRDD[] dataset = getDataset(sc);
        JavaPairRDD<String,String> em1_rdd = dataset[0];
        JavaPairRDD<String,String> em2_rdd = dataset[2];

        JavaPairRDD<String, String> union = em1_rdd.union(em2_rdd);
        List<Tuple2<String, String>> ret_union = union.collect();
        System.out.println("union result");
        System.out.println(ret_union);

        JavaPairRDD<String, String> intersect = em1_rdd.intersection(em2_rdd);
        List<Tuple2<String, String>> ret_intersection = intersect.collect();
        System.out.println("intersection result");
        System.out.println(ret_intersection);


        //注意:substract的比较是em1_rdd[i]==em2_rdd[i],键值必须完全相同才是要减去的元素!!
        JavaPairRDD<String, String> substract = em1_rdd.subtract(em2_rdd);
        List<Tuple2<String, String>> ret_substract = substract.collect();
        System.out.println("substract result");
        System.out.println(ret_substract);

        JavaPairRDD<String, String> distinct_rdd = union.distinct();
        List<Tuple2<String, String>> ret_distinct = distinct_rdd.collect();
        System.out.println("distinct result");
        System.out.println(ret_distinct);

    }

    public static void testCoGroup(JavaSparkContext sc){
        JavaPairRDD[] dataset = getDataset(sc);
        JavaPairRDD<String,String> edu1_rdd = dataset[1];
        JavaPairRDD<String,String> edu2_rdd = dataset[3];

        JavaPairRDD<String, Tuple2<Iterable<String>, Iterable<String>>> cogroup = edu1_rdd.cogroup(edu2_rdd);
        List<Tuple2<String, Tuple2<Iterable<String>, Iterable<String>>>> ret_cogroup = cogroup.collect();
        System.out.println("cogroup result");
        System.out.println(ret_cogroup);
    }
    public static JavaPairRDD[] getDataset(JavaSparkContext sc){
        List  employees1 = Arrays.asList(new Tuple2[]{
                new Tuple2<String,Integer>("zxk",30),
                new Tuple2<String,Integer>("wangmk",31),
                new Tuple2<String,Integer>("wanb",32),
        } );

        List  employees2 = Arrays.asList(new Tuple2[]{
                new Tuple2<String,Integer>("zhuliang",39),
                new Tuple2<String,Integer>("yangbin",35),
                new Tuple2<String,Integer>("mengsicong",29),
                new Tuple2<String,Integer>("wanb",32)
        } );

        List edu1 = Arrays.asList(new Tuple2[]{
                new Tuple2<String,String>("wangmk","PHD"),
                new Tuple2<String,String>("wanb","GRUDUATE"),
                new Tuple2<String,String>("zhuliang","PHD")
        } );

        List edu2 = Arrays.asList(new Tuple2[]{
                new Tuple2<String,String>("wangmk","PHD"),
                new Tuple2<String,String>("wanb","GRUDUATE"),
                new Tuple2<String,String>("wanb","UNDER_GRUDUATE"),
                new Tuple2<String,String>("zhuliang","PHD")
        } );

        JavaPairRDD<String, Integer> em1_rdd = sc.parallelizePairs(employees1);
        JavaPairRDD<String, Integer> em2_rdd = sc.parallelizePairs(employees2);
        JavaPairRDD<String, String> edu1_rdd = sc.parallelizePairs(edu1);
        JavaPairRDD<String, String> edu2_rdd = sc.parallelizePairs(edu2);



        return new JavaPairRDD[]{em1_rdd, edu1_rdd,em2_rdd,edu2_rdd};

    }
    public static void main(String[] args){
        SparkConf conf=new SparkConf();
        conf.setAppName("sparkjoint").setMaster("local");
        JavaSparkContext sc=new JavaSparkContext(conf);
        testJoin(sc);
        testUnionAndSubstractAndDistinct(sc);
        testCoGroup(sc);
        sc.stop();

    }
}
