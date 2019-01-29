package main.spark.sql;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;


/**
 * Created by zhangxk on 19-1-29.
 */
public class Spark_DF_RDD {
    public static void main(String [] args){
        //context是 driver上创建的,可以看作是一个客户端
        SparkConf conf=new SparkConf();
        conf.setMaster("local").setAppName("spark_app");
        JavaSparkContext sc=new JavaSparkContext(conf);
        SQLContext sql=new SQLContext(sc);

        JavaRDD<String> lines = sc.textFile("/home/zhangxk/projects/deepAI/dutyStudy/hadoop/spark/myfirstscala/src/main/spark/sql/datasource/person.txt", 1);
        //算子是在executor上执行的,driver让executor执行某个算子,driver 把对象传递给executor
        Person p = new Person();
        JavaRDD<Person> rdd_Person = lines.map(new Function<String, Person>() {
            @Override
            public Person call(String s) throws Exception {
                String[] split = s.split(",");
                p.setId(split[0]);
                p.setName(split[1]);
                p.setAge(Integer.parseInt(split[2]));
                return p;
            }
        });

        DataFrame df = sql.createDataFrame(rdd_Person, Person.class);
        df.show();
        df.printSchema();
        df.registerTempTable("t1");

        sql.sql("select id from t1").show();
    }
}
