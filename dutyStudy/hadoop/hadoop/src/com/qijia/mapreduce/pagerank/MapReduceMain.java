package com.qijia.mapreduce.pagerank;

import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;

import java.io.IOException;

/**
 * Created by zhangxk on 18-12-25.
 */
public class MapReduceMain {
    public static char splits_char=',';
    public static long precision=10000;

    public static double randomProb=0.85;
    public static int N=4;


    public static enum MyCounter{
        my
    }
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf= MyConfigure.getConfigure("yarn",true);

        String inputpath="/data/pagerank.txt";
        String outputpath="/out/pagerank/pr";

        int i=0;
        while (true){
            //            注意这里一定要在job创建之前设置
            conf.setInt("runcount",i);
            Job job=Job.getInstance(conf);
            job.setJarByClass(MapReduceMain.class);
            job.setJar("/home/zxk/PycharmProjects/deepAI/dutyStudy/hadoop/hadoop/out/artifacts/zxk/zxk.jar");

            //
            job.setInputFormatClass(KeyValueTextInputFormat.class);
            //mapper
            job.setMapperClass(MyMapper.class);
            job.setMapOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            //reducer
            job.setReducerClass(MyReducer.class);

            //input path
            Path p=new Path(inputpath);
            FileInputFormat.addInputPath(job,p);
            //output path
            p=new Path(outputpath+i);
            if(p.getFileSystem(conf).exists(p)){
                p.getFileSystem(conf).delete(p,true);
            }
            FileOutputFormat.setOutputPath(job,p);

            boolean f=job.waitForCompletion(true);
            if(f){
                long delta = job.getCounters().findCounter(MyCounter.my).getValue();
                if((delta/precision)<0.0001){
                    break;
                }
                inputpath=outputpath+i+"/part-r-00000";
                i+=1;
            }
        }

    }

    static class MyMapper extends Mapper<Text,Text,Text,Text>{
        private Text rkey=new Text();
        private Text rval=new Text();

        /**输入: A B C score(A)
         * 输出:
         *      A B C score(A)
         *      B score(A)/2
         *      C score(A)/2
         * */
        @Override
        protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            int runcount = context.getConfiguration().getInt("runcount",0);
            String[] sps=StringUtils.split(value.toString(),splits_char);

            //第一次,也就是默认分数三1,value全都是关系
            int length=sps.length;
            double pgscore=0;
            //非第一次,最后一个三表示key的基本分数

            if(runcount>0){
                pgscore=Double.parseDouble(sps[length-1])/(length-1);
                length-=1;
                value.set(value.toString());
                context.write(key,value); //输出关系,附带基本分数
            }else{
                pgscore=1.0/length;
                value.set(value.toString()+splits_char+1.0);
                context.write(key,value); //输出关系,附带基本分数
            }


            //length表示ourdegress,pscore三对outnode的pk分数
            //输出投票结果
            for(int i=0;i<length;i++){
                rkey.set(sps[i]);
                rval.set(pgscore+"");
                context.write(rkey,rval);
            }


        }
    }

    static class MyReducer extends Reducer<Text,Text,Text,Text>{
        private Text rval=new Text();

        private boolean isRalation(String[] spls){
            String first=spls[0];
            try{
                Float.parseFloat(first);
                return false;
            }catch (Exception e){
                return true;
            }
        }
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double prscore=0;
            double old_orscore=0;


            StringBuffer relation=new StringBuffer();
            for(Text v:values){
                String[] sps=StringUtils.split(v.toString(),splits_char);
                if(isRalation(sps)){
                    for (int i=0;i<sps.length-1;i++){
                        relation.append(sps[i]+splits_char);
                    }
                    old_orscore=Double.parseDouble(sps[sps.length-1]);
                }else {
                    prscore+=Double.parseDouble(sps[0]);
                }
            }
            relation.append(prscore*randomProb+(1-randomProb)/N);
            context.getCounter(MyCounter.my).increment((long) (precision*Math.abs(prscore-old_orscore)));

            rval.set(relation.toString());
            context.write(key,rval);
        }

    }
}
