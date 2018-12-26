package com.qijia.pagerank;

import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.util.StringUtils;

import java.io.IOException;

/**
 * Created by zhangxk on 18-12-25.
 */
public class MapReduceMain {
    public static enum MyCounter{
        my
    }
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf= MyConfigure.getConfigure("yarn");
        Job job=Job.getInstance(conf);
        job.setJarByClass(MapReduceMain.class);

        String inputpath="/data/pagerank.txt";
        String outputpath="/out/pagerank/pr";

        for (int i=0;i<10;i++){
            conf.setInt("runcount",i);
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

            boolean f=job.waitForCompletion(true);
            if(f){
                inputpath=outputpath+i+"/part-00000";
            }
        }

    }

    class MyMapper extends Mapper<Text,Text,Text,Text>{
        private Text rkey=new Text();
        private Text rval=new Text();
        @Override
        protected void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            int runcount = context.getConfiguration().getInt("runcount",0);
            String[] sps=StringUtils.split(value.toString(),' ');

            int length=sps.length;

            double pgscore=1.0/length;
            if(runcount>0){
                pgscore=Double.parseDouble(sps[length-1]);
                length-=1;

                StringBuffer val=new StringBuffer();
                for(int i=0;i<length;i++){
                    val.append(sps[i]).append(' ');
                }
                val.append(pgscore);
                value.set(val.toString());
            }
            //输出关系
            context.write(key,value);

            //输出投票结果
            for(int i=0;i<length;i++){
                rkey.set(sps[i]);
                rval.set(pgscore+"");
                context.write(rkey,rval);
            }



        }
    }

    class MyReducer extends Reducer<Text,Text,Text,Text>{
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
                String[] sps=StringUtils.split(v.toString(),' ');
                if(isRalation(sps)){
                    for (int i=0;i<sps.length-1;i++){
                        relation.append(sps[i]+' ');
                    }
                    old_orscore=Double.parseDouble(sps[sps.length-1]);
                }else {
                    prscore+=Double.parseDouble(sps[0]);
                }
            }
            context.getCounter(MyCounter.my).increment((long) (Math.abs(prscore-old_orscore)*1000));
            relation.append(prscore);
            rval.set(relation.toString());
            context.write(key,rval);
        }

    }
}
