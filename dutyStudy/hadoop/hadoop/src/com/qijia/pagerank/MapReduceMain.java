package com.qijia.pagerank;

import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.util.StringUtils;

import java.io.IOException;

/**
 * Created by zhangxk on 18-12-25.
 */
public class MapReduceMain {
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf= MyConfigure.getConfigure("yarn");
        Job job=Job.getInstance(conf);
        job.setJarByClass(MapReduceMain.class);

        String inputpath="/data/pagerank.txt";
        String outputpath="/out/pr";


        for (int i=0;i<10;i++){
            conf.setInt("runcount",i);
            //
            job.setInputFormatClass(KeyValueTextInputFormat.class);
            //mapper
            job.setMapperClass(MyMapper.class);
            job.setMapOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            //reducer
//            job.setReducerClass();

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
            double pgscore=1.0;
            if(runcount>0){
                length-=1;
                pgscore=Double.parseDouble(sps[length]);
            }


            //输出投票结果
            for(int i=0;i<length;i++){
                rkey.set(sps[i]);
                rval.set(pgscore+"");
                context.write(rkey,rval);
            }

            context.write(key,value);

        }
    }
}
