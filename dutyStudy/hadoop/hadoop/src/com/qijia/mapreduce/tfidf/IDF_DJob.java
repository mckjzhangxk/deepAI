package com.qijia.mapreduce.tfidf;

import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

/**
 * Created by zhangxk on 18-12-27.
 */
public class IDF_DJob {
    /**
     * 统计某单词在某文章出现的频次,输出:
     * wordi:IDj num
     * 输入:/data/weibo.txt
     * */
    public static boolean doJob(String inputpath,String outputpath) throws IOException, ClassNotFoundException, InterruptedException {

        Configuration conf= MyConfigure.getConfigure("yarn");
        Job job=Job.getInstance(conf);
        job.setJarByClass(IDF_DJob.class);

        job.setMapperClass(MyMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        //相同的ID是一组
        job.setReducerClass(MyReduce.class);


        Path p=new Path(inputpath);
        FileInputFormat.addInputPath(job,p);
        //output path
        p=new Path(outputpath);
        if(p.getFileSystem(conf).exists(p)){
            p.getFileSystem(conf).delete(p,true);
        }
        FileOutputFormat.setOutputPath(job,p);

        job.setNumReduceTasks(1);

        job.waitForCompletion(true);
        return true;
    }
    static class MyMapper extends Mapper<Object,Text,Text,IntWritable>{
        private Text rkey=new Text();
        private IntWritable rval=new IntWritable(1);

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            //3823890210294392   300
            //3823890210294391   20
            rkey.set("articles");
            context.write(rkey,rval);
        }
    }
    static class MyReduce extends Reducer<Text,IntWritable,Text,IntWritable> {
        private Text rkey=new Text();
        private IntWritable rval=new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum=0;
            //articiles 1
            for (IntWritable v :values){
                sum+=1;
            }
            rkey.set(key);
            rval.set(sum);
            context.write(rkey,rval);
        }
    }

    public static void main(String[] args) throws InterruptedException, IOException, ClassNotFoundException {
        String input="/out/weibo/tf_normal";
        String output="/out/weibo/idf_D";
        doJob(input,output);
    }
}
