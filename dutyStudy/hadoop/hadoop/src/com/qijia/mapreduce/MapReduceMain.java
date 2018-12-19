package com.qijia.mapreduce;

import com.qijia.HDFS;
import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.net.URI;

/**
 * Created by zhangxk on 18-12-19.
 */
public class MapReduceMain {



    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

        Configuration conf=MyConfigure.getConfigure("yarn");


        Job job=Job.getInstance(conf);

        job.setJarByClass(MapReduceMain.class);
        job.setJobName("my_map_reduce1");


        //设置输入输出文件名
        Path inputpat=new Path("/words.txt");
        FileInputFormat.addInputPath(job,inputpat);

        Path outputpath=new Path("/cc");
        if(outputpath.getFileSystem(conf).exists(outputpath)){
            outputpath.getFileSystem(conf).delete(outputpath,true);
        }
        FileOutputFormat.setOutputPath(job,outputpath);

        //设在map ,reduce
        job.setMapperClass(MyMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setReducerClass(MyReducer.class);


        //
        job.waitForCompletion(true);
    }
}
