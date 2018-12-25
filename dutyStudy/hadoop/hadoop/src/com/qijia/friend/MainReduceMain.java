package com.qijia.friend;

import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

import java.io.IOException;

/**
 * Created by zhangxk on 18-12-25.
 */
public class MainReduceMain {
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf= MyConfigure.getConfigure("yarn");
        Job job=Job.getInstance(conf);
        job.setJarByClass(MainReduceMain.class);

        //map
        job.setMapperClass(MyMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        //reduce
        job.setReducerClass(MyReduce.class);
        //input,output
        Path p=new Path("/friend.txt");
        FileInputFormat.addInputPath(job,p);
        p=new Path("/friendout");
        if(p.getFileSystem(conf).exists(p)){
            p.getFileSystem(conf).delete(p,true);
        }

        job.waitForCompletion(true);
    }
}
