package com.qijia.weather;

import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.yarn.server.resourcemanager.monitor.capacity.TempQueuePerPartition;

import java.io.IOException;


/**
 * Created by zhangxk on 18-12-24.
 */
public class MapReduceMain {
    /**
     * 输入/tq.txt
     * 输出/tqout
     *
     * */
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf= MyConfigure.getConfigure("yarn");
        Job job=Job.getInstance(conf);
        job.setJarByClass(MapReduceMain.class);

        //准备输入,输出
        Path p=new Path("/tq.txt");
        FileInputFormat.addInputPath(job,p);
        p=new Path("/tqout");
        if(p.getFileSystem(conf).exists(p)){
            p.getFileSystem(conf).delete(p,true);
        }
        //准备mapper
        job.setMapperClass(MyMapper.class);
        job.setMapOutputKeyClass(TQ.class);
        job.setMapOutputValueClass(IntWritable.class);

        //设在如何把map的输出分给reduce
        job.setPartitionerClass(TQPartition.class);
        //设在输出key是如何排序的
        job.setSortComparatorClass(TQSortComparator.class);
        //设在输入框Key如何成为一组
        job.setGroupingComparatorClass(TQGroupComparator.class);

        //准备reduce
        job.setNumReduceTasks(2);
        job.setReducerClass(MyReduce.class);

        job.waitForCompletion(true);
    }
}
