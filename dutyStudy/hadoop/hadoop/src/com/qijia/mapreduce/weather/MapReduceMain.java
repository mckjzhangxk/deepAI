package com.qijia.mapreduce.weather;

import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;


/**
 * Created by zhangxk on 18-12-24.
 */
public class MapReduceMain {
    /**
     * 输入/data/weather
     * 输出/out/weather
     *  目的: 按照月份,输出每个月最高温度的两天,注意这2天不能是相同日期
     * */
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf= MyConfigure.getConfigure("ha");
        Job job=Job.getInstance(conf);
        job.setJar("/home/zxk/PycharmProjects/deepAI/dutyStudy/hadoop/hadoop/out/artifacts/zxk/zxk.jar");
        job.setJarByClass(MapReduceMain.class);

        //准备输入,输出
        Path p=new Path("/data/weather");
        FileInputFormat.addInputPath(job,p);
        p=new Path("/out/weather");
        if(p.getFileSystem(conf).exists(p)){
            p.getFileSystem(conf).delete(p,true);
        }
        FileOutputFormat.setOutputPath(job,p);

        //准备mapper
        job.setMapperClass(MyMapper.class);
        job.setMapOutputKeyClass(TQ.class);
        job.setMapOutputValueClass(IntWritable.class);

        //设置如何把map的输出分给reduce
        job.setPartitionerClass(TQPartition.class);
        //设在map的输出 如何排序的,year-mouth-temperature
        job.setSortComparatorClass(TQSortComparator.class);
        //设置相同的KEY为一组,这里相同的year-mouth是一组,在前面的记录温度高于后面的
        job.setGroupingComparatorClass(TQGroupComparator.class);

        //准备reduce

        job.setReducerClass(MyReduce.class);
        job.setNumReduceTasks(2);
        job.waitForCompletion(true);
    }
}
