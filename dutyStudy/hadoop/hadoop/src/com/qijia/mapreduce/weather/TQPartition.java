package com.qijia.mapreduce.weather;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Partitioner;

/**
 * Created by zhangxk on 18-12-24.
 */
public class TQPartition extends Partitioner<TQ,IntWritable> {
    @Override

    public int getPartition(TQ tq, IntWritable intWritable, int i) {

        if (tq.getYear()<=1950)
            return 0;
        else return 1;
    }
}
