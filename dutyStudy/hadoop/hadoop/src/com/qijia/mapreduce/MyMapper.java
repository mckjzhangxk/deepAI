package com.qijia.mapreduce;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.StringTokenizer;

/**
 * Created by zhangxk on 18-12-19.
 */
public class MyMapper extends Mapper<Object,Text,Text,IntWritable> {
    private Text word=new Text();
    private IntWritable one=new IntWritable();
    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        StringTokenizer iter=new StringTokenizer(value.toString());

        while (iter.hasMoreTokens()){
            word.set(iter.nextToken());
            context.write(word,one);
        }
    }
}
