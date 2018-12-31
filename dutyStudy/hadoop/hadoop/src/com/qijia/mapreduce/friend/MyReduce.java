package com.qijia.mapreduce.friend;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by zhangxk on 18-12-25.
 */
public class MyReduce extends Reducer<Text,IntWritable,Text,IntWritable>{
    //    tom:hello 1
    //    tom:hello 0
    //    tom:hello 0
    //1表示有直接关联,0表示没有
    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        int sum=0;
        for(IntWritable v:values){
            if(v.get()==1)
                return;
            sum+=1;
        }
        context.write(key,new IntWritable(sum));
    }
}
