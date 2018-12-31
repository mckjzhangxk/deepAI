package com.qijia.mapreduce.weather;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by zhangxk on 18-12-24.
 */
public class MyReduce extends Reducer<TQ,IntWritable,Text,IntWritable> {
    private Map<Integer,Integer> cache=new HashMap<>();
    private Text mkey=new Text();
    private IntWritable mvalue=new IntWritable();

    @Override
    protected void reduce(TQ key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        //不要让上一次干预
        cache.clear();
        boolean first=true;
        for(IntWritable v :values){
            if(first){
                int day=key.getDay();
                cache.put(day,1);
                first=false;
                mkey.set(key.toString());
                mvalue.set(v.get());
                context.write(mkey,mvalue);
            }else {
                int day=key.getDay();
                if(!cache.containsKey(day)){
                    mkey.set(key.toString());
                    mvalue.set(v.get());
                    context.write(mkey,mvalue);
                    break;
                }
            }

        }
    }

}
