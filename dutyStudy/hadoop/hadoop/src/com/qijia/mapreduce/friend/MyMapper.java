package com.qijia.mapreduce.friend;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by zhangxk on 18-12-25.
 */
public class MyMapper extends Mapper <Object,Text,Text,IntWritable>{
    private Text rkey=new Text();
    private IntWritable rval=new IntWritable();
//    tom hello hadoop cat
//    world hadoop hello hive
//    cat tom hive
//    mr hive hello
//    hive cat hadoop world hello mr
//    hadoop tom hive world
//    hello tom world hive mr

    //输出:
    //tom:hello 1
    //tom:hadoop 0
    //tom:cat 0
    //........
    //........
    //........
    //hello:mr 0
    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] strs=value.toString().split(" ");

        for(int i=1;i<strs.length;i++){
            rkey.set(combString(strs[0],strs[i]));
            rval.set(1);
            context.write(rkey,rval);
            for(int j=i+1;j<strs.length;j++){
                rkey.set(combString(strs[i],strs[j]));
                rval.set(0);
                context.write(rkey,rval);
            }
        }
    }
    private String combString(String str1,String str2){
        if(str1.compareTo(str2)<=0)
            return str1+":"+str2;
        return str2+":"+str1;
    }
}
