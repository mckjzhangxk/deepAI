package com.qijia.mapreduce.friend;

import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

/**
 * Created by zhangxk on 18-12-25.
 */
public class MainReduceMain {
    /**
     * 目的: 找出人的间接好朋友,并给出分数
     * tom hello hadoop cat
     * world hadoop hello hive
     * cat tom hive
     *
    */
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf= MyConfigure.getConfigure("ha");
        Job job=Job.getInstance(conf);
        job.setJarByClass(MainReduceMain.class);
//        job.setJar("/home/zxk/PycharmProjects/deepAI/dutyStudy/hadoop/hadoop/out/artifacts/zxk/zxk.jar");
        //map,输出2类数据
        //直接关系,也就是原样输出,但区分类型1
        //间接关系,类型0
        //比如A B C D
        //输出:
        /**A:B 1
         * A:C 1
         * A:D 1
         * B:C 0
         * B:D 0
         * C:D 0
         * */
        job.setMapperClass(MyMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        //reduce
        job.setReducerClass(MyReduce.class);

        //input,output
        Path p=new Path("/data/friend");
        FileInputFormat.addInputPath(job,p);
        p=new Path("/out/friend");
        if(p.getFileSystem(conf).exists(p)){
            p.getFileSystem(conf).delete(p,true);
        }
        FileOutputFormat.setOutputPath(job,p);

        job.waitForCompletion(true);
    }
}
