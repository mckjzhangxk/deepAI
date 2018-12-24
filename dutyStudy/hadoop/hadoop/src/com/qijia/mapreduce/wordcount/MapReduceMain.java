package com.qijia.mapreduce.wordcount;

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
        /**源码解析1:
         * 文件存储是按照block(128M)为基本单位的,
         * 文件处理(mr)是安装splitsize为基本单位的
         * splitsize是人为干预的,你可以指定大于或者小于blocksize
         *
         * 一个split会对于于一个block_i,进而知道block_i存储的位置node[i].hosts
         * 有多少个split,就有多少个map,
         * 当node[i].hosts存在空闲机器时候,map会在node[i].hosts中的一台机器上运行(计算向数据).
         * 否在resourceManager会rack或者cluster找一台空闲机器,把split传输过去计算(数据向计算)
         * file,split_offset,split_length,hosts 称为split文件清单.
         * */
        FileInputFormat.addInputPath(job,inputpat);
//        FileInputFormat.setMaxInputSplitSize(job,1024*1024); splitSize<=1k
//        FileInputFormat.setMinInputSplitSize(job,1024*1024*1024*256); splitSize>=256M

        Path outputpath=new Path("/cc");
        if(outputpath.getFileSystem(conf).exists(outputpath)){
            outputpath.getFileSystem(conf).delete(outputpath,true);
        }
        FileOutputFormat.setOutputPath(job,outputpath);

        /**源码解析2:
         * 客户端把要运行的jar,文件切割清单提交给ResourceManager后, ResourceManager
         * 指令各个节点的NodeManager,创建资源(Container),启动ApplicationManager(JVM),
         * 在每个计算节点中,启动一个MapTask.
         *
         * MapTask:包含job,可以反射会自定义的Mapper,它知道我这个map要处理哪些切片,
         * MapTask会先处理输入流(也就是让一行一行处理splits),这个方法是调用
         * job.getInputFormatClass()=TextInputFormat--->LineRecordReader,
         * LineRecordReader是实际上getCurrentKey,getCurrentValue,
         * 复杂为自定义mapper提供key,value,注意他会跳过一行
         * */

        //设在map ,reduce
        job.setMapperClass(MyMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setReducerClass(MyReducer.class);


        //
        job.waitForCompletion(true);
    }
}
