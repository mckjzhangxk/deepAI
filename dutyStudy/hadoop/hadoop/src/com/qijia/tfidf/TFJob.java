package com.qijia.tfidf;

import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;

import java.io.IOException;
import java.io.StringReader;

/**
 * Created by zhangxk on 18-12-27.
 */
public class TFJob {
    /**
     * 统计某单词在某文章出现的频次,输出:
     * wordi:IDj num
     * 输入:/data/weibo.txt
     * */
    public static boolean doJob(String inputpath,String outputpath) throws IOException, ClassNotFoundException, InterruptedException {

        Configuration conf= MyConfigure.getConfigure("yarn");
        Job job=Job.getInstance(conf);
        job.setJarByClass(TFJob.class);

        job.setMapperClass(MyMapper.class);
        job.setMapOutputKeyClass(Weibo.class);
        job.setMapOutputValueClass(IntWritable.class);

        //相同的word,ID是一组
        job.setGroupingComparatorClass(MyGroupComparator.class);
        job.setReducerClass(MyReduce.class);


        Path p=new Path(inputpath);
        FileInputFormat.addInputPath(job,p);
        //output path
        p=new Path(outputpath);
        if(p.getFileSystem(conf).exists(p)){
            p.getFileSystem(conf).delete(p,true);
        }
        FileOutputFormat.setOutputPath(job,p);

        job.setNumReduceTasks(3);

        job.waitForCompletion(true);
        return true;
    }
    public class MyMapper extends Mapper<Object,Text,Weibo,IntWritable>{
        private Weibo rkey=new Weibo();
        private IntWritable rval=new IntWritable(1);

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            //3823890210294392	今天我约了豆浆，油条
            String[] sps = value.toString().trim().split("\t");

            rkey.setId(sps[0]);
            String content=sps[1];

            StringReader reader=new StringReader(content);
            IKSegmenter ik=new IKSegmenter(reader,false);

            Lexeme word = null;
            while ((word=ik.next())!=null){
                String w = word.getLexemeText();
                rkey.setWord(w);

                context.write(rkey,rval);
            }

        }
    }
    public class MyReduce extends Reducer<Weibo,IntWritable,Text,IntWritable> {
        private Text rkey=new Text();
        private IntWritable rval=new IntWritable();

        @Override
        protected void reduce(Weibo key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum=0;

            for (IntWritable v :values){
                sum+=1;
            }
            rkey.set(key.toString());
            rval.set(sum);
            context.write(rkey,rval);
        }
    }
    public class MyGroupComparator extends WritableComparator{
        public MyGroupComparator(){
            super(Weibo.class,true);
        }
        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            Weibo q1= (Weibo) a;
            Weibo q2= (Weibo) b;

            int c1=q1.getWord().compareTo(q2.getWord());
            if(c1!=0) return c1;

            return q1.getId().compareTo(q2.getId());
        }
    }
    public static void main(String[] args) throws InterruptedException, IOException, ClassNotFoundException {
        String input="/data/weibo.txt";
        String output="/out/weibo/tf";
        doJob(input,output);
    }
}
