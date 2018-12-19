package com.qijia.conf;

import org.apache.hadoop.conf.Configuration;

/**
 * Created by zhangxk on 18-12-19.
 */
public class MyConfigure {
    /**
     * 这个方法会加载给定前缀下的core-site.xml和hdfs.xml,
     * 注意prefix 前后都没有/,比如ha
     */
    public static Configuration getConfigure(String prefix){

        //设置true会跑到src目录下找XML文件,false不会去找
        Configuration conf=new Configuration(false);

        //这里是手动制定文件路径
//        conf.addResource(new Path(""));
//        conf.addResource(new Path(""));
//        System.out.println(HDFS.class.getResource("/conf/ha/core-site.xml"));
        conf.addResource(MyConfigure.class.getResource(prefix+"/core-site.xml"));
        conf.addResource(MyConfigure.class.getResource(prefix+"/hdfs-site.xml"));
//        conf.addResource(MyConfigure.class.getResource(prefix+"/mapred-site.xml"));
//        conf.addResource(MyConfigure.class.getResource(prefix+"/yarn-site.xml"));

//        System.out.println(conf.get("fs.defaultFS"));
//        System.out.println(conf.get("dfs.replication"));
//        System.out.println(MyConfigure.class.getResource(""));
        return conf;
    }
}
