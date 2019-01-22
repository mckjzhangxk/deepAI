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
        Configuration conf=new Configuration(true);

        //这里是手动制定文件路径
//        conf.addResource(new Path(""));
//        conf.addResource(new Path(""));
//        System.out.println(HDFS.class.getResource("/conf/ha/core-site.xml"));
        if(!prefix.contains("hbase")){
            conf.addResource(MyConfigure.class.getResource(prefix+"/core-site.xml"));
            conf.addResource(MyConfigure.class.getResource(prefix+"/hdfs-site.xml"));
        }else {
            conf.addResource(MyConfigure.class.getResource(prefix+"/hbase-site.xml"));
            conf.addResource(MyConfigure.class.getResource(prefix+"/hdfs-site.xml"));
        }



//        System.out.println(conf.get("mapreduce.framework.name"));
//        System.out.println(conf.get("dfs.replication"));
//        System.out.println(MyConfigure.class.getResource(""));
        return conf;
    }

    public static Configuration getConfigure(String prefix,boolean onyarn){
        Configuration conf=getConfigure(prefix);
        if(prefix.equals("yarn") && onyarn){
            conf.addResource(MyConfigure.class.getResource(prefix+"/mapred-site.xml"));
            conf.addResource(MyConfigure.class.getResource(prefix+"/yarn-site.xml"));
        }
        return conf;
    }
}
