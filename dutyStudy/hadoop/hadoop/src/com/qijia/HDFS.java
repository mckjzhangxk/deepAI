package com.qijia;

import com.qijia.conf.MyConfigure;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IOUtils;

import java.io.*;
import java.net.URI;


public class HDFS {
    public static void listFiles(FileSystem fs) throws IOException {
        Path path=new Path("/");
        FileStatus[] fileStatuses = fs.listStatus(path);
        System.out.println("文件系统 默认块大小(M):"+(fs.getDefaultBlockSize()/1024/1024));
        for(FileStatus s:fileStatuses)
            System.out.println(s.getPath()+" blocksize(M):"+s.getBlockSize()/1024/1024+" ,copys:"+s.getReplication()+",size(M)"+s.getLen()/1024/1024);

    }

    public static void blockLocation(FileSystem fs,String _path) throws IOException {
        Path path=new Path(_path);
        FileStatus fileStatus = fs.getFileStatus(path);
        BlockLocation[] fileBlockLocations = fs.getFileBlockLocations(fileStatus, 0, fileStatus.getLen());
        System.out.println("文件块信息");
        for (BlockLocation b:fileBlockLocations){
            System.out.println(b);
//            System.out.println(b.getHosts()+",offset:"+b.getOffset()+",len:"+b.getLength());
        }
    }

    public static void main(String[] gras) throws IOException, InterruptedException {
        //注意,必须设置环境变量HADOOP_USER_NAME 与hdfs保持一致
        System.out.println(System.getenv("HADOOP_USER_NAME"));

        Configuration conf= MyConfigure.getConfigure("ha");
        FileSystem fs=FileSystem.get(URI.create(""),conf,"root");



        //创建目录代码
        Path p=new Path("/hello");
        if(!fs.exists(p))
            fs.mkdirs(p);

        //下载文件代码
        p=new Path("/jdk-8u131-linux-x64.tar.gz");
        OutputStream outputStream=new FileOutputStream("jdk-8u131-linux-x64.tar.gz");
        InputStream inputStream = fs.open(p);
        IOUtils.copyBytes(inputStream,outputStream,4096,true);

        //上传文件代码
        p=new Path("/timg.jpeg");
        inputStream=new FileInputStream("/home/zxk/PycharmProjects/deepAI/dutyStudy/hadoop/hadoop/src/data/timg.jpeg");
        outputStream = fs.create(p);
        IOUtils.copyBytes(inputStream,outputStream,4096,true);


        blockLocation(fs,"/timg.jpeg");

//        listFiles(fs);


//        fs.delete(p,true);
        listFiles(fs);


        fs.close();
    }
}
