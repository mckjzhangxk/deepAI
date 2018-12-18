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
            System.out.println(s.getPath()+" blocksize(M):"+s.getBlockSize()/1024/1024+" copys:"+s.getReplication());

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
    public static Configuration getConfigure(String prefix){

        //设置true会跑到src目录下找XML文件,false不会去找
        Configuration conf=new Configuration(false);

        //这里是手动制定文件路径
//        conf.addResource(new Path(""));
//        conf.addResource(new Path(""));
//        System.out.println(HDFS.class.getResource("/conf/ha/core-site.xml"));
        conf.addResource(HDFS.class.getResource(prefix+"/core-site.xml"));
        conf.addResource(HDFS.class.getResource(prefix+"/hdfs-site.xml"));

//        System.out.println(conf.get("fs.defaultFS"));
//        System.out.println(conf.get("dfs.replication"));
//        System.out.println(conf);
        return conf;
    }
    public static void main(String[] gras) throws IOException, InterruptedException {
        //注意,必须设置环境变量HADOOP_USER_NAME 与hdfs保持一致
        System.out.println(System.getenv("HADOOP_USER_NAME"));

        Configuration conf=getConfigure("conf/ha");
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
        p=new Path("/face.jpeg");
        inputStream=new FileInputStream("/home/zxk/IdeaProjects/hadoop/src/data/timg.jpeg");
        outputStream = fs.create(p);
        IOUtils.copyBytes(inputStream,outputStream,4096,true);


        blockLocation(fs,"/jdk-8u131-linux-x64.tar.gz");

//        listFiles(fs);


        fs.delete(p,true);
//        listFiles(fs);


        fs.close();
    }
}
