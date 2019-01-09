package com.qijia.zookeeper;


import org.apache.zookeeper.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by zhangxk on 19-1-9.
 */
public class ZookeeperClient {
    public static Logger logger= LoggerFactory.getLogger(ZookeeperClient.class);

    private ZooKeeper m_zoo;
    private Watcher water=new Watcher() {
        @Override
        public void process(WatchedEvent watchedEvent) {
            logger.info("时间到达了:"+watchedEvent.getType());
        }
    };

    private void connect(String uri) throws IOException {
            m_zoo=new ZooKeeper(uri,20*1000,water);
    }
    private void close() throws IOException, InterruptedException {
        m_zoo.close();
    }


    private void create(String path,String data,boolean persistent) throws KeeperException, InterruptedException {
        if(persistent){
            m_zoo.create(path,data.getBytes(), ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
        }else {
            m_zoo.create(path,data.getBytes(), ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.EPHEMERAL);
        }
    }
    private void delete(String path) throws KeeperException, InterruptedException {
        m_zoo.delete(path,-1);
    }
    private void get(String path) throws KeeperException, InterruptedException {
        byte[] data = m_zoo.getData(path, null, null);
        logger.info("获得数据:"+path+"="+new String(data));
    }
    private void set(String path,String data) throws KeeperException, InterruptedException {
        m_zoo.setData(path,data.getBytes(),-1);
    }

    public static void main(String[] args) throws IOException, InterruptedException, KeeperException {
        ZookeeperClient client=new ZookeeperClient();
        client.connect("node2:2181,node3:2181,node4:2181");

        String path="myzxk";
        String data="zxkdata";

        //二选1
        client.create(path,data,true);
//        client.delete(path);

        //
        client.get(path);
        Thread.sleep(130000);
        //关闭刚才链接的客户端,看看能不能获得
        client.get(path);

        client.set(path,"xxxxxxxxxxxxxxxxxxxx");

        client.close();
    }
}
