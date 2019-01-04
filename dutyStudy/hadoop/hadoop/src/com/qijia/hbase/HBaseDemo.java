package com.qijia.hbase;

import com.qijia.conf.MyConfigure;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.*;
import org.apache.hadoop.hbase.client.*;
import org.apache.hadoop.hbase.filter.CompareFilter;
import org.apache.hadoop.hbase.filter.FilterList;
import org.apache.hadoop.hbase.filter.PrefixFilter;
import org.apache.hadoop.hbase.filter.SingleColumnValueExcludeFilter;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

/**
 * Created by zhangxk on 19-1-4.
 */
public class HBaseDemo {
    private static Configuration configure=MyConfigure.getConfigure("ha_hbase");
    private static SimpleDateFormat sdf=new SimpleDateFormat("yyyyMMdd");

    //https://hbase.apache.org/1.2/book.html#hbase_apis
    public static HBaseAdmin createConnection() throws IOException {
        HBaseAdmin admin=new HBaseAdmin(configure);
        return admin;
    }

    public static HTable createTable(HBaseAdmin admin,String TABLE_NAME,String CFNAME,boolean dropIfExist) throws IOException {
        HTableDescriptor dsc=new HTableDescriptor(TableName.valueOf(TABLE_NAME));

        //删除已经创建的表
        if(admin.tableExists(dsc.getTableName()) &&dropIfExist){
            admin.disableTable(dsc.getTableName());
            admin.deleteTable(dsc.getTableName());
        }
        //加入一个column family
        HColumnDescriptor cf=new HColumnDescriptor(CFNAME);
        dsc.addFamily(cf);
        admin.createTable(dsc);

        return new HTable(configure,dsc.getTableName());
    }

    public static void putOne(HTable tb,String cfName,String rowkey) throws InterruptedIOException, RetriesExhaustedWithDetailsException {
        Put put=new Put(rowkey.getBytes());
        put.add(cfName.getBytes(),"name".getBytes(),"zxk".getBytes());
        put.add(cfName.getBytes(),"age".getBytes(),"29".getBytes());
        put.add(cfName.getBytes(),"sex".getBytes(),"M".getBytes());

        tb.put(put);
    }

    public static void getOne(HTable tb,String cfName,String rowkey) throws IOException {
        Get get=new Get(rowkey.getBytes());
//        get.addColumn("member".getBytes(),"name".getBytes());
        Result result = tb.get(get);
        while (result.advance()){
            result.current();
//            Cell cell = result.getColumnLatestCell(cfName.getBytes(), "name".getBytes());
            Cell cell=result.current();
            String cf=new String(cell.getFamilyArray());
            String field=new String(cell.getQualifierArray());
            String value = new String(CellUtil.cloneValue(cell));
            System.out.println(cf+":"+field+"  "+value);


        }
    }

    /**
     * 1000个用户,每个人10条记录
     * rowkey
     * duration:0-100 minutes
     * type:0/1
     * phone:
     * */
    public static void putList(HTable tb,String cfName) throws InterruptedIOException, RetriesExhaustedWithDetailsException {
        List<Put> list=new ArrayList<>();
        for(int n=0;n<1000;n++){
            String phonenum1=getPhone("186");
            for(int k=0;k<10;k++){
                String duration=""+ (new Random().nextInt(100)+1);
                String type=""+ (new Random().nextInt(2));
                String phonenum2=getPhone("139");
                Date t=getRandomDate();
                String time=sdf.format(t);
                String rowkey=(Long.MAX_VALUE-t.getTime())+"";

                Put put=new Put(rowkey.getBytes());
                put.add(cfName.getBytes(),"duration".getBytes(),duration.getBytes());
                put.add(cfName.getBytes(),"type".getBytes(),type.getBytes());
                put.add(cfName.getBytes(),"phonenum2".getBytes(),phonenum2.getBytes());
                put.add(cfName.getBytes(),"time".getBytes(),time.getBytes());
            }
        }
        tb.put(list);
    }

    public static void scan(HTable tb,String cfName) throws IOException, ParseException {
        String phoneNum = "18676604687";
        String startRow = phoneNum + "_" + (Long.MAX_VALUE - sdf.parse("20180301").getTime());
        String stopRow = phoneNum + "_" + (Long.MAX_VALUE - sdf.parse("20170201").getTime());

        Scan scan=new Scan();
        scan.setStartRow(startRow.getBytes());
        scan.setStopRow(stopRow.getBytes());

        ResultScanner scanner = tb.getScanner(scan);

        for (Result rs:scanner) {
            String duration=new String(
            CellUtil.cloneValue(rs.getColumnLatestCell(cfName.getBytes(),"duration".getBytes())));
            String type=new String(
                    CellUtil.cloneValue(rs.getColumnLatestCell(cfName.getBytes(),"type".getBytes())));
            String phonenum2=new String(
                    CellUtil.cloneValue(rs.getColumnLatestCell(cfName.getBytes(),"phonenum2".getBytes())));
            String time=new String(
                    CellUtil.cloneValue(rs.getColumnLatestCell(cfName.getBytes(),"time".getBytes())));
            System.out.println(duration+"-"+type+"-"+phonenum2+"-"+time);
        }
    }

    public static void find(HTable tb,String cfName) throws IOException {
        FilterList filterList=new FilterList();
        PrefixFilter filter1=new PrefixFilter("ssss".getBytes());
        filterList.addFilter(filter1);
        SingleColumnValueExcludeFilter filter2=new SingleColumnValueExcludeFilter(
                cfName.getBytes(),
                "type".getBytes(),
                CompareFilter.CompareOp.EQUAL,
                "1".getBytes()
        );
        filterList.addFilter(filter2);

        Scan scan=new Scan();
        scan.setFilter(filterList);
        ResultScanner rs = tb.getScanner(scan);

        for (Result r:rs) {
            String type=new String(CellUtil.cloneValue(r.getColumnLatestCell(cfName.getBytes(),"type".getBytes())));
            System.out.println(type);
        }
    }
    private static String getPhone(String prefix) {
        return prefix+String.format("%08d", new Random().nextInt(99999999));
    }

    public static void main(String[] args) throws IOException {
//        HBaseAdmin admin=createConnection();
//        HTable table=createTable(admin,"myfamily","member",true);
//        putOne(table,"member","1");
//        getOne(table,"member","1");
        SimpleDateFormat df=new SimpleDateFormat("yyyyMMdd");
        System.out.println(df.format(getRandomDate()));
    }

    public static Date getRandomDate() {
        long l=3600*1000*24*15;

        long seed=new Random().nextInt(3000);

        return new Date(seed*l);
//        System.out.println(new Date(0));;
    }
}
