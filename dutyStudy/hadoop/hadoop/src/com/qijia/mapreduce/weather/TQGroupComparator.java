package com.qijia.mapreduce.weather;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

/**
 * Created by zhangxk on 18-12-24.
 */
public class TQGroupComparator extends WritableComparator {
    //要重写构造,告诉要咋比较,不然会报buffer 空指针
    public TQGroupComparator() {
        super(TQ.class,true);
    }
    //相同年月是一组
    @Override
    public int compare(WritableComparable a, WritableComparable b) {
        TQ q1= (TQ) a;
        TQ q2= (TQ) b;

        int c1=Integer.compare(q1.getYear(),q2.getYear());
        if(c1!=0)
            return c1;

        int c2=Integer.compare(q1.getMouth(),q2.getMouth());
        return c2;

    }

}
