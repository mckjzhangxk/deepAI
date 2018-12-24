package com.qijia.weather;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

/**
 * Created by zhangxk on 18-12-24.
 */
public class TQSortComparator extends WritableComparator {
    public TQSortComparator() {
        super(TQ.class,true);
    }

    @Override
    public int compare(WritableComparable a, WritableComparable b) {
        TQ q1= (TQ) a;
        TQ q2= (TQ) b;

        int c1=Integer.compare(q1.getYear(),q2.getYear());
        if(c1!=0)
            return c1;

        int c2=Integer.compare(q1.getMouth(),q2.getMouth());
        if(c2!=0)
            return c2;
        return Integer.compare(q2.getTemperature(),q1.getTemperature());
    }
}
