package com.qijia.weather;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

/**
 * Created by zhangxk on 18-12-24.
 */
public class TQSortComparator extends WritableComparator {
    //要重写构造,告诉要咋比较
    public TQSortComparator() {
        super(TQ.class,true);
    }
    /**
     * 对map的输出进行排序,这里排序的次序是year-mouth-temperature,其中temperature是下降序列
     * */
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
