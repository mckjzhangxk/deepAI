package com.qijia.mapreduce.weather;

import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by zhangxk on 18-12-24.
 */
public class TQ implements WritableComparable<TQ>{
    private int year;
    private int mouth;
    private int day;
    private int temperature;

    public int getYear() {
        return year;
    }

    public void setYear(int year) {
        this.year = year;
    }

    public int getMouth() {
        return mouth;
    }

    public void setMouth(int mouth) {
        this.mouth = mouth;
    }

    public int getDay() {
        return day;
    }

    public void setDay(int day) {
        this.day = day;
    }

    public int getTemperature() {
        return temperature;
    }

    public void setTemperature(int temperature) {
        this.temperature = temperature;
    }

    //mapper输出key,val对,key要决定如何比较,在BUFFER满了之后,对结果进行quicksort,然后输出到磁盘
    //这里选择按照年月日排序
    @Override
    public int compareTo(TQ o) {
        int c1=Integer.compare(year,o.year);
        if(c1!=0)
            return c1;
        int c2=Integer.compare(mouth,o.mouth);
        if(c2!=0)
            return c2;
        int c3=Integer.compare(day,o.day);
        return c3;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        //失误写成write造成IO错误
        dataOutput.writeInt(year);
        dataOutput.writeInt(mouth);
        dataOutput.writeInt(day);
        dataOutput.writeInt(temperature);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        year=dataInput.readInt();
        mouth=dataInput.readInt();
        day=dataInput.readInt();
        temperature=dataInput.readInt();
    }

    @Override
    public String toString() {
        return  year +"-"+ mouth +"-"+ day ;
    }
}
