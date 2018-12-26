package com.qijia.weather;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

/**
 * Created by zhangxk on 18-12-24.
 */
public class MyMapper extends Mapper<LongWritable,Text,TQ,IntWritable>{
    private TQ key=new TQ();
    private IntWritable value=new IntWritable();
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        try {
            String s=value.toString();
            String[] splits=s.split("\t");
            SimpleDateFormat sf=new SimpleDateFormat("yyyy-MM-dd");
            Date date = null;
            date = sf.parse(splits[0]);
            Calendar calendar=Calendar.getInstance();
            calendar.setTime(date);

            int year=calendar.get(Calendar.YEAR);
            int mouth=calendar.get(Calendar.MONTH)+1;
            int day=calendar.get(Calendar.DAY_OF_MONTH);
            int temperature=Integer.parseInt(splits[1].substring(0,splits[1].length()-1));

            this.key.setYear(year);
            this.key.setMouth(mouth);
            this.key.setDay(day);
            this.key.setTemperature(temperature);
            this.value.set(temperature);

            context.write(this.key,this.value);
        } catch (ParseException e) {
            e.printStackTrace();
        }


    }
}
