package com.qijia.mapreduce.tfidf;


import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by zhangxk on 18-12-27.
 */
public class Weibo implements WritableComparable<Weibo> {
    private String word;
    private String id;

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    @Override
    public int compareTo(Weibo o) {
        return word.compareTo(o.word);
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeUTF(word+":"+id);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        String s=dataInput.readUTF();
        String[] sps=s.split(":");
        setWord(sps[0]);
        setId(sps[1]);
    }
    @Override
    public String toString() {
        return  word +":"+id;
    }
}
