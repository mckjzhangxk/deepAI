package com.qijia.weather;

import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhangxk on 18-12-24.
 */
public class TQGenerator {
    private static String[] strs={
            "1949-10-01 14:21:02\t34c"
            ,"1949-10-01 19:21:02\t38c"
            ,"1949-10-02 14:01:02\t36c"
            ,"1950-01-01 11:21:02\t32c"
            ,"1950-10-01 12:21:02\t37c"
            ,"1951-12-01 12:21:02\t23c"
            ,"1950-10-02 12:21:02\t41c"
            ,"1950-10-03 12:21:02\t27c"
            ,"1951-07-01 12:21:02\t45c"
            ,"1951-07-02 12:21:02\t46c"
            ,"1951-07-03 12:21:03\t47c"};

    public static void main(String[] args) throws IOException {
        int N=10000;
        int n=strs.length;
        List <String> ls=new ArrayList<>();
        for (int i=0;i<N;i++){
            int k= (int) (Math.random()*n);
            ls.add(strs[k]);
        }

        FileWriter fs=new FileWriter("tq.txt");
        for(String s:ls)
            fs.write(s+"\n");
        fs.close();
    }
}
