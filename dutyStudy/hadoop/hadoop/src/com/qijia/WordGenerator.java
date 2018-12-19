package com.qijia;

import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;

public class WordGenerator {
    public static void main(String[] args) throws IOException {
        int N=9000000;

        FileWriter ous=new FileWriter("words.txt");
        String line="I like machine learning\n";
        for(int i=0;i<N;i++){
            ous.write(line);
        }
        ous.close();
    }
}
