package com.middle;

import com.sun.xml.internal.messaging.saaj.util.ByteOutputStream;

import javax.net.ssl.HttpsURLConnection;
import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Formatter;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class WxUtils {
    /**
     * ����post����
     *
     * @param urlNameString
     * @param param
     * @return
     * @throws IOException
     */
    public static String sendPost(String urlNameString, String param) throws IOException {
        URL realUrl = new URL(urlNameString);
        URLConnection connection = realUrl.openConnection();

        connection.setRequestProperty("accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8");
        connection.setRequestProperty("Connection", "Keep-Alive");
        connection.setRequestProperty("Accept-Encoding", "gzip,deflate");
        connection.setRequestProperty("Accept-Language", "zh-CN,zh;q=0.8");
        connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");


        byte[] buf = param.getBytes("utf-8");
        int len = buf.length;
        connection.setRequestProperty("Content-Length", buf.length + "");
        connection.setDoOutput(true);
        connection.setDoInput(true);

        OutputStream ous = connection.getOutputStream();
        ous.write(buf);
        ous.flush();
        ous.close();

        InputStream ins1 = connection.getInputStream();
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        while ((len = ins1.read(buf)) != -1) {
            o.write(buf, 0, len);
        }
        ins1.close();
        param = new String(o.toByteArray());

        return param;
    }

    public static byte[] accessUrl(String uri){
        URL u = null;
        try {
            u = new URL(uri);
            InputStream ins = u.openStream();

            ByteOutputStream ous = new ByteOutputStream();
            int len = 0;
            byte[] buf = new byte[1024];
            while ((len = ins.read(buf)) != -1) {
                ous.write(buf, 0, len);
            }
            ins.close();
            ous.close();


            return ous.getBytes();
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }





    public static void main(String[] args) throws Exception {
        String url="https://dopen.weimob.com/fuwu/b/oauth2/token?code="+WxConstant.getCode()+"&grant_type=authorization_code&client_id=A07DFBCFD051CBE408E3945BF4AD52FE&client_secret=88F5B45ACFF3B1A6A2FD43580A31FD1A&redirect_uri=https://www.baidu.com/";
        String ret=WxUtils.sendPost(url,"{}");
        System.out.println(new String(ret));
    }
}
