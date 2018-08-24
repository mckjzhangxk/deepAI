package com.zxk.utils;

import com.sun.xml.internal.messaging.saaj.util.ByteOutputStream;

import java.io.*;
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
        // ����ͨ�õ���������
        connection.setRequestProperty("accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8");
        connection.setRequestProperty("Connection", "Keep-Alive");
        connection.setRequestProperty("Accept-Encoding", "gzip,deflate");
        connection.setRequestProperty("Accept-Language", "zh-CN,zh;q=0.8");
        connection.setRequestProperty("Content-Type", "application/json");

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


    public static byte[] accessUrl(String uri) throws IOException {
        URL u = null;
        u = new URL(uri);
        InputStream ins = u.openStream();

        ByteArrayOutputStream ous = new ByteArrayOutputStream();
        int len = 0;
        byte[] buf = new byte[1024];
        while ((len = ins.read(buf)) != -1) {
            ous.write(buf, 0, len);
        }
        ins.close();
        ous.close();
        return ous.toByteArray();
    }



    public static Map<String, String> sign(String jsapi_ticket, String url) {
        Map<String, String> ret = new HashMap<String, String>();
        String nonce_str = create_nonce_str();
        String timestamp = create_timestamp();
        String string1;
        String signature = "";

        string1 = "jsapi_ticket=" + jsapi_ticket +
                "&noncestr=" + nonce_str +
                "&timestamp=" + timestamp +
                "&url=" + url;

        try {
            MessageDigest crypt = MessageDigest.getInstance("SHA-1");
            crypt.reset();
            crypt.update(string1.getBytes("UTF-8"));
            signature = byteToHex(crypt.digest());
        }
        catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        ret.put("url", url);
        ret.put("jsapi_ticket", jsapi_ticket);
        ret.put("nonceStr", nonce_str);
        ret.put("timestamp", timestamp);
        ret.put("signature", signature);

        return ret;
    }

    public static String byteToHex(final byte[] hash) {
        Formatter formatter = new Formatter();
        for (byte b : hash) {
            formatter.format("%02x", b);
        }
        String result = formatter.toString();
        formatter.close();
        return result;
    }

    private static String create_nonce_str() {
        return UUID.randomUUID().toString();
    }
    private static String create_timestamp() {
        return Long.toString(System.currentTimeMillis() / 1000);
    }

    public static void main(String[] args) throws IOException {
        String url="https://dopen.weimob.com/fuwu/b/oauth2/token?code=KVP587&grant_type=authorization_code&client_id=A07DFBCFD051CBE408E3945BF4AD52FE&client_secret=88F5B45ACFF3B1A6A2FD43580A31FD1A&redirect_uri=https://www.baidu.com/";
        String accessToken="";
        String param="{}";
        String ret=WxUtils.sendPost(url,param);
        System.out.println(ret);
    }
}
