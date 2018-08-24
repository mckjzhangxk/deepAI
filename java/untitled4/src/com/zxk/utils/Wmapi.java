package com.zxk.utils;

import com.zxk.conf.Configure;
import net.sf.json.JSONObject;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Wmapi {

    private String getAccessToken(){
        try {
            byte[] bytes = WxUtils.accessUrl(Configure.api_url);
            return new String(bytes);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public Map call(String apiKey,Map param){
        String api_url=apilist.get(apiKey);
        if(api_url==null)
            return null;

        String accessToken=getAccessToken();
        api_url=api_url+accessToken;


        String param_str=JSONObject.fromObject(param).toString();
        try {
            String ret_str = WxUtils.sendPost(api_url, param_str);
            Map retObj=JSONObject.fromObject(ret_str);
            return retObj;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }



    private Map<String,String> apilist=new HashMap<String, String>();
    public Wmapi(){
        apilist.put(queryStoreList,"https://dopen.weimob.com/api/1_0/o2o/store/queryStoreList?accesstoken=");
        apilist.put(queryStoreDetail,"http://dopen.weimob.com/api/1_0/o2o/store/queryStoreDetail?accesstoken=");
        apilist.put(queryOrderList,"https://dopen.weimob.com/api/1_0/o2o/order/queryOrderList?accesstoken=");
    }
    //查询门店类目列表
    public static String queryStoreList="F1";
    //查询门店详情
    public static String queryStoreDetail="F2";

    //查询订单列表
    public static String queryOrderList="F3";


    public static void main(String[] args){
        Wmapi api=new Wmapi();
        Map param=new HashMap();
            param.put("merchantId","100000019033");
            param.put("storeId",23809033);
            param.put("pageNum",1);
            param.put("pageSize",10);
        Map ret = api.call(queryOrderList, param);
        System.out.println(ret);
    }
}
