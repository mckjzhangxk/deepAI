package com.middle;
import net.sf.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Date;
import java.util.Map;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.omg.CORBA.SystemException;

import com.sun.xml.internal.messaging.saaj.util.ByteOutputStream;


public class MiddleServlet extends HttpServlet {

	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		String str=getAccessToken();
		response.getOutputStream().write(str.getBytes("utf-8"));
		response.getOutputStream().flush();
	}
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		doGet(request,response);
	}
	private static String accessToken="";
    private static long updateAccessTokenTimastamp = 0;
    private static String refleshToken="";


    //update TokenEvery 2hour
    private synchronized static String getAccessToken() {
        long currentTime = new Date().getTime();
        System.out.println("re:"+refleshToken);
        if (refleshToken.equals("") ||(currentTime - updateAccessTokenTimastamp) > 3600 * 1000) {
            updateAccessToken();
            updateAccessTokenTimastamp = currentTime;
        }
        return accessToken;
    } 
	private static void updateAccessToken() {

	        URL u = null;
	        try {
                String url=null;
                if(refleshToken.equals("")){
                    url="https://dopen.weimob.com/fuwu/b/oauth2/token?code="+WxConstant.getCode()+"&grant_type=authorization_code&client_id="+WxConstant.appID+"&client_secret="+WxConstant.appsecret+"&redirect_uri="+WxConstant.redirect_uri;
                }else {
                    url="https://dopen.weimob.com/fuwu/b/oauth2/token?grant_type=refresh_token&client_id="+WxConstant.appID+"&client_secret="+WxConstant.appsecret+"&refresh_token="+refleshToken;
                }



	            String str = WxUtils.sendPost(url,"{}");
	            Map m = JSONObject.fromObject(str);
	            accessToken = (String) m.get("access_token");
                refleshToken = (String) m.get("refresh_token");
	        } catch (Exception e) {
	            e.printStackTrace();
	        }

	    }

}
