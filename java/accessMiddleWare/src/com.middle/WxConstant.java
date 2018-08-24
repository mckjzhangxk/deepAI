package com.middle;

import java.io.IOException;
import java.util.Properties;


/**
 * Created by Administrator on 15-12-3.
 */
public class WxConstant {
    static {
    	Properties pps = new Properties();
    	try {
			pps.load(WxConstant.class.getResourceAsStream("wm.properties"));
			appID = pps.getProperty("appID");
			appsecret = pps.getProperty("appsecret");
			redirect_uri= pps.getProperty("redirect_uri");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
	public static String getCode() throws IOException {
		Properties pps = new Properties();
		pps.load(WxConstant.class.getResourceAsStream("wm.properties"));
		code=pps.getProperty("code");
		return code;
	}
    public static  String appID;
    public static  String appsecret;
    public static  String code="Gegqc";
	public static String redirect_uri;
}
