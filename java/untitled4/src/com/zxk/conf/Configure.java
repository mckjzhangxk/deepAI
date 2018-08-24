package com.zxk.conf;

import java.io.IOException;
import java.util.Properties;

public class Configure {
    public static String api_url="";
    static {
        Properties pps = new Properties();

        try {
            pps.load(Configure.class.getResourceAsStream("conf.properties"));
            api_url=pps.getProperty("api_url");

        } catch (IOException var2) {
            var2.printStackTrace();
        }

    }
}
