package com.qijia.hive;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.io.Text;
/**
 * Created by zhangxk on 18-12-29.
 *
 * add jar /root/zxk.jar;
 * CREATE TEMPORARY FUNCTION myfilter AS 'com.qijia.hive.SensitiveFilter';
 */
public class SensitiveFilter extends UDF{
    public Text evaluate(final Text s) {
        if (s == null) {
            return null;
        }
        String str = s.toString().substring(0, 1) + "***";
        return new Text(str);
    }
}
