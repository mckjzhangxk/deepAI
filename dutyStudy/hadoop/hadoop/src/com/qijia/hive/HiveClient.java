package com.qijia.hive;

import java.sql.*;

/**
 * Created by zhangxk on 18-12-29.
 * https://cwiki.apache.org/confluence/display/Hive/HiveServer2+Clients#HiveServer2Clients-JDBC
 *
 *
 * # To run the program using remote hiveserver in non-kerberos mode, we need the following jars in the classpath
 * # from hive/build/dist/lib
 * #     hive-jdbc*.jar
 * #     hive-service*.jar
 * #     libfb303-0.9.0.jar
 * #        libthrift-0.9.0.jar
 * #     log4j-1.2.16.jar
 * #     slf4j-api-1.6.1.jar
 * #    slf4j-log4j12-1.6.1.jar
 * #     commons-logging-1.0.4.jar
 */
public class HiveClient {
    public static Connection getConnect() throws ClassNotFoundException, SQLException {
        Class.forName("org.apache.hive.jdbc.HiveDriver");

        Connection conn = DriverManager.getConnection("jdbc:hive2://node5:10000/default", "root", "123");
        return conn;
    }
    public static void main(String[] args) throws SQLException, ClassNotFoundException {
        Connection connect = getConnect();
        String sql="select avg(id) from psn";
        PreparedStatement statement = connect.prepareStatement(sql);
        ResultSet resultSet = statement.executeQuery();
        while (resultSet.next()){

            System.out.println("avg:"+resultSet.getString(1));
        }
//        statement.close();
        connect.close();
    }
}
