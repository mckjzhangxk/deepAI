package com.qijia.hive;

import java.sql.*;

/**
 * Created by zhangxk on 18-12-29.
 * https://cwiki.apache.org/confluence/display/Hive/HiveServer2+Clients#HiveServer2Clients-JDBC
 */
public class HiveClient {
    public static Connection getConnect() throws ClassNotFoundException, SQLException {
        Class.forName("org.apache.hive.jdbc.HiveDriver");

        Connection conn = DriverManager.getConnection("jdbc:hive2://node2:10000/default", "root", "123");
        return conn;
    }
    public static void main(String[] args) throws SQLException, ClassNotFoundException {
        Connection connect = getConnect();
        String sql="select avg(id) from psn";
        PreparedStatement statement = connect.prepareStatement(sql);
        ResultSet resultSet = statement.executeQuery();
        while (resultSet.next()){

            System.out.println("habit:"+resultSet.getString(1));
        }
//        statement.close();
        connect.close();
    }
}
