package com.hjj.dao;

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

import org.apache.ibatis.io.Resources;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.session.SqlSessionFactoryBuilder;

import com.hjj.model.User;

public class MysqlDAO {

    private SqlSessionFactory sessionFactory = null;
    private SqlSession session = null;

    public MysqlDAO() throws IOException{
        String resource = "conf.xml";

        Reader reader = Resources.getResourceAsReader(resource);

        sessionFactory = new SqlSessionFactoryBuilder().build(reader);

    }
    public List<User> getUsers(){

        List<User> users = new ArrayList<User>();

        session = sessionFactory.openSession();

        String statement = "com.hjj.dao.userMapper.getUser";

        users = session.selectList(statement);

        session.commit();

        return users;
    }

    public List<User> getUsers(User u){

        List<User> users = new ArrayList<User>();



        session = sessionFactory.openSession();
        String statement = "com.hjj.dao.userMapper.getUser";
        users = session.selectList(statement,u);

        session.commit();

        return users;
    }

    public void insertUser(User user){

        session = sessionFactory.openSession();

        String statement = "com.hjj.dao.userMapper.insertUser";

        int row = session.insert(statement, user);
        session.commit();
        System.out.println(row);
    }
}