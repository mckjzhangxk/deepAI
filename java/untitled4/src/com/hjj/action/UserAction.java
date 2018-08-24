package com.hjj.action;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.hjj.dao.MysqlDAO;
import com.hjj.model.*;
import com.opensymphony.xwork2.ActionSupport;
import com.opensymphony.xwork2.ModelDriven;

public class UserAction extends ActionSupport implements ModelDriven<User>{
    private User user = new User();
    List<User> users = new ArrayList<User>();


    public List<User> getUsers() {
        return users;
    }

    public void setUsers(List<User> users) {
        this.users = users;
    }

    public String add() throws IOException{
        System.out.println(user);
        MysqlDAO dao = new MysqlDAO();
        dao.insertUser(user);
        return "add";
    }

    public String get() throws IOException{
        MysqlDAO dao = new MysqlDAO();
        users = dao.getUsers(user);
        return "get";
    }


    public User getModel() {
        if(user == null){
            user = new User();
        }
        return user;
    }

}