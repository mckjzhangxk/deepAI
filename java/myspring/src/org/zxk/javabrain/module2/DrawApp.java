package org.zxk.javabrain.module2;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;


/*
* In contrast to the other scopes, Spring does not manage the complete lifecycle of a prototype bean:
* the container instantiates, configures, and otherwise assembles a prototype object, and hands it to the client,
 * with no further record of that prototype instance.
  * Thus, although initialization lifecycle callback methods are called on all objects regardless of scope,
   * in the case of prototypes, configured destruction lifecycle callbacks are not called.﻿
* */
public class DrawApp {
    /*
    * 14.bean 的生命周期,init-method,destory-method,default-init-method,initializeBean,DisposalBean...
    *
    * 15.BeanPostProcessor,告诉spring,在创建bean前后要做什么工作，可以认为一个全局处理器
    * */
    public static void main(String[] argv){
        AbstractApplicationContext context= new ClassPathXmlApplicationContext("spring2.xml");
        context.registerShutdownHook();
        //单例与原型模式的比较
        Triangle triangle = (Triangle) context.getBean("triangle");
        System.out.println(triangle);
//        triangle = (Triangle) context.getBean("triangle");
//        System.out.println(triangle);

    }
}
