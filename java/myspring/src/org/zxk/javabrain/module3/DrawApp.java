package org.zxk.javabrain.module3;

import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;



public class DrawApp {
    /*
    https://www.youtube.com/watch?v=szNWTBlewQI&list=PLC97BDEFDCDD169D7&index=16
    * 16.创建post bean factory,一个重要的应用就是属性替代
    *
    * 15.BeanPostProcessor,告诉spring,在创建bean前后要做什么工作，可以认为一个全局处理器
    * */
    public static void main(String[] argv){
        AbstractApplicationContext context= new ClassPathXmlApplicationContext("spring3.xml");
        Point p1= (Point) context.getBean("p1");
        System.out.println(p1);
    }
}
