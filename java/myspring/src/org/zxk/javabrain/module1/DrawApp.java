package org.zxk.javabrain.module1;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class DrawApp {
    /*
    * spring 的工厂模式
    *https://www.youtube.com/watch?v=ZxLaEovze3M&index=5&list=PLC97BDEFDCDD169D7
    * */
    public static void main(String[] argv){
        //Bean工厂
//        BeanFactory factory=null;
        //过时的方法，XMLBeanFactory 在项目路径找xml
//        factory=new XmlBeanFactory(new FileSystemResource("spring.xml"));

        //classpathxmlapplicationcontext 在类路径找xml
//        factory=new ClassPathXmlApplicationContext("spring.xml");

        //app context 是比factory更高级的接口
        ApplicationContext cx=new ClassPathXmlApplicationContext("spring.xml");
        Triangle triangle= (Triangle) cx.getBean("triangle");

        triangle.draw();

    }
}
