package org.zxk.javabrain.module2;

import org.springframework.beans.BeansException;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.zxk.javabrain.module2.Point;

import java.util.List;

public class Triangle implements ApplicationContextAware/*, InitializingBean, DisposableBean*/ {
    private ApplicationContext context;
    private Point pt1;
    private Point pt2;
    private Point pt3;

    private List<Point> pts;

    public Point getPt1() {
        return pt1;
    }

    public void setPt1(Point pt1) {
        this.pt1 = pt1;
    }

    public Point getPt2() {
        return pt2;
    }

    public void setPt2(Point pt2) {
        this.pt2 = pt2;
    }

    public Point getPt3() {
        return pt3;
    }

    public void setPt3(Point pt3) {
        this.pt3 = pt3;
    }

    public List<Point> getPts() {
        return pts;
    }

    public void setPts(List<Point> pts) {
        this.pts = pts;
    }

    @Override
    public String toString() {
        return "Triangle{" +
                "context=" + context +
                ", pt1=" + pt1 +
                ", pt2=" + pt2 +
                ", pt3=" + pt3 +
                ", pts=" + pts +
                '}';
    }

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
        context=applicationContext;
        System.out.println("-------获得上下文");
    }

//    @Override
//    public void destroy() throws Exception {
//        System.out.println("destory bean");
//    }
//
//    @Override
//    public void afterPropertiesSet() throws Exception {
//        System.out.println("initial bean");
//    }

    public void myinit(){
        System.out.println("myinit bean");
    }
    public void mycleanup(){
        System.out.println("mycleanup bean");
    }
}
