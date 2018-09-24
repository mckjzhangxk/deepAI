package org.zxk.javabrain.module1;

import java.util.List;

public class Triangle {


    public Triangle(String name) {
        this.name = name;
    }

    public Triangle(int age) {
        this.age = age;
    }

    public Triangle(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public void draw(){
        System.out.println(name+"("+age+"):Drawing Triangle");
        System.out.println(p1);
        System.out.println(p2);
        System.out.println(p3);
        System.out.println(pts);

    }

    public String getName() {
        return name;
    }


    public Point getP1() {
        return p1;
    }

    public void setP1(Point p1) {
        this.p1 = p1;
    }

    public Point getP2() {
        return p2;
    }

    public void setP2(Point p2) {
        this.p2 = p2;
    }

    public Point getP3() {
        return p3;
    }

    public void setP3(Point p3) {
        this.p3 = p3;
    }

    public List<Point> getPts() {
        return pts;
    }

    public void setPts(List<Point> pts) {
        this.pts = pts;
    }

    private String name;
    private int age;
    private Point p1;
    private Point p2;
    private Point p3;
    private List<Point> pts;

}
