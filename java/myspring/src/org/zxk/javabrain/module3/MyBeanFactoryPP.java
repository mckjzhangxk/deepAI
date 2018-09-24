package org.zxk.javabrain.module3;

import org.springframework.beans.BeansException;
import org.springframework.beans.factory.config.BeanFactoryPostProcessor;
import org.springframework.beans.factory.config.ConfigurableListableBeanFactory;

public class MyBeanFactoryPP implements BeanFactoryPostProcessor {
    /**
     * 自定义的替换PostProcessor
     * */
    @Override
    public void postProcessBeanFactory(ConfigurableListableBeanFactory beanFactory) throws BeansException {
        System.out.println(beanFactory);

        Point p1 = (Point) beanFactory.getBean("p1");
        p1.setX(1000.0);
        p1.setY(500);
    }
}
