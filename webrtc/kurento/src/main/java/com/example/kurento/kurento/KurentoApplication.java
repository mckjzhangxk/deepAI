package com.example.kurento.kurento;

import org.kurento.client.KurentoClient;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;
import org.springframework.web.socket.server.standard.ServletServerContainerFactoryBean;


@SpringBootApplication
@EnableWebSocket
public class KurentoApplication implements WebSocketConfigurer {



    @Bean
    public HelloHandler handler()
    {
        return new HelloHandler();
    }


    @Bean
    public KurentoClient kurentoClient()
    {
        KurentoClient kurentoClient = KurentoClient.create();
        return kurentoClient;
    }
    @Bean
    public  MyService myService(){
        return new MyService();
    }
    @Bean
    public ServletServerContainerFactoryBean createServletServerContainerFactoryBean() {
        ServletServerContainerFactoryBean container = new ServletServerContainerFactoryBean();
        container.setMaxTextMessageBufferSize(32768);
        return container;
    }


    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(handler(),"/helloworld");
    }



    public static void main(String[] args) {
        SpringApplication.run(KurentoApplication.class, args);
    }
}
