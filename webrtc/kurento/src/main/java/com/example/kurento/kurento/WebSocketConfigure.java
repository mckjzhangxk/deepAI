package com.example.kurento.kurento;

import com.example.kurento.kurento.handler.BroadcastHandler;
import com.example.kurento.kurento.handler.HelloHandler;
import com.example.kurento.kurento.service.MyService;
import org.kurento.client.KurentoClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

/**
 * Created by mathai on 21-5-26.
 */
@Configuration
public class WebSocketConfigure implements WebSocketConfigurer {
    @Bean
    WebSocketHandler getHelloWorldHandler(){
        return new HelloHandler();
    }
    @Bean
    WebSocketHandler getBroadcastHandler(){
        return new BroadcastHandler();
    }


    @Bean
    public KurentoClient kurentoClient()
    {
        KurentoClient kurentoClient = KurentoClient.create();
        return kurentoClient;
    }
    @Bean
    public MyService myService(){
        return new MyService();
    }

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(getHelloWorldHandler(),"/helloworld");
        registry.addHandler(getBroadcastHandler(),"/call");
    }



}
