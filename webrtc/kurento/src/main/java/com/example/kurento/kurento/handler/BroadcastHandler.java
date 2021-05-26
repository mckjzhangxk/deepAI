package com.example.kurento.kurento.handler;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.kurento.client.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;


/**
 * Created by mathai on 21-5-26.
 */

class UserSession{
    public UserSession(WebSocketSession session, WebRtcEndpoint webRtcEndpoint) {
        this.session=session;
        this.webRtcEndpoint=webRtcEndpoint;
    }

    public WebRtcEndpoint getWebRtcEndpoint() {
        return webRtcEndpoint;
    }

    public void setWebRtcEndpoint(WebRtcEndpoint webRtcEndpoint) {
        this.webRtcEndpoint = webRtcEndpoint;
    }

    public WebSocketSession getSession() {
        return session;
    }

    public void setSession(WebSocketSession session) {
        this.session = session;
    }

    private WebRtcEndpoint webRtcEndpoint;
    private WebSocketSession session;
}
public class BroadcastHandler extends TextWebSocketHandler {
    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        JSONObject p = JSON.parseObject(message.getPayload());
        String id = p.getString("id");

        switch (id){
            case "presenter":
                onPresenter(session, p);
                break;
            case "onIceCandidate":
                onIceCandidate(session, p);
                break;
        }
    }

    private void onIceCandidate(WebSocketSession session, JSONObject p) {
        String sessionId = session.getId();
        IceCandidate iceCandidate = JSON.parseObject(p.getString("candidate"), IceCandidate.class);

        if(president_session!=null&&sessionId.equals(president_session.getSession().getId())){
            president_session.getWebRtcEndpoint().addIceCandidate(iceCandidate);
            LOGGER.info("receive presenter candidate{}:",iceCandidate.getCandidate());
        }else {
            UserSession userSession = users.get(sessionId);
            if(userSession==null)return;
            userSession.getWebRtcEndpoint().addIceCandidate(iceCandidate);
        }




    }

    private void onPresenter(WebSocketSession session, JSONObject p) {
        String sdpOffer = p.getString("sdpOffer");
        JSONObject response = new JSONObject();
        response.put("id","presenterResponse");

        if(president_session==null){
            pipeline = kurentoClient.createMediaPipeline();
            WebRtcEndpoint webRtcEndpoint = new WebRtcEndpoint.Builder(pipeline).build();

            response.put("sdpAnswer", webRtcEndpoint.processOffer(sdpOffer));


            president_session=new UserSession(session,webRtcEndpoint);
            initWebRtc(president_session);

            //response
            response.put("response","accepted");

            sendMessage(session,response);
        }else if(president_session.getSession().getId().equals(session.getId())){
            response.put("response","rejected");
            response.put("message","你已经是主讲了");
            sendMessage(session,response);
        }else {
            response.put("response","rejected");
            response.put("message","本room已经有主讲了");
        }
    }

    private void initWebRtc(UserSession userSession){;
        WebRtcEndpoint webRtcEndpoint=userSession.getWebRtcEndpoint();
        WebSocketSession session=userSession.getSession();
        webRtcEndpoint.addIceCandidateFoundListener(new EventListener<IceCandidateFoundEvent>() {
            @Override
            public void onEvent(IceCandidateFoundEvent event) {
                JSONObject p = new JSONObject();
                p.put("id","iceCandidate");
                p.put("candidate",(JSONObject)JSON.toJSON(event.getCandidate()));
                sendMessage(session, p);
            }
        });

        webRtcEndpoint.gatherCandidates();
    }
    synchronized private void sendMessage(WebSocketSession session,JSONObject m){
        try {
            session.sendMessage(new TextMessage(m.toString()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private MediaPipeline pipeline;
    @Autowired
    private KurentoClient kurentoClient;
    private Map<String,UserSession> users=new ConcurrentHashMap<>();
    private UserSession president_session;


    public static final Logger LOGGER= LoggerFactory.getLogger(BroadcastHandler.class);
}
