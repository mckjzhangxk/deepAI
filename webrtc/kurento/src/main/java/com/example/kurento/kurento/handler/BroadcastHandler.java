package com.example.kurento.kurento.handler;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.kurento.client.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.socket.CloseStatus;
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
            case "viewer":
                onViewer(session, p);
                break;
            case "stop":
                stop(session);
        }
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {

        stop(session);
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
            LOGGER.info("receive viewer candidate{}:",iceCandidate.getCandidate());
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
        }else if(president_session.getSession().getId().equals(session.getId())){
            response.put("response","rejected");
            response.put("message","你已经是主讲了");

        }else {
            response.put("response","rejected");
            response.put("message","本room已经有主讲了");
        }
        sendMessage(session,response);
    }



    private void onViewer(WebSocketSession session, JSONObject p) {
        String sdpOffer = p.getString("sdpOffer");
        JSONObject response = new JSONObject();
        response.put("id","viewerResponse");


        if(pipeline!=null){

            WebRtcEndpoint webRtcEndpoint=new WebRtcEndpoint.Builder(pipeline).build();
            response.put("sdpAnswer",webRtcEndpoint.processOffer(sdpOffer));

            UserSession currentUser = new UserSession(session, webRtcEndpoint);

            users.put(session.getId(),currentUser);

            initWebRtc(currentUser);

            president_session.getWebRtcEndpoint().connect(webRtcEndpoint);

            response.put("response","accepted");


        }else {
            response.put("response","rejected");
            response.put("message","room的主讲没有上线");
        }
        sendMessage(session,response);

    }

    private void stop(WebSocketSession session) {
        if(president_session!=null&&session.getId().equals(president_session.getSession().getId())){
            president_session.getWebRtcEndpoint().release();
            pipeline.release();
            //通知其他人

            for(UserSession ss:users.values()){
                JSONObject jsonObject = new JSONObject();
                jsonObject.put("id","stopCommunication");
                sendMessage(ss.getSession(), jsonObject);
            }
            president_session=null;
            pipeline=null;
            users.clear();
        }else {
            UserSession viewSession = users.get(session.getId());
            if(viewSession!=null){
                viewSession.getWebRtcEndpoint().release();
                users.remove(session.getId());
            }
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
