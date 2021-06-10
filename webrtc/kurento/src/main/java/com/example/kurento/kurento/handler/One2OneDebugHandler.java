package com.example.kurento.kurento.handler;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.kurento.client.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;

/**
 * Created by mathai on 21-5-27.
 */


public class One2OneDebugHandler extends TextWebSocketHandler {
    class UserSession {
        public UserSession(WebSocketSession session, WebRtcEndpoint webRtcEndpoint, String sdpOffer) {
            this.session = session;
            this.webRtcEndpoint = webRtcEndpoint;
            this.sdpOffer = sdpOffer;
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

        public String getSdpOffer() {
            return sdpOffer;
        }

        public void setSdpOffer(String sdpOffer) {
            this.sdpOffer = sdpOffer;
        }

        private WebRtcEndpoint webRtcEndpoint;
        private WebSocketSession session;
        private String sdpOffer;
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        JSONObject p = JSON.parseObject(message.getPayload());
        switch (p.getString("id")) {

            case "PROCESS_SDP_OFFER":
                onJoin(session, p);
                break;
            case "candidate":
                onCandidate(session, p);
                break;
            case "stop":
                onStop(session);
                break;
        }
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        logger.info("new Connection arrived to one2one {}",session.getId());
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        logger.info("Connection exit one2one {}",session.getId());
        onStop(session);
    }

    private void onStop(WebSocketSession session) {
        WebRtcEndpoint webRtcEndpoint = user1.getWebRtcEndpoint();
        webRtcEndpoint.release();
        user1 = null;

    }

    private void onCandidate(WebSocketSession session, JSONObject p) {

        IceCandidate iceCandidate = JSON.toJavaObject(p.getJSONObject("candidate"), IceCandidate.class);
        user1.getWebRtcEndpoint().addIceCandidate(iceCandidate);




    }

    private void onJoin(WebSocketSession session, JSONObject p) {
        JSONObject result = new JSONObject();
        result.put("id", "PROCESS_SDP_ANSWER");
        result.put("result", "accept");
        if (pipeline == null)
            pipeline = kurento.createMediaPipeline();
        if (user1 == null) {

            WebRtcEndpoint rtc1 = new WebRtcEndpoint.Builder(pipeline).build();
            user1 = new UserSession(session, rtc1, p.getString("sdpOffer"));

            String sdpAnswer = initWebRTC(user1);
//            result.put("message", "用户A");
//            result.put("sdpAnswer", sdpAnswer);
            rtc1.connect(rtc1);
//            tryConnect();
        }
//        sendMessage(session, result);
    }


    private String initWebRTC(UserSession u) {
        WebRtcEndpoint webRtcEndpoint = u.getWebRtcEndpoint();
        WebSocketSession session = u.getSession();

        //forget create answer
        String sdpAnswer = webRtcEndpoint.processOffer(u.getSdpOffer());

        JSONObject m=new JSONObject();
        m.put("id", "PROCESS_SDP_ANSWER");
        m.put("sdpAnswer",sdpAnswer);

        sendMessage(session,m);

        webRtcEndpoint.addIceCandidateFoundListener(new EventListener<IceCandidateFoundEvent>() {
            @Override
            public void onEvent(IceCandidateFoundEvent event) {
                JSONObject m = new JSONObject();
                m.put("id", "candidate");
                m.put("candidate", JSON.toJSON(event.getCandidate()));
                sendMessage(session, m);
            }
        });

        webRtcEndpoint.addMediaFlowInStateChangeListener(new EventListener<MediaFlowInStateChangeEvent>() {
            @Override
            public void onEvent(MediaFlowInStateChangeEvent mediaFlowInStateChangeEvent) {
                System.out.println("流入==============================");
                String SourceName = mediaFlowInStateChangeEvent.getSource().getName();
                String padName = mediaFlowInStateChangeEvent.getPadName();
                System.out.println(mediaFlowInStateChangeEvent.getMediaType());
                System.out.println("sourceName:" + SourceName);
                System.out.println("==============================");

            }
        });

        webRtcEndpoint.addMediaFlowOutStateChangeListener(new EventListener<MediaFlowOutStateChangeEvent>() {
            @Override
            public void onEvent(MediaFlowOutStateChangeEvent mediaFlowOutStateChangeEvent) {
                System.out.println("流出==============================");
                String SourceName = mediaFlowOutStateChangeEvent.getSource().getName();
                String padName = mediaFlowOutStateChangeEvent.getPadName();
                System.out.println(mediaFlowOutStateChangeEvent.getMediaType());
                System.out.println("sourceName:"+SourceName);
                System.out.println("==============================");
            }
        });
        //收集完成ice
        webRtcEndpoint.addIceGatheringDoneListener(new EventListener<IceGatheringDoneEvent>() {
            @Override
            public void onEvent(IceGatheringDoneEvent iceGatheringDoneEvent) {
                System.out.println("收集完成ICE");
            }
        });


        webRtcEndpoint.addNewCandidatePairSelectedListener(new EventListener<NewCandidatePairSelectedEvent>() {
            @Override
            public void onEvent(NewCandidatePairSelectedEvent newCandidatePairSelectedEvent) {
                String streamID = newCandidatePairSelectedEvent.getCandidatePair().getStreamID();
                String localCandidate = newCandidatePairSelectedEvent.getCandidatePair().getLocalCandidate();
                String remoteCandidate = newCandidatePairSelectedEvent.getCandidatePair().getRemoteCandidate();

                System.out.println("切换线路");
                System.out.println("streamId="+streamID);
                System.out.println("local="+localCandidate);
                System.out.println("remote="+remoteCandidate);

            }
        });
        webRtcEndpoint.gatherCandidates();


        return sdpAnswer;
    }

    synchronized void sendMessage(WebSocketSession session, JSONObject m) {
        try {
            session.sendMessage(new TextMessage(m.toJSONString()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private UserSession user1;



    private MediaPipeline pipeline;
    @Autowired
    private KurentoClient kurento;

    public static final Logger logger = LoggerFactory.getLogger(One2OneDebugHandler.class);
}
