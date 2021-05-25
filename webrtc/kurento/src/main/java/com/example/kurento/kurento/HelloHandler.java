package com.example.kurento.kurento;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.google.gson.JsonObject;
import org.kurento.client.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import javax.websocket.Session;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by mathai on 21-5-25.
 */
public class HelloHandler extends TextWebSocketHandler {
    @Autowired
    private MyService service;
    @Autowired
    private  KurentoClient kurento;
    private Map<String,WebRtcEndpoint> dict=new HashMap<>();

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {


        JSONObject p= JSON.parseObject(message.getPayload());
        String messageId = p.getString("id");



        switch (messageId){
            case "PROCESS_SDP_OFFER":
                MediaPipeline mediaPipeline = kurento.createMediaPipeline();
                //创建webrtc element
                WebRtcEndpoint webRtcEndpoint = new WebRtcEndpoint.Builder(mediaPipeline).build();
                dict.put(session.getId(),webRtcEndpoint);


                //创建answer
                String sdpAnswer =webRtcEndpoint.processOffer(p.getString("sdpOffer"));
                JSONObject answermessage = new JSONObject();
                answermessage.put("id", "PROCESS_SDP_ANSWER");
                answermessage.put("sdpAnswer", sdpAnswer);
                //通知客户端
                sendMessage(session, answermessage.toJSONString());


                //source->sink
                webRtcEndpoint.connect(webRtcEndpoint);


                initWebRtcBasetListeners(session,webRtcEndpoint);
                initWebRtcEventListeners(session,webRtcEndpoint);
                webRtcEndpoint.gatherCandidates();
                break;
            case "ADD_ICE_CANDIDATE":

                WebRtcEndpoint webRtcEndpoint1 = dict.get(session.getId());
                JSONObject candidate = p.getJSONObject("candidate");

                IceCandidate iceCandidate=new IceCandidate(candidate.getString("candidate"),candidate.getString("sdpMid"),candidate.getInteger("sdpMLineIndex"));
                webRtcEndpoint1.addIceCandidate(iceCandidate);
                break;
            case "STOP":
                stop(session);
                break;
        }
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        super.afterConnectionEstablished(session);
        System.out.println("new Connection:"+session.getId());
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        super.afterConnectionClosed(session, status);
        stop(session);
        System.out.println("Connection exit:"+session.getId());
    }



    private  void initWebRtcEventListeners(WebSocketSession session,WebRtcEndpoint rtcEndpoint){
        //收集 发现一个candidate
        rtcEndpoint.addIceCandidateFoundListener(new EventListener<IceCandidateFoundEvent>() {
            @Override
            public void onEvent(IceCandidateFoundEvent event) {
                String candidate = event.getCandidate().getCandidate();
                String sdpMid = event.getCandidate().getSdpMid();
                int sdpMLineIndex = event.getCandidate().getSdpMLineIndex();

                JSONObject p=new JSONObject();
                p.put("candidate",candidate);
                p.put("sdpMid",sdpMid);
                p.put("sdpMLineIndex",sdpMLineIndex);

                JSONObject message = new JSONObject();
                message.put("id", "ADD_ICE_CANDIDATE");
                message.put("candidate", p);


                sendMessage(session,message.toJSONString());
            }
        });

        //收集完成ice
        rtcEndpoint.addIceGatheringDoneListener(new EventListener<IceGatheringDoneEvent>() {
            @Override
            public void onEvent(IceGatheringDoneEvent iceGatheringDoneEvent) {
                System.out.println("收集完成ICE");
            }
        });


        rtcEndpoint.addNewCandidatePairSelectedListener(new EventListener<NewCandidatePairSelectedEvent>() {
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
    }


    private void  initWebRtcBasetListeners(WebSocketSession session,WebRtcEndpoint rtcEndpoint){
        rtcEndpoint.addMediaFlowInStateChangeListener(new EventListener<MediaFlowInStateChangeEvent>() {
            @Override
            public void onEvent(MediaFlowInStateChangeEvent mediaFlowInStateChangeEvent) {
                String SourceName = mediaFlowInStateChangeEvent.getSource().getName();
                String padName = mediaFlowInStateChangeEvent.getPadName();
                System.out.println(mediaFlowInStateChangeEvent.getMediaType());
                System.out.println("sourceName:"+SourceName);

                System.out.println("padName:"+padName);
            }
        });

        rtcEndpoint.addMediaFlowOutStateChangeListener(new EventListener<MediaFlowOutStateChangeEvent>() {
            @Override
            public void onEvent(MediaFlowOutStateChangeEvent mediaFlowOutStateChangeEvent) {

            }
        });
    }
    synchronized private void sendMessage(WebSocketSession session,String msg){
        if(session.isOpen()==false) return;

        try {
            session.sendMessage(new TextMessage(msg.toString()));
        } catch (IOException e) {
            System.err.println("[Handler::sendMessage] Exception: " + e.getMessage());
        }
    }

    private void stop(WebSocketSession session){
        String sessionId = session.getId();
        if(dict.containsKey(sessionId)){
            dict.get(sessionId).release();
            dict.remove(sessionId);
        }

    }
}
