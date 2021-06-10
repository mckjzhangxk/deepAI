package com.example.kurento.kurento.handler;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.example.kurento.kurento.service.MyService;
import org.kurento.client.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by mathai on 21-5-25.
 */
public class HelloHandler extends TextWebSocketHandler {
    @Autowired
    private MyService service;
    @Autowired
    private  KurentoClient kurento;
    private Map<String,WebRtcEndpoint> dict=new ConcurrentHashMap<>();
    private MediaPipeline pipeline;
    private void onOffer(WebSocketSession session,JSONObject m){
        if(pipeline==null)
            pipeline=kurento.createMediaPipeline();

        //创建webrtc element
        WebRtcEndpoint webRtcEndpoint = new WebRtcEndpoint.Builder(pipeline).build();
        webRtcEndpoint.setName("张小楷");

        dict.put(session.getId(),webRtcEndpoint);


        //创建answer
        String sdpAnswer =webRtcEndpoint.processOffer(m.getString("sdpOffer"));

        //回复answer
        JSONObject answermessage = new JSONObject();
        answermessage.put("id", "PROCESS_SDP_ANSWER");
        answermessage.put("sdpAnswer", sdpAnswer);
        sendMessage(session, answermessage.toJSONString());


//        FaceOverlayFilter faceOverlayFilter = new FaceOverlayFilter.Builder(mediaPipeline).build();
//        faceOverlayFilter.setOverlayedImage("https://192.168.1.36:8443/helloworld/img/mario-wings.png", -0.35F, -1.2F, 1.6F, 1.6F);

//        webRtcEndpoint.connect(faceOverlayFilter);
//        faceOverlayFilter.connect(webRtcEndpoint);

        //source->sink
        webRtcEndpoint.connect(webRtcEndpoint);


        initWebRtcBasetListeners(session,webRtcEndpoint);
        initWebRtcEventListeners(session,webRtcEndpoint);
        webRtcEndpoint.gatherCandidates();

    }
    private void  onRemoteCandidate(WebSocketSession session,JSONObject m){

        WebRtcEndpoint webRtcEndpoint = dict.get(session.getId());
        JSONObject candidate = m.getJSONObject("candidate");

        IceCandidate iceCandidate=new IceCandidate(candidate.getString("candidate"),candidate.getString("sdpMid"),candidate.getInteger("sdpMLineIndex"));
        webRtcEndpoint.addIceCandidate(iceCandidate);
    }
    private void stop(WebSocketSession session){
        String sessionId = session.getId();
        if(dict.containsKey(sessionId)){
            dict.get(sessionId).release();
            dict.remove(sessionId);
        }

    }
    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {

        JSONObject p= JSON.parseObject(message.getPayload());
        String messageId = p.getString("id");

        switch (messageId){
            case "PROCESS_SDP_OFFER":
                onOffer(session,p);
                break;
            case "candidate":
                onRemoteCandidate(session,p);
                break;
            case "stop":
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
                message.put("id", "candidate");
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
                System.out.println("流入==============================");
                String SourceName = mediaFlowInStateChangeEvent.getSource().getName();
                String padName = mediaFlowInStateChangeEvent.getPadName();
                System.out.println(mediaFlowInStateChangeEvent.getMediaType());
                System.out.println("sourceName:" + SourceName);
                System.out.println("==============================");

            }
        });

        rtcEndpoint.addMediaFlowOutStateChangeListener(new EventListener<MediaFlowOutStateChangeEvent>() {
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
    }
    synchronized private void sendMessage(WebSocketSession session,String msg){
        if(session.isOpen()==false) return;

        try {
            session.sendMessage(new TextMessage(msg.toString()));
        } catch (IOException e) {
            System.err.println("[Handler::sendMessage] Exception: " + e.getMessage());
        }
    }


}
