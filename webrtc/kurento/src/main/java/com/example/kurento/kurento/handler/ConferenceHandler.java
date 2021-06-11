package com.example.kurento.kurento.handler;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;


public class ConferenceHandler extends TextWebSocketHandler {

    class UserSession {

        public UserSession(String name, WebSocketSession session,MediaPipeline pipeline) {
            this.session = session;
            this.name = name;
            this.sendPoint = new WebRtcEndpoint.Builder(pipeline).build();
            this.sendPoint.setName("sender_"+name);
            initWebRtcListener(name,sendPoint,session);
            this.pipeline=pipeline;
        }

        public String getId() {
            return session.getId();
        }

        public WebSocketSession getSession() {
            return session;
        }

        public void setSession(WebSocketSession session) {
            this.session = session;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public void connect( UserSession other) {
            WebRtcEndpoint recvEndPoint = new WebRtcEndpoint.Builder(pipeline).build();
            recvEndPoint.setName("recv_"+name+"_from"+"_"+other.name);
            recvPoints.put(other.name, recvEndPoint);

            other.sendPoint.connect(recvEndPoint);
            initWebRtcListener(other.name, recvEndPoint, session);


            WebRtcEndpoint o_recvEndPoint = new WebRtcEndpoint.Builder(pipeline).build();
            o_recvEndPoint.setName("recv_"+other.name+"_from_"+name);
            other.recvPoints.put(name, o_recvEndPoint);
            sendPoint.connect(o_recvEndPoint);

            initWebRtcListener(name, o_recvEndPoint, other.session);


        }

        public void disconnect(UserSession user) {
            WebRtcEndpoint webRtcEndpoint = recvPoints.remove(user.name);
            if (webRtcEndpoint == null) return;
            webRtcEndpoint.release();
        }



        private void dispose() {
            if (sendPoint != null) {
                sendPoint.release();
            }
            for (WebRtcEndpoint e : recvPoints.values()) {
                e.release();
            }
        }
        public WebRtcEndpoint getWebRtcByName(String name) {
            if(name.equals(this.name)){
                return sendPoint;
            }

            else {
                for(String w:recvPoints.keySet()){
                    if(w.equals(name))
                        return recvPoints.get(w);
                }
            }
            return null;
        }

        private WebSocketSession session;
        private String name;
        private WebRtcEndpoint sendPoint;
        private Map<String, WebRtcEndpoint> recvPoints = new ConcurrentHashMap<>();

        private MediaPipeline pipeline;
    }

    class Room {
        public Room(String name) {
            this.name = name;
            pipeline = kurentoClient.createMediaPipeline();
        }

        public void dispose() {
            if (pipeline != null)
                pipeline.release();
        }

        public String join(WebSocketSession session,String name,String sdpOffer) {
            String sessionId=session.getId();

            UserSession user = users.get(sessionId);
            if (user == null) return null;

            WebRtcEndpoint rtc = user.getWebRtcByName(name);
            if(rtc==null) return null;

            logger.info("============================webrct {}============================",rtc.getName());

            String sdpAnswer = rtc.processOffer(sdpOffer);
            rtc.gatherCandidates();

            return sdpAnswer;
        }

        public JSONArray requireJoin(String name, WebSocketSession session) {
            UserSession user = new UserSession(name, session,pipeline);

            JSONArray partpants = new JSONArray();
            for (UserSession u : users.values()) {
                user.connect(u);
                partpants.add(u.name);
            }

            users.put(user.getId(), user);


            return partpants;
        }

        public List<UserSession> leave(String id) {
            List<UserSession> participants=new ArrayList<>();

            UserSession user = users.remove(id);
            if (user == null) return participants;
            user.dispose();

            for (UserSession other : users.values()) {
                other.disconnect(user);
                participants.add(other);
            }
            return participants;
        }


        public WebSocketSession getSessionByName(String name) {
            for (UserSession u : users.values()) {
                if (u.getName().equals(name))
                    return u.getSession();
            }
            return null;
        }

        public String getNameBySessionId(String sessionId) {
            UserSession userSession = users.get(sessionId);
            return userSession==null?null:userSession.name;

        }

        public void addCandidate(String sessionId,String name,JSONObject candidate){
            IceCandidate ice=JSON.toJavaObject(candidate,IceCandidate.class);

            UserSession userSession = users.get(sessionId);
            WebRtcEndpoint webRtcEndpoint = userSession.getWebRtcByName(name);
            webRtcEndpoint.addIceCandidate(ice);


        }
        public boolean contains(String id) {
            return users.containsKey(id);
        }

        private Map<String, UserSession> users = new HashMap<>();
        private MediaPipeline pipeline;
        private String name;

    }


    class RoomManager {
        Room get(String name) {
            Room room = rooms.get(name);
            if (room == null){
                room = new Room(name);
                rooms.put(name,room);
            }
            return room;
        }

        public Room getRoomBySessionId(String id) {
            for (Room room : rooms.values()) {
                if(room.contains(id))
                    return room;
            }
            return null;
        }

        private Map<String, Room> rooms = new ConcurrentHashMap<>();

    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        JSONObject msgJson = JSONObject.parseObject(message.getPayload());
        String messageId = msgJson.getString("id");
        System.out.println(messageId);
        switch (messageId) {
            case "joinRoom":
                onRequestJoinRoom(session, msgJson);
                break;
            case "receiveVideoFrom":
                onReceiveVideo(session, msgJson);
                break;
            case "onIceCandidate":
                onCandidate(session,msgJson);
                break;
            case "leaveRoom":
                onLeaveRoom(session);
                break;
        }

    }



    private void onLeaveRoom(WebSocketSession session) {
        String sessionId = session.getId();

        Room room = manager.getRoomBySessionId(sessionId);
        if(room==null) return;

        String name=room.getNameBySessionId(sessionId);

        List<UserSession> participants = room.leave(sessionId);

        JSONObject leaveMessage=new JSONObject();
        leaveMessage.put("id","participantLeft");
        leaveMessage.put("name",name);

        for(UserSession u:participants){
            sendMessage(u.getSession(),leaveMessage);
        }
    }


    private void onRequestJoinRoom(WebSocketSession session, JSONObject msgJson) {
        String _name = msgJson.getString("name");
        String _room = msgJson.getString("room");

        Room room = manager.get(_room);

        JSONArray others_names = room.requireJoin(_name, session);

        JSONObject myMsg = new JSONObject();
        myMsg.put("id", "existingParticipants");
        myMsg.put("name", _name);
        myMsg.put("data", others_names);
        sendMessage(session, myMsg);


        JSONObject otherMsg = new JSONObject();
        otherMsg.put("id", "newParticipantArrived");
        otherMsg.put("name", _name);

        for (int i = 0; i < others_names.size(); i++) {
            String name = others_names.getString(i);
            WebSocketSession _session = room.getSessionByName(name);
            if (_session == null) continue;
            sendMessage(_session, otherMsg);
        }
    }

    private void onReceiveVideo(WebSocketSession session, JSONObject msgJson) {
        String name = msgJson.getString("sender");
        String sdpOffer = msgJson.getString("sdpOffer");

        Room room = manager.getRoomBySessionId(session.getId());

        String sdpAnswer =room.join(session,name, sdpOffer);
        JSONObject answerJson = new JSONObject();

        answerJson.put("id", "receiveVideoAnswer");
        answerJson.put("name", name);
        answerJson.put("sdpAnswer", sdpAnswer);
        sendMessage(session, answerJson);
    }
    private void onCandidate(WebSocketSession session, JSONObject msgJson) {
        Room room = manager.getRoomBySessionId(session.getId());
        room.addCandidate(session.getId(),msgJson.getString("name"),msgJson.getJSONObject("candidate"));
    }
    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        logger.info("new connection established with session id {}", session.getId());
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        logger.info("new connection exit with session id {}", session.getId());
        onLeaveRoom(session);
    }

    private void initWebRtcListener(String name, WebRtcEndpoint ep, WebSocketSession session) {

        ep.addIceCandidateFoundListener(new EventListener<IceCandidateFoundEvent>() {
            @Override
            public void onEvent(IceCandidateFoundEvent iceCandidateFoundEvent) {
                JSONObject message = new JSONObject();
                message.put("id", "iceCandidate");
                message.put("name", name);
                message.put("candidate", (JSONObject) JSON.toJSON(iceCandidateFoundEvent.getCandidate()));

                sendMessage(session, message);
            }
        });

    }

    synchronized private void sendMessage(WebSocketSession session, JSONObject p) {
        try {
            session.sendMessage(new TextMessage(p.toJSONString()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Logger logger = LoggerFactory.getLogger(ConferenceHandler.class);

    @Autowired
    private KurentoClient kurentoClient;
    private RoomManager manager = new RoomManager();
}
