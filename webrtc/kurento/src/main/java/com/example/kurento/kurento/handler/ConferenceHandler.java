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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;


public class ConferenceHandler extends TextWebSocketHandler {

    class UserSession {
        /**
         * 为用户,session创建实体对象
         * 同时创建webrtc
         * */
        public UserSession(String name, WebSocketSession session,MediaPipeline pipeline) {
            this.session = session;
            this.name = name;
            this.sendPoint = new WebRtcEndpoint.Builder(pipeline).build();
            this.sendPoint.setName("sender("+name+")");
            initWebRtcListener(name,sendPoint,session);
            this.pipeline=pipeline;
        }
        /**
         * 设置当ep收集到candidate的事件监听，
         * 他会使用session，发送消息，用name表示对应客户端的那个webrtc(recv,send?)
         *
         * */
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
        public String getId() {
            return session.getId();
        }

        public WebSocketSession getSession() {
            return session;
        }


        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        /**
         * 建立内部网络关系
         *
         * my.connect(other)等加入以下 伪代码
         * my.send.connect(other.recv[my.name])
         * other.send.connect(my.recv[other.name])
         *
         * recv:map(name,rtc),表示从那个name接受视频信息
         *
         * 对于一个recv，并名为recv(zxk) from qijia
         * */
        public void connect( UserSession other) {
            //别人发送给我
            WebRtcEndpoint recvEndPoint = new WebRtcEndpoint.Builder(pipeline).build();
            recvEndPoint.setName("recv("+name+") from"+"_"+other.name);
            recvPoints.put(other.name, recvEndPoint);

            other.sendPoint.connect(recvEndPoint);
            initWebRtcListener(other.name, recvEndPoint, session);

            //我发送给别人
            WebRtcEndpoint o_recvEndPoint = new WebRtcEndpoint.Builder(pipeline).build();
            o_recvEndPoint.setName("recv("+other.name+") from_"+name);
            other.recvPoints.put(name, o_recvEndPoint);
            sendPoint.connect(o_recvEndPoint);

            initWebRtcListener(name, o_recvEndPoint, other.session);


        }

        /**
         *
         * 释放 一个接收端recv[user.name]，
         * 从内部数据结构中移除
         * */
        public void disconnect(UserSession user) {
            WebRtcEndpoint webRtcEndpoint = recvPoints.remove(user.name);
            if (webRtcEndpoint == null) return;
            webRtcEndpoint.release();
        }


        /**
         *
         *  释放这个user的所有 webrtc
         *  1个send + (N-1)个recv
         * */
        private void dispose() {
            if (sendPoint != null) {
                sendPoint.release();
            }
            for (WebRtcEndpoint e : recvPoints.values()) {
                e.release();
            }
        }
        /**
         *
         * 根据name，选择返回sendRtc，或者某个recv[name].rtc,
         * 或者null
         * */
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
        //key=name
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
        /**
         *
         * 处理sdpOffer，处理的逻辑
         * session->userSession
         * userSession,name-->webrtc
         * rtc.processOffer
         *
         * 并且把消息返回给客户端
         * (id:receiveVideo,name,sdpAnswer)
         * */
        public void join(WebSocketSession session,String name,String sdpOffer) {
            String sessionId=session.getId();

            UserSession user = users.get(sessionId);
            if (user == null){
                logger.error("{} 对应的用户找不到",sessionId);
                return;
            }

            WebRtcEndpoint rtc = user.getWebRtcByName(name);
            if(rtc==null){
                logger.error("{}:{} 找不到对应的webrtc",sessionId,name);
                return;
            }

            logger.info("============================webrct {}============================",rtc.getName());

            //一定要再gatherCandidates之 把消息给客户端
            String sdpAnswer = rtc.processOffer(sdpOffer);
            JSONObject answerJson = new JSONObject();
            answerJson.put("id", "receiveVideoAnswer");
            answerJson.put("name", name);
            answerJson.put("sdpAnswer", sdpAnswer);
            sendMessage(session, answerJson);

            rtc.gatherCandidates();

        }
        /**
         *
         * 把新用户(name,session) 保存到内部数据结构中,使用sessionId唯一标识这个用户
         *
         * 备注：这里服务器端的webrtc已经建立好了，并且通信关系已经建立好了
         *
         * 返回room中其他的人的name【array】
         *
         * */
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

        /**
         *
         * 从内部数据结构移除 sessionId对应的数据。
         * 释放sessionId对应用户的所有webrtc(以及组内其他成员对应的接收端)
         *
         * 返回room内其他成员
         * */
        public List<UserSession> leave(String sessionId) {
            List<UserSession> participants=new ArrayList<>();

            UserSession user = users.remove(sessionId);
            if (user == null) return participants;
            user.dispose();

            for (UserSession other : users.values()) {
                other.disconnect(user);
                participants.add(other);
            }
            return participants;
        }

        /**
         * 返回room中[users结构体中] name对应的通信 session
         *
         * null 表示没找到
         *
         * */
        public WebSocketSession getSessionByName(String name) {
            for (UserSession u : users.values()) {
                if (u.getName().equals(name))
                    return u.getSession();
            }
            return null;
        }

        /**
         * 返回这个sessionId对应用户的name.null ->没有查到
         *
         * */
        public String getNameBySessionId(String sessionId) {
            UserSession userSession = users.get(sessionId);
            return userSession==null?null:userSession.name;

        }

        /**
         * (sessionId,name)唯一标识这一个webrtc,
         * 然后添加candidate
         *
         * */
        public void addCandidate(String sessionId,String name,JSONObject candidate){
            IceCandidate ice=JSON.toJavaObject(candidate,IceCandidate.class);

            UserSession userSession = users.get(sessionId);
            if (userSession == null){
                logger.error("addCandidate：{} 对应的用户找不到",sessionId);
                return;
            }

            WebRtcEndpoint webRtcEndpoint = userSession.getWebRtcByName(name);
            if(webRtcEndpoint==null){
                logger.error("addCandidate：{}:{} 找不到对应的webrtc",sessionId,name);
                return;
            }
            webRtcEndpoint.addIceCandidate(ice);


        }
        public boolean contains(String id) {
            return users.containsKey(id);
        }
        //key=sessionId
        private Map<String, UserSession> users = new ConcurrentHashMap<>();
        private MediaPipeline pipeline;
        private String name;

    }


    class RoomManager {
        /**
         * 根据name,返回一个room,这个room要不是已经存在的，
         * 要不是新创建的
         *
         * */
        Room get(String name) {
            Room room = rooms.get(name);
            if (room == null){
                room = new Room(name);
                rooms.put(name,room);
            }
            return room;
        }
        /**
         *
         * 一个room可以用name标识，也可以用sessionId进行查找。
         * manager的room是用name标识的。
         * 查找需要遍历每个room，问问你们那有没有那个user的session
         * 是本id
         *
         * 查找不到返回null
         * */
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


    /**
     * session对应的用户离开,销毁一切与之相关的记录,webrtc，
     * 并且发送给 room内其他成员，通知他们这个用户的离开
     * 消息内容
     * {
     *     id:participantLeft
     *     name:离开的用户名
     * }
     * */
    private void onLeaveRoom(WebSocketSession session) {
        String sessionId = session.getId();

        Room room = manager.getRoomBySessionId(sessionId);
        if(room==null) return;

        String name=room.getNameBySessionId(sessionId);
        if(name==null) return;
        List<UserSession> participants = room.leave(sessionId);

        JSONObject leaveMessage=new JSONObject();
        leaveMessage.put("id","participantLeft");
        leaveMessage.put("name",name);

        for(UserSession u:participants){
            sendMessage(u.getSession(),leaveMessage);
        }
    }

    /**
     * 当(session,name)申请加入某个room,为这个user创建好数据结构userSession后，
     * 加入到room内部，定义好room内部成员之间的通信网络
     * 然后返回如下信息：
     *
     *existParticipants(data) 给session
     *newParticipantArrived(name)给room的其他组员。
     *
     * */
    private void onRequestJoinRoom(WebSocketSession session, JSONObject msgJson) {
        String _name = msgJson.getString("name");
        String _room = msgJson.getString("room");

        Room room = manager.get(_room);

        JSONArray others_names = room.requireJoin(_name, session);

        JSONObject myMsg = new JSONObject();
        myMsg.put("id", "existingParticipants");
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

    /**
     * 接受来自客户端的offer，客户端通过sender标识是 “这是那个webrtc的offer”的，
     * 以便服务器处理对应的answer。
     *
     * 生成answer后，把结构以
     * (id:receiveVideo,name:sender,sdpAnswer:"xxxx")的消息发送给客户端。
     *
     *
     *
     * */
    private void onReceiveVideo(WebSocketSession session, JSONObject msgJson) {
        String name = msgJson.getString("sender");
        String sdpOffer = msgJson.getString("sdpOffer");

        Room room = manager.getRoomBySessionId(session.getId());

        if(room==null){
            logger.error("{} 没有room对应",session.getId());
            return;
        }
        room.join(session,name, sdpOffer);
    }


    /**
     * 处理客户端的candidate,通过session,msgJson.name可以找到对应的webrtc,进行处理
     *
     * */
    private void onCandidate(WebSocketSession session, JSONObject msgJson) {
        Room room = manager.getRoomBySessionId(session.getId());
        if(room==null){
            logger.error("{} 没有room对应",session.getId());
            return;
        }
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
