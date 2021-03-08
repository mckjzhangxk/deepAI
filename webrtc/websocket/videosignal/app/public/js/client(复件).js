let socket;
let localStream;
let localRTC;
let remoteStream;
let dataChannel = null;

let isCreator = true;

queryString = location.search;
const urlParams = new URLSearchParams(queryString);
let roomNum = urlParams.get("room") ? urlParams.get("room") : "room1";
let userid = urlParams.get("user") ? urlParams.get("user") : "test";

username.value = userid;

var pcConfig = {
  iceServers: [
    // {
    //     'urls':'stun:stun.mathai.xyz:3478'
    // }
    // {
    //   'urls': 'stun:stun.l.google.com:19302'
    // }
    // ,
    // {
    // 'urls':'stun:stun.services.mozilla.com'
    // }
    // ,
    {
      urls: "stun:stun.voipstunt.com",
    },
  ],
};
// var pcConfig = null

/**
 * @param {String} HTML representing a single element
 * @return {Element}
 */
function htmlToElement(html) {
  var template = document.createElement("template");
  html = html.trim(); // Never return a text node of whitespace as the result
  template.innerHTML = html;
  return template.content.firstChild;
}

let connection2Signal = false;
const onCallClick = () => {
  if (connection2Signal) {
    hangeup();
  } else {
    signalCall();
  }
  connection2Signal = !connection2Signal;
};

const openOrCloseClick = (flag) => {
  //打开关闭 视频和音频的 [事件回调函数]
  //flag video|audio
  var enabled =
    flag == "video"
      ? localStream.getVideoTracks()[0].enabled
      : localStream.getAudioTracks()[0].enabled;
  if (enabled) {
    setClose(flag);
  } else {
    setOpen(flag);
  }
};

const showOrHideChat = () => {
  var display = document.querySelector(".mainRight").style.display;
  if (display == "none") {
    document.querySelector(".mainRight").style.display = "";
    document.querySelector(".mainLeft").style.flex = "";
  } else {
    //hide chat
    document.querySelector(".mainRight").style.display = "none";
    document.querySelector(".mainLeft").style.flex = 1;
  }
};
function signalCall() {
  if (!socket) socket = new io(location.origin);

  socket.on("join", (data) => {
    /**
     * 收到join表示有新的成员加入了room
     * 换句话说，room的成员都与signal服务器取得
     *
     * 作为room的创建者
     *
     * 1）建立peer2peer 的connection
     * 2)  更新界面=>对方已经上线
     *
     * 上述过程 与新加入者是一样的
     *  因为 只有 通信 【双方都在线时】，才有必要建立 peer2peer的connection
     *    建立peer2peer 马上会有ice candidate产生，第一时间通知到 另一端才是
     * 正确的做法，如果【另一端不再线】，这个通知就要  【延后】，所以简单的设计
     * 只有【确保双方都在线时】，再建立【peer2peer connection.
     */
    console.log(data);
    yourstatus.className = "ok";

    callOthers();
  });

  socket.on("joined", (data) => {
    /**
     * 收到joined表示我已经 了room
     * 换句话说，room的成员都与signal服务器取得
     * 了联系。
     * 作为 新加入者 的创建者
     *
     * 1）建立peer2peer 的connection
     * 2)  更新界面=>对方已经上线
     */
    console.log(data);
    isCreator = false;
    initLocalPeer();
    yourstatus.className = "ok";
  });

  socket.on("message", (data) => {
    /**
     * 根据data的类型
     *
     * 0)candidate
     *      调用localRTC.addIceCandidate，注意 添加的是对方的【ice】
     * 1)offer
     *      【对方】收到发起人的offer
     *    ——创建answer应答发起人
     *    ——更新界面=>收到对方的通知，可以挂断
     *
     * 2)answer
     *   发起人收到【对方的】answer，表示peer2peer 完成通信。
     *   更新界面=>收到对方的通知，可以挂断
     */
    var msgType = data["type"];
    var sdp = data["sdp"];

    if (msgType == "offer") {
      localRTC.setRemoteDescription(sdp);
      receivestatus.className = "ok";
      localRTC.addStream(localStream);
      callCreator();
    } else if (msgType == "answer") {
      localRTC.setRemoteDescription(sdp);
      receivestatus.className = "ok";
    } else if (msgType == "candidate") {
      localRTC.addIceCandidate(
        new RTCIceCandidate({
          candidate: data.candidate,
          sdpMLineIndex: data.label,
          sdpMid: data.id,
        })
      );
    }
  });

  socket.on("bye", () => {
    //收到有人主动退出

    console.log("getbye");
    someoneLeave();
  });

  //创建这主动发起事件
  socket.emit("createOrJoin", {
    room: roomNum,
    username: userid,
  });

  setHangupButton();
}

function hangeup() {
  /**
   * 如果 我主动发起 【挂断操作】
   * 执行如下：
   * 1.关闭socket，也就是和signal服务器断开连接。
   * 2。关闭peer2peer
   * 3.恢复默认界面
   */
  if (socket) socket.close();
  socket = null;

  releasePeer();

  //ui
  yourvideo.srcObject = null;
  yourstatus.className = "cancle";
  notifyOthers.className = "cancle";
  receivestatus.className = "cancle";

  setCallButton();
}
function someoneLeave() {
  /**得知有人退出的时候，我也
   * 准备退出
   * 1)释放peer2peer
   * 2)恢复默认界面
   * 3)bug isCreator设置为true
   *
   */
  releasePeer();

  yourvideo.srcObject = null;

  yourstatus.className = "cancle";
  notifyOthers.className = "cancle";
  receivestatus.className = "cancle";

  isCreator = true;
}

function setDataChannelEvent() {
  dataChannel.onopen = (e) => console.log("chatChannel open");
  dataChannel.onmessage = (e) => {
    var em = htmlToElement(`<li class="messageItem">${e.data}</li>`);
    var ul = document.querySelector(".chatContent ul");
    ul.append(em);

    //scroll to bottom
    /* chatContent的css设置overflow-y:scroll
                  表示当 children的高度超过了 我，我添加滚动条 */
    /* 只有设置这个overflow-y:scroll 的div, scrollHeight和client Height不同 */
    var ct = document.querySelector(".chatContent");
    ct.scrollTop = ct.scrollHeight - ct.clientHeight;
  };
}
function initLocalPeer() {
  /**
   * 创建peer2peer的connection
   *
   * 1)new RTCPeerConnection
   * 2)当有ice candidate的时候，立即通过signal服务器，把ice candidate通知room的成员
   * 3)当有video |audio stream的时候，设置video tag,播放远程流
   *
   * 4)补充：对于room的【创建者】，他创建一个datachannel
   *              对于room的【其他】  他监听ondatachannel，获得【创建者】的datachannel
   */

  localRTC = new RTCPeerConnection(pcConfig);
  if (isCreator) {
    const dataChannelOptions = {
      id: 1000,
      negotiated: true,
      ordered: false, // do not guarantee order
      maxPacketLifeTime: 3000, // in milliseconds
    };
    dataChannel = localRTC.createDataChannel("chatChannel");

    setDataChannelEvent();
  } else {
    localRTC.ondatachannel = (e) => {
      dataChannel = e.channel;
      setDataChannelEvent();
    };
  }

  localRTC.onicecandidate = (e) => {
    // console.log(e)
    if (e.candidate) {
      socket.emit("message", {
        type: "candidate",
        sdp: "",
        label: e.candidate.sdpMLineIndex,
        id: e.candidate.sdpMid,
        candidate: e.candidate.candidate,
      });
    }
  };
  localRTC.onaddstream = (event) => {
    remoteStream = event.stream;
    yourvideo.srcObject = remoteStream;
  };
}

function releasePeer() {
  /**
   * 关闭peer2peer的connection，
   * 并且关闭远程流
   * localRTC，remoteStream都是null
   */
  if (dataChannel) {
    dataChannel.close();
    dataChannel = null;
  }
  if (localRTC) {
    localRTC.close();
    localRTC = null;
  }
  remoteStream = null;
}

async function getCustomStream() {
  var cameraStream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      sampleRate: 44100,
    },
  });

  await document.querySelector("#filevideo").play();
  var movieStream = await filevideo.captureStream(0);

  return [
    {
      name: "摄像头",
      stream: cameraStream,
    },
    {
      name: "电影",
      stream: movieStream,
    },
  ];
}

function opencamera() {
  /**
   * 打开摄像头
   */
  //new3 静音
  myvideo.muted = true;

  navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 44100,
      },
    })
    .then((stream) => {
      myvideo.srcObject = stream;
      localStream = stream;

      //new2: 2中加入event 监听的写法
      myvideo.addEventListener("loadedmetadata", () => {
        myvideo.play();
      });

      // myvideo.onloadedmetadata=()=>{
      //     myvideo.play()
      // }
    });
}

function callOthers() {
  /**
   * room的发起人 得知【对方上线】后的动作
   *
   * 1)建立【peer2peer】，设置好 connection的响应事件。
   * 2) 添加【本地流】到connection
   * 3)创建offer(SDP),完成后把sdp发送给对方。
   *
   * 这里要注意
   * 1）创建好的localSDP 要通知【对方】
   * 2)  每次的 new candidate  要通知【对方】
   */

  initLocalPeer();
  localRTC.addStream(localStream);

  localRTC.createOffer().then((o) => {
    localRTC.setLocalDescription(o);
    socket.emit("message", {
      sdp: o,
      type: "offer",
    });
    notifyOthers.className = "ok";
  });
}

function callCreator() {
  /**
   * 创建通知【发起人】的answer(sdp)，并且
   * 发送给【发起人】。
   *
   */
  localRTC.createAnswer().then((a) => {
    localRTC.setLocalDescription(a);
    console.log(a);
    socket.emit("message", {
      sdp: a,
      type: "answer",
    });
    notifyOthers.className = "ok";
  });
}

function setHangupButton() {
  document.querySelector(".callOrHangup").innerHTML =
    '<i class="fas fa-phone-slash stop"></i>\
                    <span  class="stop">挂断</span>';
}
function setCallButton() {
  document.querySelector(".callOrHangup").innerHTML =
    ' <i class="fa fa-phone" aria-hidden="true"></i> \
                    <span>连线</span>';
}

function setClose(flag) {
  /**
   * 关闭本地视频,更新界面显示[开启视频]
   *
   * 关闭本地音频,,更新界面显示[语音]
   *
   */
  if (flag == "video") {
    localStream.getVideoTracks()[0].enabled = false;
    document.querySelector(".openOrCloseVideo").innerHTML =
      '<i class="fas fa-video"></i>\
                <span>开启视频</span>';
  } else {
    localStream.getAudioTracks()[0].enabled = false;
    document.querySelector(".openOrCloseAudio").innerHTML =
      '<i class="fa fa-microphone" aria-hidden="true"></i>\
            <span>语音</span>';
  }
}

function setOpen(flag) {
  /**
   * 开启本地视频,更新界面显示[关闭视频]
   *
   * 开启本地音频,,更新界面显示[静音]
   *
   */

  if (flag == "video") {
    localStream.getVideoTracks()[0].enabled = true;
    document.querySelector(".openOrCloseVideo").innerHTML =
      '<i class="fas fa-video-slash stop"></i>\
                <span class="stop">关闭视频</span>';
  } else {
    localStream.getAudioTracks()[0].enabled = true;
    document.querySelector(".openOrCloseAudio").innerHTML =
      ' <i class="fas fa-microphone-slash stop"></i>\
                <span class="stop">静音</span>';
  }
}

function loadLocalStream(s) {
  localStream = s;

  myvideo.muted = true;
  myvideo.srcObject = localStream;
  myvideo.addEventListener("loadedmetadata", () => {
    myvideo.play();
  });
}
// opencamera()
let videoRources = null;
(async () => {
  videoRources = await getCustomStream();
  var options = "";
  for (var source of videoRources) {
    options += ` <option value="${source.name}">${source.name}</option>`;
  }
  document.querySelector("#videoSourceSelect").innerHTML = options;
  document
    .querySelector("#videoSourceSelect")
    .addEventListener("change", (e) => {
      var selectSourceName = document.querySelector("#videoSourceSelect").value;
      for (var s of videoRources) {
        if (s.name == selectSourceName) {
          loadLocalStream(s.stream);
          if (localRTC != null) {
            hangeup();
            signalCall();
          }
          break;
        }
      }
    });

  loadLocalStream(videoRources[0].stream);

  //聊天
  messageInput.addEventListener("keydown", (e) => {
    if (e.keyCode == 13 && messageInput.value.trim() != "") {
      var msg = messageInput.value.trim();
      messageInput.value = "";
      console.log(msg);

      if (dataChannel) {
        dataChannel.send(msg);
      }
    }
  });
})();
