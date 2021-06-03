/*
 * Copyright 2018 Kurento (https://www.kurento.org)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const ws = new WebSocket("wss://" + location.host + "/helloworld");

let webRtcPeer;

// UI
let uiLocalVideo;
let uiRemoteVideo;
let uiState = null;
const UI_IDLE = 0;
const UI_STARTING = 1;
const UI_STARTED = 2;

window.onload = function () {
  console = new Console();
  console.log("Page loaded");
  uiLocalVideo = document.getElementById("uiLocalVideo");
  uiRemoteVideo = document.getElementById("uiRemoteVideo");
  uiSetState(UI_IDLE);
};

window.onbeforeunload = function () {
  console.log("Page unload - Close WebSocket");
  ws.close();
};

function sendError(message) {
  console.error(message);

  sendMessage({
    id: "ERROR",
    message: message,
  });
}

function sendMessage(message) {
  if (ws.readyState !== ws.OPEN) {
    console.warn("[sendMessage] Skip, WebSocket session isn't open");
    return;
  }

  const jsonMessage = JSON.stringify(message);
  console.log("[sendMessage] message: " + jsonMessage);
  ws.send(jsonMessage);
}

/* ============================= */
/* ==== WebSocket signaling ==== */
/* ============================= */

ws.onmessage = function (message) {
  const jsonMessage = JSON.parse(message.data);
  // console.log("[onmessage] Received message: " + message.data);

  switch (jsonMessage.id) {
    case "PROCESS_SDP_ANSWER":
      handleProcessSdpAnswer(jsonMessage);
      break;
    case "ADD_ICE_CANDIDATE":
      handleAddIceCandidate(jsonMessage);
      break;
    case "ERROR":
      handleError(jsonMessage);
      break;
    default:
      // Ignore the message
      console.warn("[onmessage] Invalid message, id: " + jsonMessage.id);
      break;
  }
};

// PROCESS_SDP_ANSWER ----------------------------------------------------------

function handleProcessSdpAnswer(jsonMessage) {
  console.log(
    "[handleProcessSdpAnswer] SDP Answer from Kurento, process in WebRTC Peer"
  );

  let sdpAnswer = jsonMessage["sdpAnswer"];
  webRtcPeer.setRemoteDescription({
    type:'answer',
    sdp:sdpAnswer
  });
  console.log("receive Answer");

  uiSetState(UI_STARTED);
}

// ADD_ICE_CANDIDATE -----------------------------------------------------------

function handleAddIceCandidate(jsonMessage) {
  let ice = new RTCIceCandidate(jsonMessage.candidate);
  webRtcPeer.addIceCandidate(ice);
  console.log("add candidate");
}

// STOP ------------------------------------------------------------------------

function stop() {
  if (uiState == UI_IDLE) {
    console.log("[stop] Skip, already stopped");
    return;
  }

  console.log("[stop]");

  if (webRtcPeer) {
    webRtcPeer.close();
    webRtcPeer = null;
  }

  uiSetState(UI_IDLE);
  hideSpinner(uiLocalVideo, uiRemoteVideo);

  sendMessage({
    id: "STOP",
  });
}

// ERROR -----------------------------------------------------------------------

function handleError(jsonMessage) {
  const errMessage = jsonMessage.message;
  console.error("Kurento error: " + errMessage);

  console.log("Assume that the other side stops after an error...");
  stop();
}

/* ==================== */
/* ==== UI actions ==== */
/* ==================== */

// Start -----------------------------------------------------------------------

async function openLocalStream() {
  var constraints = {
    audio: true,
    video: {
      width: 640,
      framerate: 15,
    },
  };
  var stream = await navigator.mediaDevices.getUserMedia(constraints);
  uiLocalVideo.srcObject = stream;
  startVideo(uiLocalVideo);
  return stream;
}
async function uiStart() {
  console.log("[start] Create WebRtcPeerSendrecv");

  let lcStream = await openLocalStream();

  webRtcPeer = new RTCPeerConnection();
  webRtcPeer.addStream(lcStream);
  lcStream.onicecandidate = (e) => {
    if (e.candidate) {
      sendMessage({
        id: "ADD_ICE_CANDIDATE",
        candidate: e.candidate.candidate,
        sdpMid: e.candidate.sdpMid,
        sdpMLineIndex: e.candidate.sdpMLineIndex,
      });
    }
  };

    webRtcPeer.onaddstream=(e)=>{
      let remoteStream=e.stream;
      uiRemoteVideo.srcObject=remoteStream;
      startVideo(uiRemoteVideo);
    }
    webRtcPeer.createOffer().then(e=>{
      webRtcPeer.setLocalDescription(e);
      sendMessage({
        id:'PROCESS_SDP_OFFER',
        sdpOffer:e.sdp
      })
      console.log("创建了offer")
    });

 
}

// Stop ------------------------------------------------------------------------

function uiStop() {
  stop();
}

// -----------------------------------------------------------------------------

/* ================== */
/* ==== UI state ==== */
/* ================== */

function uiSetState(newState) {
  switch (newState) {
    case UI_IDLE:
      uiEnableElement("#uiStartBtn", "uiStart()");
      uiDisableElement("#uiStopBtn");
      break;
    case UI_STARTING:
      uiDisableElement("#uiStartBtn");
      uiDisableElement("#uiStopBtn");
      break;
    case UI_STARTED:
      uiDisableElement("#uiStartBtn");
      uiEnableElement("#uiStopBtn", "uiStop()");
      break;
    default:
      console.warn("[setState] Skip, invalid state: " + newState);
      return;
  }
  uiState = newState;
}

function uiEnableElement(id, onclickHandler) {
  $(id).attr("disabled", false);
  if (onclickHandler) {
    $(id).attr("onclick", onclickHandler);
  }
}

function uiDisableElement(id) {
  $(id).attr("disabled", true);
  $(id).removeAttr("onclick");
}

function showSpinner() {
  for (let i = 0; i < arguments.length; i++) {
    arguments[i].poster = "./img/transparent-1px.png";
    arguments[i].style.background =
      "center transparent url('./img/spinner.gif') no-repeat";
  }
}

function hideSpinner() {
  for (let i = 0; i < arguments.length; i++) {
    arguments[i].src = "";
    arguments[i].poster = "./img/webrtc.png";
    arguments[i].style.background = "";
  }
}

function startVideo(video) {
  // Manually start the <video> HTML element
  // This is used instead of the 'autoplay' attribute, because iOS Safari
  // requires a direct user interaction in order to play a video with audio.
  // Ref: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/video
  video.play().catch((err) => {
    if (err.name === "NotAllowedError") {
      console.error("[start] Browser doesn't allow playing video: " + err);
    } else {
      console.error("[start] Error in video.play(): " + err);
    }
  });
}

/**
 * Lightbox utility (to display media pipeline image in a modal dialog)
 */
$(document).delegate('*[data-toggle="lightbox"]', "click", function (event) {
  event.preventDefault();
  $(this).ekkoLightbox();
});
