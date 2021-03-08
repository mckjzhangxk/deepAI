var zhangxk = zhangxk || {};

async function openCameraStream() {
  var ss = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      sampleRate: 44100,
    },
  });
  return {
    video: ss.getVideoTracks(),
    audio: ss.getAudioTracks(),
  };
}
zhangxk.createStream = function (videoTrack, audioTrack) {
  let s = new MediaStream();
  if (videoTrack) s.addTrack(videoTrack);
  if (audioTrack) s.addTrack(audioTrack);

  return s;
};
zhangxk.TRACK_CAMERA = "camera";
zhangxk.TRACK_MIC = "mic";
zhangxk.TRACK_MEDIA = "videlfile";
zhangxk.TRACK_SCREEN = "screen";

zhangxk.initLocalStream = async function () {
  zhangxk.localTracks = await openCameraStream();
};

zhangxk.Media = class {
  constructor(name, type, source) {
    this.type = type;
    this.name = name;
    this.source = source;
  }
  async getTrack() {
    if (this.track) return this.track;

    if (!zhangxk.localTracks) await zhangxk.initLocalStream();

    if (this.source == zhangxk.TRACK_CAMERA) {
      this.track =
        zhangxk.localTracks["video"].length > 0
          ? zhangxk.localTracks["video"][0]
          : null;
    }
    if (this.source == zhangxk.TRACK_MIC) {
      this.track =
        zhangxk.localTracks["audio"].length > 0
          ? zhangxk.localTracks["audio"][0]
          : null;
    }

    if (this.source == zhangxk.TRACK_MEDIA) {
      var videoHtml =
        '  <video playsinline autoplay controls loop muted style="display: none;">\
                <source src="video/chrome.webm" type="video/webm"/>\
        </video>';

      var template = document.createElement("template");
      videoHtml = videoHtml.trim(); // Never return a text node of whitespace as the result
      template.innerHTML = videoHtml;
      var videoobj = template.content.firstChild;

      document.querySelector("body").append(videoobj);

      await videoobj.play();
      var movieStream = await videoobj.captureStream(0);
      this.track = movieStream.getVideoTracks()[0];
    }

    if (this.source ==zhangxk.TRACK_SCREEN) {
      let options = {
        video: { cursor: "always" },
      };
      let screenStream = await navigator.mediaDevices.getDisplayMedia(options);
      this.track = screenStream.getVideoTracks()[0];
    }

    return this.track;
  }
  setTrack(track) {
    this.track = track;
  }
};

zhangxk.getAllMediaTrack = function () {
  return {
    video: [
      new zhangxk.Media("摄像头", "video", zhangxk.TRACK_CAMERA),
      new zhangxk.Media("动画片", "video", zhangxk.TRACK_MEDIA),
      new zhangxk.Media("投屏", "video", zhangxk.TRACK_SCREEN),
    ],
    audio: [new zhangxk.Media("麦克风", "audio", zhangxk.TRACK_MIC)],
  };
};
