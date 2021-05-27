var ws = new WebSocket('wss://' + location.host + '/one2one');
var videoInput;
var videoOutput;
var webRtcPeer;
let options=null;

function sendMessage(m){
    ws.send(JSON.stringify(m));
}
window.onload = function() {
	console = new Console();
	//setRegisterState(NOT_REGISTERED);
	var drag = new Draggabilly(document.getElementById('videoSmall'));
	videoInput = document.getElementById('videoInput');
	videoOutput = document.getElementById('videoOutput');


	options={
		localVideo:videoInput,
		remoteVideo:videoOutput,
		onicecandidate:(candidate)=>{
			sendMessage({id:'candidate',candidate:candidate});
			// console.log(`receive candidate ${candidate.candidate}`)
		},
		mediaConstraints:{
			audio:true,
			video:{
				width:640,framerate:15
			}
		}
	};

}

window.onbeforeunload = function() {
	ws.close();
}

ws.onmessage = function(message) {
	var parsedMessage = JSON.parse(message.data);
	// console.info('Received message: ' + message.data);
	switch (parsedMessage.id) {
        case "onJoin":
			onJoin(parsedMessage);
			break;
		case 'candidate':
			onRecvCandidate(parsedMessage);
			break;
		break;
	default:
		console.error('Unrecognized message', parsedMessage);
	}
}

onJoin=function(result){
	if(result.result=="accept"){
		console.log(`你是${result.message}`);
		webRtcPeer.processAnswer(result.sdpAnswer,(err)=>{
			if(err){
				console.log(`远程sdp设置错误：${err}`);
			}else{
				console.log(`远程sdp设置完成`);
			}
		})
	}else{
		console.log(`被拒绝：${result.message}`);
	}
}
onRecvCandidate=function(c){
	webRtcPeer.addIceCandidate(c.candidate);
}
let vueapp=new Vue({
    el:"#app",
    methods:{
        callClick:function(){

			webRtcPeer=kurentoUtils.WebRtcPeer.WebRtcPeerSendrecv(options,()=>{
				webRtcPeer.generateOffer((err,sdpOffer)=>{
					if(err){
						console.log(`生成offer报错:${err}`);
					}else{
						sendMessage({id:'join',sdpOffer:sdpOffer})
					}
				})
			})
           
        },

        stopClick:function(){
            sendMessage({id:'stop'});
			console.log('退出')
        }
    }
})


/**
 * Lightbox utility (to display media pipeline image in a modal dialog)
 */
$(document).delegate('*[data-toggle="lightbox"]', 'click', function(event) {
	event.preventDefault();
	$(this).ekkoLightbox();
});



