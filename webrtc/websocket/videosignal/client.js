let socket;
let localStream;
let localRTC;
let remoteStream;
let isCreator=true;


        function signalCall(){
            if(!socket)
                socket=new io(location.origin);

                socket.on('join',data=>{
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
                    console.log(data)
                    yourstatus.className='ok'
    
                    callOthers()

            
                })

            socket.on('joined',data=>{
                /**
                 * 收到joined表示我已经 了room
                 * 换句话说，room的成员都与signal服务器取得
                 * 了联系。
                 * 作为 新加入者 的创建者
                 *  
                 * 1）建立peer2peer 的connection
                 * 2)  更新界面=>对方已经上线
                */
                console.log(data)
                initLocalPeer()

                isCreator=false;
                yourstatus.className='ok'
            })

            socket.on('message',data=>{
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
                var msgType=data['type']
                var sdp=data['sdp']

                if(msgType=='offer'){
                    localRTC.setRemoteDescription(sdp)
                    receivestatus.className='ok'
                    localRTC.addStream(localStream)
                    callCreator();
                    hangeupBth.disabled=false;
                }else if(msgType=='answer'){
                localRTC.setRemoteDescription(sdp)
                receivestatus.className='ok'
                hangeupBth.disabled=false;
        
                }else if(msgType=='candidate'){

                    localRTC.addIceCandidate(new RTCIceCandidate({
                                candidate: data.candidate,
                                sdpMLineIndex: data.label,
                                sdpMid: data.id
                     }));
                }

                
            })
           
            socket.on('bye',()=>{
                //收到有人主动退出
                
                console.log('getbye')
                someoneLeave()
            })

            //创建这主动发起事件
            socket.emit('createOrJoin',{
                'room':roomNum.value,
                'username':username.value
            })

            enterBtn.disabled=true
            hangeupBth.disabled=false
        }
    
        function hangeup(){
            /**
             * 如果 我主动发起 【挂断操作】
             * 执行如下：
             * 1.关闭socket，也就是和signal服务器断开连接。
             * 2。关闭peer2peer
             * 3.恢复默认界面
            */
            if(socket)
                socket.close();
            socket=null;

            releasePeer()

                            //ui 
            yourvideo.srcObject=null;
             yourstatus.className='cancle'
                notifyOthers.className='cancle'
                receivestatus.className='cancle'
                hangeupBth.disabled=true;
                enterBtn.disabled=false;
        }
        function someoneLeave(){
            /**得知有人退出的时候，我也
             * 准备退出
             * 1)释放peer2peer
             * 2)恢复默认界面
             * 
            */
            releasePeer();

            yourvideo.srcObject=null;
           
            yourstatus.className='cancle'
            notifyOthers.className='cancle'
            receivestatus.className='cancle'
        }


 


        function initLocalPeer(){
            /**
             * 创建peer2peer的connection
             * 
             * 1)new RTCPeerConnection
             * 2)都有ice candidate的时候，立即通过signal服务器，把ice candidate通知room的成员
             * 3)当有video |audio stream的时候，设置video tag,播放远程流
            */
            localRTC=new RTCPeerConnection()
            localRTC.onicecandidate=e=>{
                    console.log(e)
                    if(e.candidate){
                        socket.emit('message',{
                           type:'candidate',
                           sdp:'',
                           label: e.candidate.sdpMLineIndex,
                            id: e.candidate.sdpMid,
                            candidate: e.candidate.candidate
                        })
                    }
            }
            localRTC.onaddstream=event=>{
                remoteStream=event.stream;
                yourvideo.srcObject=remoteStream
            }
        }

        function releasePeer(){
            /**
             * 关闭peer2peer的connection，
             * 并且关闭远程流
             * localRTC，remoteStream都是null
            */
            if(localRTC){
                localRTC.close()
                localRTC=null
            }
            remoteStream=null;
        }
        function opencamera(){
            /**
             * 打开摄像头
            */
           myvideo.muted=true
            navigator.mediaDevices.getUserMedia({video:true,audio:true})
            .then(stream=>{
                myvideo.srcObject=stream;
                localStream=stream;
            })
        }

        function callOthers(){
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
            localRTC.addStream(localStream)

            localRTC.createOffer().then(o=>{
                return localRTC.setLocalDescription(o);
            }).then(()=>{
                socket.emit('message',{
                    'sdp':localRTC.localDescription,
                    'type':'offer'
                })
                notifyOthers.className='ok'
            })
        }


        function callCreator(){
            /**
             * 创建通知【发起人】的answer(sdp)，并且
             * 发送给【发起人】。
             * 
            */
            localRTC.createAnswer().then(a=>{
                localRTC.setLocalDescription(a);
                console.log(a)
                socket.emit('message',
                {
                    'sdp':a,
                    'type':'answer'
                }
                );
                notifyOthers.className='ok'
            })
        }

        opencamera()