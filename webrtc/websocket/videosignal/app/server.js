const express=require('express')
const app=express()
const server=require('http').createServer(app)
const port=3001


app.get('/uuid',(req,res)=>{
  res.redirect(`/video/uuid-${21232}`)
})
app.get('/uuid:room',(req,res)=>{
  console.log(req.params.room)
  res.send('uuid')
})


//new1:设置静态路径，这样html引用的css,js文件才能找到
console.log(`${__dirname}/public`)
const fs = require('fs')
console.log(fs.existsSync(`${__dirname}/public`))

app.use(express.static(`${__dirname}/public`))

app.get('/videochat',(req,res)=>{
  res.sendFile(__dirname + '/videochat.html');
})

server.listen(port, () => {
  console.log(`Socket.IO server running at http://localhost:${port}/`);
});




const sio=require('socket.io')(server)

let rooms={

}


let socket_room={

}

function showRoomInfo(roomname){
  var current=rooms[roomname]
  console.log(`room=${roomname},#=${current.length}`)
}

sio.on('connection',(socket)=>{

    console.log('connect:'+socket.id)
  
    socket.on('createOrJoin',(data)=>{

      /**
       * data里面必须有room字段，用于表示加入或在创建的房间
       * 
       * 全局字典里面如果有这个room
       * 1)是加入room
       *    A.检查本socket是否已经加入了room
       *    B.如果A不成立
       *      B.1）通知已经加入room的所有成员(event=join)
       *      B.2）通知这个新加入者((event=join,并且把他加入到room.list中)
       * 2)否则是创建room
       *   把socketid,data作为一个object，保存到当前room中(list)
       *  
      */
      var room=data['room']
      
      if(room in rooms){
        var peers=rooms[room]
        var isMember=false;
        for(let p of peers){
            if(p['socket'].id==socket.id){
              isMember=true;
              break;
            }
          }
            if(!isMember){
                  for(let p of peers){
                      p['socket'].emit('join',data)
                      socket.emit('joined',p['data'])
                    }
                  var d={'socket':socket,'data':data}
                  peers.push(d);
                  socket_room[socket.id]=room;
            }

      }else{
        var d={'socket':socket,'data':data}
        rooms[room]=[d];
        socket_room[socket.id]=room
      }
      
      showRoomInfo(room)
  })

  socket.on('message',(data)=>{
       /**
        * 把 消息转发给 【和个成员同room】的【其他
        * 成员】
        * 这里需要一个【反转表socket_room】
        * 根据socket.id就可以找出 【其他room成员】
        * 
       */
        var room=socket_room[socket.id]
        if(room in rooms){
          let peers=rooms[room]
          for(let p of peers){
              if(p['socket'].id==socket.id){
                continue;
              }
              p['socket'].emit('message',data)
          }

        }
  })
      socket.on('disconnect',()=>{
        /**
         * 当收到有人 断开socket的时候：
         * 
         * 1）通知与这个socket同房间的成员【event=bye】
         * 2) 从房间 【移除】这个成员(room表+反转表)
         * 3)如果【房间人数=0】，移除这个房间。
        */
      console.log('disconnected:'+socket.id)
      
      var roomName=socket_room[socket.id]
      delete socket_room[socket.id]

      var peers=rooms[roomName]
      
      if(!peers) return;

      let delIndex=-1;
      for(var i=0;i<peers.length;i++){
        if(peers[i]['socket'].id==socket.id){
          delIndex=i;
        }else{
          peers[i]['socket'].emit('bye')
        }
      }
      if(delIndex>=0)
        peers.splice(delIndex,1)
      
      showRoomInfo(roomName);

      if(peers.length==0){
        delete  rooms[roomName]
      }
  })


    
})