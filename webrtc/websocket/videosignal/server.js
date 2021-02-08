const app=require('express')()
const server=require('http').createServer(app)
const port=3001

app.get('/',(req,res)=>{
  res.sendFile(__dirname + '/chat.html');
})

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
      console.log('disconnected:'+socket.id)
      
      var roomName=socket_room[socket.id]
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
        peers.splice(delIndex)
      
      showRoomInfo(roomName);

      if(peers.length==0){
        delete  rooms[roomName]
      }
  })


    
})