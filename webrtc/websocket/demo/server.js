let LOG=console.log
const http=require('http')
const express=require('express')
const app=express()
const WebSocketServer=require('websocket').server;

const  httpServer=http.createServer(app)

app.get('/',(req,res)=>{
    res.sendFile(__dirname + '/index.html');
  })


const ws=new WebSocketServer({
    'httpServer':httpServer
})
const port=3001;
httpServer.listen(port,()=>{
    LOG(`http server listen on ${port}`)
})

let client=null;
ws.on('request',request=>{
    client=request.accept(null,request.orgin)
    
    client.on('open',e=>LOG('client open'));
    client.on('close',e=>LOG('client closed'))
    client.on('message',message=>{
        LOG(`get Message ${message.utf8Data}`)
    })
    keepAlive()
})


function keepAlive(){
    client.send('alive?')
    setTimeout(keepAlive,1000)
}