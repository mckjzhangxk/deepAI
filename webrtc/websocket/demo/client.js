let ws=new WebSocket('ws://127.0.0.1:3001')
ws.onclose=(e=>{console.log('server closed')})
ws.onmessage=(m=>{
    console.log(`receive server message:${m.data}`)
})

ws.send('hello server')