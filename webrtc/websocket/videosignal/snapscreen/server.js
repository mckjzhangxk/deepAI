var express=require('express')
var http=require('http')
var app=express()
var server=http.Server(app)



app.get('/',(req,res)=>{
    res.sendFile(`${__dirname}/index.html`)
})
server.listen(3001,()=>{
    console.log('listen on 3001')
})

