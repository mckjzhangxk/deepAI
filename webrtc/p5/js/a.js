let pg1=null
let pg2=null
let mode=0
function setup() {
    // smooth();
    var H=400,W=400
    createCanvas(W, H);
  
     pg1=createGraphics(W,H)
     pg1. background(255,0,0);
     
     pg2=createGraphics(W,H)
     pg2.fill(0,255,0)
    pg2.noStroke()
  }
  
  function keyPressed(){
    mode=1
  }
  function keyReleased(){
    mode=0
  }
  function mouseDragged(){


  }
  function draw() {
    if(mode==0){
      pg2. circle(mouseX,mouseY,30)
    } else{
      pg2.erase(255,255);
      pg2. circle(mouseX,mouseY,50)
      pg2.noErase();
    }   
     
    image(pg1,0,0)
    image(pg2,0,0)
    // pg.erase(255,255);
    // fill(0,0,0)
    // circle(50,50,50)
     
 
    // image(pg,0,0)
    // image(pg,0,0)
    // pg.noErase();
    // image(pg,0,0)
  }