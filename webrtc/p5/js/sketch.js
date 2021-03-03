

function drawPage(lines){
    background(255,255,255)
    let padding=windowHeight/lines

    let x1=30,x2=windowWidth-30,y=padding

    stroke(152)
    for(var i=0;i<lines;i++){
        line(x1,y,x2,y);
        y+=padding;
    }
}
function setup() {
    // put setup code here
   var W=windowWidth;
   var H=windowHeight
    var cv=createCanvas(W,H)
    // cv.position((windowWidth-W)/2,(windowHeight-H)/2)
    cv.style('display', 'block');
     
    drawPage(23)


  colorPicker = createColorPicker('#ed225d');
  colorPicker.position(19, 19);
  }
  
  function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
    drawPage(23)
  }
  let beginDraw=false;
  let lastPts=[0,0]
  function mousePressed() {
    beginDraw=true;
    lastPts=[mouseX,mouseY]
    return false
  }
  function mouseReleased(){
    beginDraw=false;
  }
  function mouseDragged() {
    console.log(beginDraw)
    if(beginDraw){
         
        stroke(255,0,0)
        strokeWeight(2)
        line(lastPts[0],lastPts[1],mouseX,mouseY)
        lastPts=[mouseX,mouseY]
    }
  }
  function draw() {
    // put drawing code here
    
}