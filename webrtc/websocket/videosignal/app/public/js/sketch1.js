let img;
let context=null
function setup() {
  let c = createCanvas(windowWidth, windowHeight);
  context=createGraphics(windowWidth, windowHeight)
  background(200);
  textAlign(CENTER);
  text('drop image', width / 2, height / 2);
  c.drop(gotFile);
}

function draw() {
  if (img) {
    W=300;
    H=img.height*W/img.width
    context.image(img,0,0,W,H);
    image(context,0,0)
  }
}

function gotFile(file) {
  img = createImg(file.data, '').hide();
}