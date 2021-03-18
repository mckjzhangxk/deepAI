function preload() {}
function setup() {
  // put setup code here

  if (zhangxk.isEmpty()) {
    var W = windowWidth;
    var H = windowHeight;
    var cv = createCanvas(W, H);
    cv.style("display", "block");

    //

    cv.drop((f) => {
      let img = createImg(f.data)
      img = new zhangxk.Image(img, mouseX, mouseY);
      img.context = pathContext;
      zhangxk.addDrawObject(img);

      img.draw();
      img.render();
    });

    //
    pathContext = createGraphics(W, H);

    var page = new zhangxk.Page(22, (255, 255, 255), 127, 30);
    page.context = createGraphics(W, H);

    zhangxk.commandHistory.push(page);
  }

  zhangxk.draw();
  zhangxk.render();
}

function windowResized() {
  // resizeCanvas(windowWidth, windowHeight);
  // setup();

}

function mousePressed(e) {
  if (e.button == 2) {
    var p1 = createElement(
      "div",
      '<div class="toolbar-item" role="button" tabindex="0" onclick="clear();zhangxk.clear();setup()">新的一页</div>'
    );
    p1.position(e.x - 50, e.y - p1.size().height);

    p1.mousePressed((eb) => {
      p1.remove(1);
    });
  }
  if (mode != MODE_MOVE) {
    currentPath = new zhangxk.Path(penColor, penWeight);
    currentPath.context = pathContext;
    currentPath.add(mouseX, mouseY);

    if (mode == MODE_CLEAR) {
      currentPath.clearMode = 1;
      currentPath.eraseWeight = eraseWeight;
    }
  }
}
function mouseReleased() {
  if (currentPath) zhangxk.addDrawObject(currentPath);
  currentPath = null;
}
function mouseDragged() {
  if (currentPath) {
    currentPath.add(mouseX, mouseY);

    if (mode == MODE_EDIT) {
      currentPath.draw();
    } else if (mode == MODE_CLEAR) {
      zhangxk.render(0);
      currentPath.draw();
      currentPath.render();
    }
  }

  if (mode != MODE_MOVE) {
    return false;
  }
}
function draw() {}

// console.log = (s) => {
//   logdiv.innerHTML = s;
// };
// function touchStarted(e) {
//   console.log('touchStarted'+JSON.stringify(e))
//   // prevent default
//   return true;
// }
// function touchEnded() {
//   console.log('touchEnd')
//   // prevent default
//   return true;
// }

document
  .getElementsByTagName("body")[0]
  .addEventListener("touchstart", function (evt) {
    // should be either "stylus" or "direct"

    console.log(evt.touches[0].touchType + `${evt.touches[0].force}`);
  });

const MODE_EDIT = 0;
const MODE_CLEAR = 1;
const MODE_MOVE = 2;

let currentPath = null;
let penColor = [255, 0, 0];
let penWeight = 2;
let eraseWeight = 20;

let mode = MODE_EDIT;

let pathContext = null;
