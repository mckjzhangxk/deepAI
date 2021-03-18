let inp1, inp2;



function setup() {
  createCanvas(windowWidth, windowHeight - 200);
  // background('grey');
  let inp1 = createColorPicker('#ff0000');
  inp1.style('width','100%')
  inp1.style('margin','10px')
  
  select("#colorGroup").elt.append(inp1.elt)
  slider = createSlider(0, 255, 100);
  slider.style('margin','10px')
  select("#weightGroup").elt.append(slider.elt)

  let toolbar = select('.btn-toolbar-slider')
   
  let toolbarMain=select('.btn-toolbar')
  toolbarMain.position(100,100)
  toolbar.mousePressed(function (e) {
    toolbar.moved = true
    toolbar.clickX = mouseX
    toolbar.clickY = mouseY
  })


  function moveToolbar(e){
    if (toolbar.moved) {
      var ux = mouseX - toolbar.clickX
      var uy = mouseY - toolbar.clickY
      var ps = toolbarMain.position()
      toolbarMain.position(ps.x + ux, ps.y + uy)
      toolbar.clickX = mouseX
      toolbar.clickY = mouseY
      e.preventDefault()
    }
  }
  toolbar.mouseMoved(moveToolbar)
  toolbar.mouseOut(moveToolbar)
  toolbar.mouseReleased(function (e) {
    console.log('xxx')
    if (toolbar.moved) {
      toolbar.moved = false
      e.preventDefault()
    }
  })
  
  
  // slider.position(10, 10);
  // slider.style('width', '80px');
  slider.input((e) => {
    alert(e)
    e.preventDefault()
  })
  
  
}

function setMidShade() {
  // Finding a shade between the two
  let commonShade = lerpColor(inp1.color(), inp2.color(), 0.5);
  fill(commonShade);
  rect(20, 20, 60, 60);
}

function setShade1() {
  setMidShade();
  console.log('You are choosing shade 1 to be : ', this.value());
}
function setShade2() {
  setMidShade();
  console.log('You are choosing shade 2 to be : ', this.value());
}