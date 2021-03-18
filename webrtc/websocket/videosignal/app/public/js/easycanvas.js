var zhangxk = zhangxk || {};

zhangxk.commandHistory = [];

zhangxk.addDrawObject = function (o) {
  this.commandHistory.push(o);
};
zhangxk.draw = function () {
  for (var i = 0; i < this.commandHistory.length; i++) {
    this.commandHistory[i].draw();
  }
};
zhangxk.render = function () {
  if (arguments.length == 0)
    for (var i = 0; i < this.commandHistory.length; i++) {
      this.commandHistory[i].render();
    }
  else if (arguments.length == 1) {
    this.commandHistory[arguments[0]].render();
  }
};
zhangxk.clear = function () {
  this.commandHistory = [];
};
zhangxk.isEmpty = function () {
  return this.commandHistory.length == 0;
};
zhangxk.Page = function (lines, background, color, padding) {
  this.lines = lines;
  this.backgroundColor = background;
  this.color = color;
  this.padding = padding;

  this.context = null;
};

zhangxk.Page.prototype.draw = function () {
  this.context.background(this.backgroundColor);

  let padding = windowHeight / this.lines;
  let x1 = this.padding,
    x2 = windowWidth - this.padding,
    y = padding;

  this.context.stroke(this.color);
  this.context.strokeWeight(1);
  for (var i = 0; i < this.lines; i++) {
    this.context.line(x1, y, x2, y);
    y += padding;
  }
};
zhangxk.Page.prototype.render = function () {
  if (this.context) {
    image(this.context, 0, 0);
  }
};
zhangxk.Point = class {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }
};
zhangxk.Path = class {
  constructor(color, strokeWeight) {
    this.color = color;
    this.strokeWeight = strokeWeight;
    this.points = [];
    this.clearMode = 0;
    this.context = null;
  }
  add(x, y) {
    this.points.push(new zhangxk.Point(x, y));
  }
  draw() {
    if (!this.clearMode) {
      this.context.strokeWeight(this.strokeWeight);
      this.context.stroke(this.color);
      strokeWeight(this.strokeWeight);
      stroke(this.color);
      var x1 = this.points[0].x,
        y1 = this.points[0].y;
      for (var i = 1; i < this.points.length; i++) {
        line(x1, y1, this.points[i].x, this.points[i].y);
        this.context.line(x1, y1, this.points[i].x, this.points[i].y);
        (x1 = this.points[i].x), (y1 = this.points[i].y);
      }
    } else {
      this.context.erase();
      var x1 = this.points[0].x,
        y1 = this.points[0].y;
      for (var i = 1; i < this.points.length; i++) {
        this.context.ellipse(x1, y1, this.eraseWeight ? this.eraseWeight : 30);
        (x1 = this.points[i].x), (y1 = this.points[i].y);
      }
      this.context.noErase();
    }
  }
  render() {
    if (this.context) {
      image(this.context, 0, 0);
    }
  }
};


zhangxk.Image=class{
  constructor(img,x,y){
    this.img=img
    this.x=x
    this.y=y
    this.context = null;
  }

  draw() {
    let W=this.img.width;
    let  H=this.img.height*W/this.img.width
    this.context.image(this.img,this.x,this.y,W,H);
    console.log('draw')
  }
  render() {
    if (this.context) {
      image(this.context, 0, 0);
    }
  }
}
