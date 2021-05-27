function setup() {
    createCanvas(500, 500);
    background(100);
    rectMode(CENTER)
}
let x=0
function draw() {
    background(100);

    push()
    translate(width / 2, height / 2);
    var u = p5.Vector.fromAngle(millis() / 1000, 40);
    translate(u);
    fill(255)
    rect(0, 0, 20, 20);
    pop();


    push()
    translate(width / 3, height / 3);
    rotate(millis() / 1000)
    fill(255)
    line(0, 0, 40, 40);
    pop();


    push()
    rotate(-millis() / 1000)
    translate(width / 3, height / 3);

    fill(255, 0, 0)
    rect(0, 0, 20, 20);
    pop()

    push()
    translate(mouseX, mouseY);
    fill(255, 255, 0)
    circle(0, 0, 10)
    rotate(-millis() / 1000)
    translate(40, 40);
    fill(255, 255, 0)
    rect(0, 0, 20, 20);
    pop()


    push()
    translate(100, 100);
    fill(0, 144, 255)
    circle(0,0,10)
    
    scale(mouseX / 100)
    fill(0, 0, 255,128)
    rect(0, 0, 40, 20);
 
    pop()

    push()
        translate(x, 100);
        rotate(millis() / 200)
        
        rect(0, 0, 20, 20);
        x+=2
        if(x>width)
            x=0
    pop()

}