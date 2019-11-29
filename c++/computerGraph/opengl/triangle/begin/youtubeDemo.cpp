#include "commonHeader.h"
#include "myutils.h"

using namespace std;

float position[]={
    -0.8,-0.8,
    0.8,-0.8,
    0.8,0.8,
    -0.8,0.8,



};
GLuint buffers[1];

GLuint ibo;
unsigned int indices[]={
    0,1,3,
    1,2,3
};




void init(){

    glGenBuffers(1,buffers);
    glBindBuffer(GL_ARRAY_BUFFER,buffers[0]);
    glBufferData(GL_ARRAY_BUFFER,sizeof(position),position,GL_STATIC_DRAW);
    //2represent (x,y),false ->no normal,stride=bytes between 2 attribute
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(float)*2,0);
    glEnableVertexAttribArray(0);


    glGenBuffers(1,&ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(indices),indices,GL_STATIC_DRAW);

}

void display(){
    glClear(GL_COLOR_BUFFER_BIT);
//    glDrawArrays(GL_TRIANGLES,0,6);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ibo);
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,nullptr);
}

void run(){
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);


    api_init();

    init();
    ShaderSource source=parseShader("res/shader/myshader.shader");
    int shader=createShader(source.vertex,source.fragment);


    glUseProgram(shader);
    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */

        display();
        /* Swap front and back buffers */
        glfwSwapBuffers(window);
        /* Poll for and process events */
        glfwPollEvents();
    }
    glDeleteProgram(shader);
    glfwTerminate();

}
int main( int argc, char** argv ){

    run();
}
