#include "commonHeader.h"
#include "myutils.h"

using namespace std;

float position[6]={
    0,0,
    0,0.5,
    0.5,0
};
GLuint buffers[1];
string v_shadefile=
        "#version 330 core\n"
        "\n"
        "layout(location = 0) in vec4 position;\n"
        "\n"
        "void main()\n"
        "{\n"
        "gl_Position = position;\n"
        "}\n";
string f_shadefile=
        "#version 330 core\n"
        "\n"
        "layout(location =0) out vec4 color;\n"
        "\n"
        "void main()\n"
        "{\n"
        "   color=vec4(1.0,0.0,0.0,1.0);\n"
        "}\n";



void init(){
    glGenBuffers(1,buffers);
    glBindBuffer(GL_ARRAY_BUFFER,buffers[0]);
    glBufferData(GL_ARRAY_BUFFER,sizeof(float)*6,position,GL_STATIC_DRAW);
    //2represent (x,y),false ->no normal,stride=bytes between 2 attribute
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(float)*2,0);
    glEnableVertexAttribArray(0);

}

void display(){
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES,0,3);
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

    int shader=createShader(v_shadefile,f_shadefile);
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

    glfwTerminate();

}
