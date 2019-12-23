#include "commonHeader.h"
#include "myutils.h"
#include "vmath.h"
#include "vertexarray.h"
#include "vertexbuffer.h"
#include "indexbuffer.h"
#include "vertexlayout.h"

using namespace std;

float position[]={
    -0.8,-0.8,
    1,0,0,
    0.8,-0.8,
    0,1,0,
    0.8,0.8,
    0,0,1,
    -0.8,0.8,
    1,0,0
};
unsigned int indices[]={
    0,1,2,
    0,2,3
};
VertexArray * va;
VertexBuffer *vbuffer;
IndexBuffer  *ibuffer;
//GLuint vbo;






void init(){

//    glGenVertexArrays(1,&vbo);
//    glBindVertexArray(vbo);

    //create vertex buffer,and transfer data to this buffer
    vbuffer=new VertexBuffer(position,sizeof (position));
    //define a layout
    //2represent (x,y),false ->no normal,stride=bytes between 2 attribute
//    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(float)*2,0);
//    glEnableVertexAttribArray(0);

    //create a index buffer,and transfer data to this buffer
    ibuffer=new IndexBuffer(indices,sizeof (indices)/sizeof (unsigned int));


    VertexLayout layout;
    layout.add(2);
    layout.add(3);

    va=new VertexArray(*vbuffer,layout);

    //unbind
    va->unbind();
    vbuffer->ubind();
    vbuffer->ubind();
}

void display(){
    glClear(GL_COLOR_BUFFER_BIT);
//    glDrawArrays(GL_TRIANGLES,0,6);
//    glBindVertexArray(vbo);
    va->bind();
    ibuffer->bind();
    ASSERT(glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,nullptr));
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

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
//    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_COMPAT_PROFILE);

    api_init();

    init();
    ShaderSource source=parseShader("res/shader/myshader.shader");
    int shader=createShader(source.vertex,source.fragment);


    glUseProgram(shader);
    //uniform!

    GLfloat u_color[12]{1,0,0,1,\
                       1,1,0,1,\
                       0,1,1,1,\
                      };
    int location=glGetUniformLocation(shader,"u_color");
    ASSERT(glUniform4fv(location,3,u_color));
//    ASSERT(glUniform4f(location,1,0,0,0));

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

    delete vbuffer;
    delete ibuffer;
}
int main( int argc, char** argv ){
    vmath::mat4 t1=vmath::translate(10.f,0.f,0.f);
    vmath::mat4 s1=vmath::scale(5.f);
    vmath::mat4 st=s1*t1;

    run();
}
