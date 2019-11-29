#include "myutils.h"
#include <fstream>
#include <sstream>
#include <string>
bool api_init(){
    #ifdef USE_GLEW
        GLenum err = glewInit();
        if (err)
        {
            /* Problem: glewInit failed, something is seriously wrong. */
            std::cerr<<"init glew error!"<<std::endl;
            exit(0);
        }
    #else
        GLenum err = gl3wInit();
        if(err){
          std::cerr<<"init gl3w error!"<<std::endl;
          exit(0);
        }
    #endif
}
bool checkError(int id){
    int result;
    glGetShaderiv(id,GL_COMPILE_STATUS,&result);
    if(result==GL_FALSE){
        int length;
        glGetShaderiv(id,GL_INFO_LOG_LENGTH,&length);
        char* message=(char *)alloca(sizeof(char)*length);

        glGetShaderInfoLog(id,length,&length,message);
        std::cerr<<message<<std::endl;
    }
    return result==GL_FALSE;
}

unsigned int compile_shader(std::string & sourcefile,unsigned int type){
    GLuint id=glCreateShader(type);
    const char *src=sourcefile.c_str();
    glShaderSource(id,1,&src,nullptr);
    //error hander
    glCompileShader(id);
    int result;
    glGetShaderiv(id,GL_COMPILE_STATUS,&result);

    if(checkError(id)){
        return  0;
    }
    return id;
}

int createShader(std::string& vsfile,std::string& fsfile){
    unsigned int program=glCreateProgram();
    unsigned int vs=compile_shader(vsfile,GL_VERTEX_SHADER);
    unsigned int fs=compile_shader(fsfile,GL_FRAGMENT_SHADER);

    glAttachShader(program,vs);
    glAttachShader(program,fs);

    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return  program;
}
ShaderSource parseShader(const std::string & filename){
    enum class TYPE{None=-1,VERTEX=0,FRAGMENT=1};

    std::ifstream fin(filename);
    std::string line;
    std::stringstream ss[2];
    TYPE type=TYPE::None;
    while (getline(fin,line)) {
        if(line.find("#shader")!=std::string::npos){
            if(line.find("vertex")!=std::string::npos){
                type=TYPE::VERTEX;
            }else if(line.find("fragment")!=std::string::npos){
                type=TYPE::FRAGMENT;
            }
        }else{
            ss[(int)type]<<line<<std::endl;
        }

    }
    ShaderSource ret;

    ret.vertex=ss[0].str();
    ret.fragment=ss[1].str();

    return {ss[0].str(),ss[1].str()};
}
