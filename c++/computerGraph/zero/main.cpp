#include <GL/glut.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <vecmath.h>
using namespace std;

// Globals

// This is the list of points (3D vectors)
vector<Vector3f> vecv;

// This is the list of normals (also 3D vectors)
vector<Vector3f> vecn;

// This is the list of faces (indices into vecv and vecn)
vector<vector<unsigned> > vecf;

float field_angle=0;
// You will need more global variables to implement color and position changes
int current_step=0;
const int STEP_PER_COLOR=10;
const int G_SIZE=4;

GLfloat GPos[2]={1,1};
GLfloat angle=0;
const int INTERVAL=100;
bool rotate=false;

// These are convenience functions which allow us to call OpenGL 
// methods on Vec3d objects
inline void glVertex(const Vector3f &a) 
{ glVertex3fv(a); }

inline void glNormal(const Vector3f &a) 
{ glNormal3fv(a); }

void timefunc(int value){
    if(rotate){
        angle+=value;
        glutPostRedisplay();
    }
    glutTimerFunc(INTERVAL,timefunc,value);
}
// Called when the window is resized
// w, h - width and height of the window in pixels.
void reshapeFunc(int w, int h)
{
    // Always use the largest square viewport possible
    if (w > h) {
        glViewport((w - h) / 2, 0, h, h);
    } else {
        glViewport(0, (h - w) / 2, w, w);
    }
    // glViewport(0,0,w,h);
    // Set up a perspective view, with square aspect ratio
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // 50 degree fov, uniform aspect ratio, near = 1, far = 100
    gluPerspective(field_angle, 1.0*w/h, 1.0, 100.0);
}
// This function is called whenever a "Normal" key press is received.
void keyboardFunc( unsigned char key, int x, int y )
{
    switch ( key )
    {
    case 27: // Escape key
        exit(0);
        break;
    case 'c':
        
        current_step=(current_step+1)%(G_SIZE*STEP_PER_COLOR);
        
        cout << "Unhandled key press " << key << "." << endl; 
        break;
    case 'r':
        rotate=!rotate;  
        break;
    case 'x':
        field_angle+=5;
        {
            int W=glutGet(GLUT_WINDOW_WIDTH);
            int H=glutGet(GLUT_WINDOW_HEIGHT);
            cout<<W<<"xxx"<<H<<"ss"<<field_angle<<endl;
            reshapeFunc(W,H);
        }

        break;
    default:
        cout << "Unhandled key press " << key << "." << endl;        
    }

	// this will refresh the screen so that the user sees the color change
    glutPostRedisplay();
}

// This function is called whenever a "Special" key press is received.
// Right now, it's handling the arrow keys.
void specialFunc( int key, int x, int y )
{
    switch ( key )
    {
    case GLUT_KEY_UP:
        // add code to change light position
		GPos[1]+=0.5;
		break;
    case GLUT_KEY_DOWN:
        // add code to change light position
		GPos[1]-=0.5;
		break;
    case GLUT_KEY_LEFT:
        GPos[0]-=0.5;
		
		break;
    case GLUT_KEY_RIGHT:
        // add code to change light position
		GPos[0]+=0.5;
        
		break;
    }

	// this will refresh the screen so that the user sees the light position
    glutPostRedisplay();
}

// This function is responsible for displaying the object.
void drawScene(void)
{
    int i;

    // Clear the rendering window
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Rotate the image
    glMatrixMode( GL_MODELVIEW );  // Current matrix affects objects positions
    glLoadIdentity();              // Initialize to the identity

    // Position the camera at [0,0,5], looking at [0,0,0],
    // with [0,1,0] as the up direction.
    gluLookAt(0.0, 0.0, 5.0,
              0.0, 0.0, 0.0,
              0.0, 1.0, 0.0);
    glRotated(angle,0,1,0);

    // Set material properties of object

	// Here are some colors you might use - feel free to add more
    GLfloat diffColors[4][4] = { 
                                 {0.5, 0.5, 0.9, 1.0},
                                 {0.9, 0.5, 0.5, 1.0},
                                 {0.5, 0.9, 0.3, 1.0},
                                 {0.3, 0.8, 0.9, 1.0} };
    //自定义的过度
    int currentColorIndex=current_step/STEP_PER_COLOR;
    int nextColorIndex=(currentColorIndex+1)%G_SIZE;
    double t=((float)current_step/(float)STEP_PER_COLOR-currentColorIndex)/(nextColorIndex-currentColorIndex);
    GLfloat* c1=diffColors[currentColorIndex];
    GLfloat* c2=diffColors[nextColorIndex];
    GLfloat cnew[4];
    {
        for(int i=0;i<4;i++){
            cnew[i]=(1-t)*c1[i]+t*c2[i];
        }
    }
    //
	// Here we use the first color entry as the diffuse color
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,cnew);

	// Define specular color and shininess
    GLfloat specColor[] = {1.0, 1.0, 1.0, 1.0};
    GLfloat shininess[] = {100.0};

	// Note that the specular color and shininess can stay constant
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
  
    // Set light properties

    // Light color (RGBA)
    GLfloat Lt0diff[] = {1.0,1.0,1.0,1.0};
    // Light position
	GLfloat Lt0pos[] = {1.0f, 1.0f, 5.0f, 1.0f};


    glLightfv(GL_LIGHT0, GL_DIFFUSE, Lt0diff);
    Lt0pos[0]=GPos[0];
    Lt0pos[1]=GPos[1];
    cout<<"light position:("<<Lt0pos[0]<<","<<Lt0pos[1]<<","<<Lt0pos[2]<<")"<<endl;
    glLightfv(GL_LIGHT0, GL_POSITION, Lt0pos);

	// This GLUT method draws a teapot.  You should replace
	// it with code which draws the object you loaded.

  

    if(vecf.size()==0)
        glutSolidTeapot(1.0);
    //////////////////my code///////////////////////////////
    else
    // https://www.youtube.com/watch?v=Q_kFcRlLTk0
        for(unsigned int j=0;j<vecf.size();j++){
            vector<unsigned> face=vecf[j];
            unsigned int a=face[0]-1,b=face[1]-1,c=face[2]-1;
            unsigned int d=face[3]-1,e=face[4]-1,f=face[5]-1;
            unsigned int g=face[6]-1,h=face[7]-1,i=face[8]-1;

            glBegin(GL_TRIANGLES);

            glNormal3d(vecn[c][0],vecn[c][1],vecn[c][2]);
            glVertex3d(vecv[a][0],vecv[a][1],vecv[a][2]);

            glNormal3d(vecn[f][0],vecn[f][1],vecn[f][2]);
            glVertex3d(vecv[d][0],vecv[d][1],vecv[d][2]);

            glNormal3d(vecn[i][0],vecn[i][1],vecn[i][2]);
            glVertex3d(vecv[g][0],vecv[g][1],vecv[g][2]);
            glEnd();
        }
    
    ////////////////////////////////////////////////////////

    
    // Dump the image to the screen.
    glutSwapBuffers();


}

// Initialize OpenGL's rendering modes
void initRendering()
{
    glEnable(GL_DEPTH_TEST);   // Depth testing must be turned on
    glEnable(GL_LIGHTING);     // Enable lighting calculations
    glEnable(GL_LIGHT0);       // Turn on light #0.
}



void loadInput()
{
	const unsigned int MAXBUFFER=1024;
    char BUFFER[MAXBUFFER];
    
    while (cin.getline(BUFFER,MAXBUFFER))
    {
        stringstream ss(BUFFER);   
        string objtype;
        ss>>objtype;

        if(objtype=="v"){
            Vector3f v;
            ss>>v[0]>>v[1]>>v[2];
            vecv.push_back(v);
            
        }else if (objtype=="vn")
        {
            Vector3f v;
            ss>>v[0]>>v[1]>>v[2];
            vecn.push_back(v);

        }else if (objtype=="f")
        {
            vector<unsigned> faceinfo;
            string s;
            string delimiter="/";
            //读取三段数据 a/b/c d/e/f g/h/i
            for(int i=0;i<3;i++){
                
                ss>>s;
                int start=0;
                int currentfind=-1;
                //读取每段数据的三个字段
                for(int j=0;j<3;j++){
                    currentfind=s.find(delimiter,start);
                    unsigned int index=atoi(s.substr(start,currentfind-start).c_str());
                    start=currentfind+1;
                    faceinfo.push_back(index);
                }
            }

            vecf.push_back(faceinfo);
        }
        
        
    }
    
    cout<<"成功读取数据"<<endl;
    cout<<"顶点:"<<vecv.size()<<endl;
    cout<<"顶点法线:"<<vecn.size()<<endl;
    cout<<"面:"<<vecf.size()<<endl;
}

// Main routine.
// Set up OpenGL, define the callbacks and start the main loop
int main( int argc, char** argv )
{
    loadInput();

    glutInit(&argc,argv);

    // We're going to animate it, so double buffer 
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );

    // Initial parameters for window position and size
    glutInitWindowPosition( 60, 60 );
    glutInitWindowSize( 360, 360 );
    glutCreateWindow("Assignment 0");

    // Initialize OpenGL parameters.
    initRendering();

    // Set up callback functions for key presses
    glutKeyboardFunc(keyboardFunc); // Handles "normal" ascii symbols
    glutSpecialFunc(specialFunc);   // Handles "special" keyboard keys

     // Set up the callback function for resizing windows
    glutReshapeFunc( reshapeFunc );

    // Call this whenever window needs redrawing
    glutDisplayFunc( drawScene );

    //注册选择场景
    glutTimerFunc(INTERVAL,timefunc,5);

    // Start the main loop.  glutMainLoop never returns.
    glutMainLoop( );

    return 0;	// This line is never reached.
}
