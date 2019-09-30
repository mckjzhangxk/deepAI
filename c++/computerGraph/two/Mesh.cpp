#include "Mesh.h"

using namespace std;

void Mesh::load( const char* filename )
{
	// 2.1.1. load() should populate bindVertices, currentVertices, and faces

	// Add your code here.
	ifstream fin(filename);
	if(!fin){
		cout<<"error when load file:"<<filename<<endl;
		exit(0);
	}
	const int maxbuf=1024;
	char buf[maxbuf];
	while (fin.getline(buf,maxbuf))
	{
		stringstream ss(buf);
		string type;
		ss>>type;
		if(type=="v"){
			float x,y,z;
			ss>>x;ss>>y;ss>>z;
			bindVertices.push_back(Vector3f(x,y,z));
		}else if (type=="f"){
			int a,b,c;
			ss>>a;ss>>b;ss>>c;
			faces.push_back(Tuple3u(a,b,c));
		}
		
	}
	
	cout<<"load vertex #"<<bindVertices.size()<<",face #"<<faces.size()<<endl;;
	// make a copy of the bind vertices as the current vertices
	currentVertices = bindVertices;
}
void drawPoint(const Vector3f &point,const Vector3f& normal){
        glNormal3d(normal.x(),normal.y(),normal.z());  
		glVertex3d(point.x(),point.y(),point.z());
}

void drawTriangle(const vector<vector<Vector3f> > pts){
    glBegin(GL_TRIANGLES);
        for(int i=0;i<pts.size();i++){
            drawPoint(pts[i][0],pts[i][1]);
        }
    glEnd();
}
Vector3f faceNormal(Vector3f v1,Vector3f v2,Vector3f v3){
	Vector3f nm=Vector3f::cross(v2-v1,v3-v1);
	nm.normalize();
	return nm;
}
void Mesh::draw()
{
	// Since these meshes don't have normals
	// be sure to generate a normal per triangle.
	// Notice that since we have per-triangle normals
	// rather than the analytical normals from
	// assignment 1, the appearance is "faceted".

	for(Tuple3u face :faces){
		Vector3f v1=currentVertices[face[0]-1];
		Vector3f v2=currentVertices[face[1]-1];
		Vector3f v3=currentVertices[face[2]-1];
		Vector3f n=faceNormal(v1,v2,v3);
		drawTriangle({
			{v1,n},
			{v2,n},
			{v3,n}
		});
	}
}

void Mesh::loadAttachments( const char* filename, int numJoints )
{
	// 2.2. Implement this method to load the per-vertex attachment weights
	// this method should update m_mesh.attachments
	ifstream fin(filename);
	if(!fin){
		cout<<"error when load file:"<<filename<<endl;
		exit(0);
	}
	const int maxBuffer=1024;
	char buf[maxBuffer];
	while (fin.getline(buf,maxBuffer))
	{
		stringstream ss(buf);
		float w=0;
		vector<float> ws;
		//如果先push_back，是对ws做个个clone,加入进去之后对ws的更改不会反映到attachments中！！
		// attachments.push_back(ws);
		for(int i=0;i<numJoints-1;i++){
			ss>>w;
			ws.push_back(w);
		}
		attachments.push_back(ws);	
	}
	cout<<"acctch num:"<<attachments.size()<<endl;
}
