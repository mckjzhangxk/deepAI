#include "SkeletalModel.h"

#include <FL/Fl.H>

using namespace std;

void SkeletalModel::load(const char *skeletonFile, const char *meshFile, const char *attachmentsFile)
{
	loadSkeleton(skeletonFile);

	m_mesh.load(meshFile);
	m_mesh.loadAttachments(attachmentsFile, m_joints.size());

	computeBindWorldToJointTransforms();
	updateCurrentJointToWorldTransforms();
}

void SkeletalModel::draw(Matrix4f cameraMatrix, bool skeletonVisible)
{
	// draw() gets called whenever a redraw is required
	// (after an update() occurs, when the camera moves, the window is resized, etc)

	m_matrixStack.clear();
	m_matrixStack.push(cameraMatrix);

	if( skeletonVisible )
	{
		drawJoints();

		drawSkeleton();
	}
	else
	{
		// Clear out any weird matrix we may have been using for drawing the bones and revert to the camera matrix.
		glLoadMatrixf(m_matrixStack.top());

		// Tell the mesh to draw itself.
		m_mesh.draw();
	}
}

void SkeletalModel::loadSkeleton( const char* filename )
{
	// Load the skeleton from file here.
	ifstream fin(filename);
	cout<<filename<<endl;
	if(!fin){
		cout<<"error input file"<<endl;
		exit(0);
	}
	int MAXBUFFER=1024;
	char BUF[MAXBUFFER];
	
	while (fin.getline(BUF,MAXBUFFER))
	{
		stringstream sin(BUF);
		float x,y,z;
		int parentId;
		sin>>x;
		sin>>y;
		sin>>z;
		sin>>parentId;

		/*this set the transform relative to it's parent,
		R_B_A,whenever you have local coordinate c,you want
		to know what the coordinate of parent,simply multiply
		R_B_A with c
		cp=R_B_A*c
		*/
		Joint* joint=new Joint;
		m_joints.push_back(joint);

		Matrix4f tr=Matrix4f::identity();

		tr.setCol(3,Vector4f(x,y,z,1)); //I make a mistake,set col=4,ooh my god!!
		joint->transform=tr;


		//need double check whether is identity
		if(parentId==-1){
			joint->bindWorldToJointTransform=Matrix4f::identity();
			joint->currentJointToWorldTransform=Matrix4f::identity();
			continue;
		}
		 
		/*
		update parent's child list,note for non-root node!
		*/
		Joint * parent=m_joints[parentId];
		
		parent->children.push_back(joint);

		/*
		I'm not sure whether it's proper time to set currentJointToWorldTransform
		or bindWorldToJointTransform 

		I think currentJointToWorldTransform mean that a map to local frame to global frame,
		bindWorldToJointTransform is a map from global frame to local
		*/

		Matrix4f current2World=tr*parent->currentJointToWorldTransform;
		joint->currentJointToWorldTransform=current2World;
		joint->bindWorldToJointTransform=current2World.inverse();

		
	}
	m_rootJoint=m_joints[0];
}
void drawPoint(const Vector3f &point){
        glVertex3d(point.x(),point.y(),point.z());     
}
void drawLines(const vector<Vector3f> pts,const Vector3f &color,GLfloat linewidth){
    glLineWidth(linewidth);//放在begin里面不起作用
    /*
    GL_LINES:每2个点组成一条线
    GL_LINE_STRIP:一个点连接下一个点
    */
    glBegin(GL_LINES);    
        glColor3d(color.x(),color.y(),color.z());
        for(int i=0;i<pts.size();i++){
            drawPoint(pts[i]);
        }
    glEnd();
}
//画x,y,z三个轴
void drawAxis(GLfloat LINEWIDTH=1,GLfloat scale=0.1){
    drawLines({{0,0,0},{scale,0,0}},{1,0,0},LINEWIDTH);
    drawLines({{0,0,0},{0,scale,0}},{0,1,0},LINEWIDTH);
    drawLines({{0,0,0},{0,0,scale}},{0,0,1},LINEWIDTH);
}
/*
计算当前需要旋转对齐的矩阵,z轴对齐 从 parent到child的向量，
另为两个轴与这个对齐的Z轴彼此垂直

这个矩阵是相对于一个父坐标系,而言的
*/
Matrix4f calcNewFrame(Joint * child){
	
	Vector3f Z=child->transform.getCol(3).xyz();
	if(Z.x()==0&&Z.y()==0&&Z.z()==0){
		return Matrix4f::identity();
	}
	Z.normalize();
	Vector3f rndV(0,0,1);
	Vector3f Y=Vector3f::cross(Z,rndV);
	Y.normalize();
	Vector3f X=Vector3f::cross(Y,Z);
	X.normalize();

	Matrix3f f=Matrix3f::identity();
	f.setCol(0,X);
	f.setCol(1,Y);
	f.setCol(2,Z);

	Matrix4f M=Matrix4f::identity();
	M.setSubmatrix3x3(0,0,f);
	return M;
}
Matrix4f localTransform2Global(Matrix4f f,Matrix4f local2global){
	bool sigular;

	Matrix4f inv=local2global.inverse(&sigular,1e-3);
	return local2global*f*inv;
}
void SkeletalModel::drawJoint_dfs(Joint* root){
	Matrix4f currentTransform=root->transform;
	
	m_matrixStack.push(currentTransform);
	Matrix4f currentState=m_matrixStack.top();
	//how to transform current joint
	glLoadMatrixf(currentState);
	glutSolidSphere(0.025f,12,12);
	drawAxis(2,0.2);

	for(Joint *child:root->children){ 	
		drawJoint_dfs(child);
	}
	m_matrixStack.pop();
}
void SkeletalModel::drawJoints( )
{
	// Draw a sphere at each joint. You will need to add a recursive helper function to traverse the joint hierarchy.
	//
	// We recommend using glutSolidSphere( 0.025f, 12, 12 )
	// to draw a sphere of reasonable size.
	//
	// You are *not* permitted to use the OpenGL matrix stack commands
	// (glPushMatrix, glPopMatrix, glMultMatrix).
	// You should use your MatrixStack class
	// and use glLoadMatrix() before your drawing call.
	
	
	drawJoint_dfs(m_rootJoint);
	
}
void SkeletalModel::drawSkeleton_dfs(Joint* root){
	Matrix4f currentTransform=root->transform;
	m_matrixStack.push(currentTransform);

	Matrix4f tr=Matrix4f::translation(Vector3f(0,0,0.5));
	Matrix4f sc=Matrix4f::scaling(0.05,0.05,0.5);
	sc.print();
	
	Matrix4f currentState=m_matrixStack.top();
	//how to transform current joint
	bool sigular;

	glLoadMatrixf(currentState*(sc*tr)*currentState.inverse(&sigular,0.001));

	glutSolidCube(1);


	for(Joint *child:root->children){ 	
		drawSkeleton_dfs(child);
	}
	m_matrixStack.pop();
} 
void SkeletalModel::drawSkeleton( )
{
	// Draw boxes between the joints. You will need to add a recursive helper function to traverse the joint hierarchy.
	// m_matrixStack.clear();
	// drawSkeleton_dfs(m_rootJoint);
	// m_matrixStack.clear();
}

void SkeletalModel::setJointTransform(int jointIndex, float rX, float rY, float rZ)
{
	// Set the rotation part of the joint's transformation matrix based on the passed in Euler angles.
}


void SkeletalModel::computeBindWorldToJointTransforms()
{
	// 2.3.1. Implement this method to compute a per-joint transform from
	// world-space to joint space in the BIND POSE.
	//
	// Note that this needs to be computed only once since there is only
	// a single bind pose.
	//
	// This method should update each joint's bindWorldToJointTransform.
	// You will need to add a recursive helper function to traverse the joint hierarchy.
}

void SkeletalModel::updateCurrentJointToWorldTransforms()
{
	// 2.3.2. Implement this method to compute a per-joint transform from
	// joint space to world space in the CURRENT POSE.
	//
	// The current pose is defined by the rotations you've applied to the
	// joints and hence needs to be *updated* every time the joint angles change.
	//
	// This method should update each joint's bindWorldToJointTransform.
	// You will need to add a recursive helper function to traverse the joint hierarchy.
}

void SkeletalModel::updateMesh()
{
	// 2.3.2. This is the core of SSD.
	// Implement this method to update the vertices of the mesh
	// given the current state of the skeleton.
	// You will need both the bind pose world --> joint transforms.
	// and the current joint --> world transforms.
}

