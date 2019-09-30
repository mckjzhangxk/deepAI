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
			// joint->bindWorldToJointTransform=Matrix4f::identity();
			// joint->currentJointToWorldTransform=Matrix4f::identity();
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

		// Matrix4f current2World=parent->currentJointToWorldTransform*tr;
		// joint->currentJointToWorldTransform=current2World;
		// joint->bindWorldToJointTransform=current2World.inverse();

		
	}
	m_rootJoint=m_joints[0];
}


/*
计算当前需要旋转对齐的矩阵,z轴对齐 从 parent到child的向量，
另为两个轴与这个对齐的Z轴彼此垂直

这个矩阵是相对于一个父坐标系,而言的
*/
Matrix4f alignTransform(Joint * child){
	
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

void SkeletalModel::drawJoint_dfs(Joint* root){
	Matrix4f currentTransform=root->transform;
	m_matrixStack.push(currentTransform);
	Matrix4f currentState=m_matrixStack.top();
	//how to transform current joint
	glLoadMatrixf(currentState);
	glutSolidSphere(0.025f,12,12);

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
	Matrix4f currentState=m_matrixStack.top();
	
	//local translate
	Matrix4f tr=Matrix4f::translation(Vector3f(0,0,0.5));	
	


	for(Joint *child:root->children){ 	
		//我误认为d是child 与parent 的第四维度之差
		double d=sqrt(child->transform.getCol(3).xyz().absSquared());
		
		if(d>1e-4){
			//local scale,and align
			Matrix4f sc=Matrix4f::scaling(0.05,0.05,d);
			Matrix4f alignM=alignTransform(child);
			glLoadMatrixf(currentState*(alignM*sc*tr));
			glutSolidCube(1);

		}
		
		drawSkeleton_dfs(child);
	}
	m_matrixStack.pop();
} 
void SkeletalModel::drawSkeleton( )
{
	// Draw boxes between the joints. You will need to add a recursive helper function to traverse the joint hierarchy.
	drawSkeleton_dfs(m_rootJoint);

}

void SkeletalModel::setJointTransform(int jointIndex, float rX, float rY, float rZ)
{
	// Set the rotation part of the joint's transformation matrix based on the passed in Euler angles.
	Matrix3f M=Matrix3f::rotateZ(rZ)*Matrix3f::rotateY(rY)*Matrix3f::rotateX(rX);
	m_joints[jointIndex]->transform.setSubmatrix3x3(0,0,M);
}

void SkeletalModel::computeBindWorldToJointTransformsHelper(Joint* root){
 


	m_matrix_mesh.push(root->transform);
	Matrix4f current=m_matrix_mesh.top();
	root->bindWorldToJointTransform=current.inverse();
	for(Joint* child:root->children){
		computeBindWorldToJointTransformsHelper(child);
	}
	m_matrix_mesh.pop();
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


	computeBindWorldToJointTransformsHelper(m_rootJoint);

}
/*
设置root的currentJointToWorldTransform属性，
	这个属性是stack.top()* root->transform
1.把currentJointToWorldTransform入stack
2.遍历子节点
3.出stack

我犯的错误有2个，
1.不熟悉自定义stack的用法，push是与站顶相成后在压stack,默认占顶有个I
2.不能使用m_matrixStack,因为m_matrixStack的占有个camera_view transform
3.自定义的m_matrix_mesh 在updateCurrentJointToWorldTransformsHelper声名为
private的时候报错，为还不清楚原因 :(
*/
void  SkeletalModel::updateCurrentJointToWorldTransformsHelper(Joint* root){
	m_matrix_mesh.push(root->transform);

	Matrix4f current=m_matrix_mesh.top();
	root->currentJointToWorldTransform=current;
	for(Joint* child:root->children){
		updateCurrentJointToWorldTransformsHelper(child);
	}
	m_matrix_mesh.pop();

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
	
	updateCurrentJointToWorldTransformsHelper(m_rootJoint);

}

void SkeletalModel::updateMesh()
{
	// 2.3.2. This is the core of SSD.
	// Implement this method to update the vertices of the mesh
	// given the current state of the skeleton.
	// You will need both the bind pose world --> joint transforms.
	// and the current joint --> world transforms.
	


	for(int i=0;i<m_mesh.currentVertices.size();i++){
		Vector4f bindP(m_mesh.bindVertices[i],1);
		Vector3f currentP(0.0);
		vector<float> ws=m_mesh.attachments[i];
		 
		for(int j=0;j<ws.size();j++){
			float w=ws[j];
			if(w>0){
				Joint* joint=m_joints[j+1];
				currentP=currentP+w*(joint->currentJointToWorldTransform*(joint->bindWorldToJointTransform*bindP)).xyz();
			}
		}
		m_mesh.currentVertices[i]=currentP;
	}
}

