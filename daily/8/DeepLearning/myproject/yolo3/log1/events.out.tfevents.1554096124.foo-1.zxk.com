       �K"	   �g(�Abrain.Event:2tֿK4      �)3	6��g(�A"�
L
PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
*scope1/W1/Initializer/random_uniform/shapeConst*
_class
loc:@scope1/W1*
valueB *
dtype0*
_output_shapes
: 
�
(scope1/W1/Initializer/random_uniform/minConst*
_class
loc:@scope1/W1*
valueB
 *׳ݿ*
dtype0*
_output_shapes
: 
�
(scope1/W1/Initializer/random_uniform/maxConst*
_class
loc:@scope1/W1*
valueB
 *׳�?*
dtype0*
_output_shapes
: 
�
2scope1/W1/Initializer/random_uniform/RandomUniformRandomUniform*scope1/W1/Initializer/random_uniform/shape*
dtype0*
_output_shapes
: *

seed *
T0*
_class
loc:@scope1/W1*
seed2 
�
(scope1/W1/Initializer/random_uniform/subSub(scope1/W1/Initializer/random_uniform/max(scope1/W1/Initializer/random_uniform/min*
T0*
_class
loc:@scope1/W1*
_output_shapes
: 
�
(scope1/W1/Initializer/random_uniform/mulMul2scope1/W1/Initializer/random_uniform/RandomUniform(scope1/W1/Initializer/random_uniform/sub*
T0*
_class
loc:@scope1/W1*
_output_shapes
: 
�
$scope1/W1/Initializer/random_uniformAdd(scope1/W1/Initializer/random_uniform/mul(scope1/W1/Initializer/random_uniform/min*
T0*
_class
loc:@scope1/W1*
_output_shapes
: 
�
	scope1/W1
VariableV2*
_output_shapes
: *
shared_name *
_class
loc:@scope1/W1*
	container *
shape: *
dtype0
�
scope1/W1/AssignAssign	scope1/W1$scope1/W1/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@scope1/W1*
validate_shape(*
_output_shapes
: 
d
scope1/W1/readIdentity	scope1/W1*
T0*
_class
loc:@scope1/W1*
_output_shapes
: 
O

scope1/mulMulPlaceholderscope1/W1/read*
T0*
_output_shapes
: 
�
*scope2/W2/Initializer/random_uniform/shapeConst*
_class
loc:@scope2/W2*
valueB *
dtype0*
_output_shapes
: 
�
(scope2/W2/Initializer/random_uniform/minConst*
_class
loc:@scope2/W2*
valueB
 *׳ݿ*
dtype0*
_output_shapes
: 
�
(scope2/W2/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@scope2/W2*
valueB
 *׳�?
�
2scope2/W2/Initializer/random_uniform/RandomUniformRandomUniform*scope2/W2/Initializer/random_uniform/shape*
T0*
_class
loc:@scope2/W2*
seed2 *
dtype0*
_output_shapes
: *

seed 
�
(scope2/W2/Initializer/random_uniform/subSub(scope2/W2/Initializer/random_uniform/max(scope2/W2/Initializer/random_uniform/min*
T0*
_class
loc:@scope2/W2*
_output_shapes
: 
�
(scope2/W2/Initializer/random_uniform/mulMul2scope2/W2/Initializer/random_uniform/RandomUniform(scope2/W2/Initializer/random_uniform/sub*
_output_shapes
: *
T0*
_class
loc:@scope2/W2
�
$scope2/W2/Initializer/random_uniformAdd(scope2/W2/Initializer/random_uniform/mul(scope2/W2/Initializer/random_uniform/min*
T0*
_class
loc:@scope2/W2*
_output_shapes
: 
�
	scope2/W2
VariableV2*
shared_name *
_class
loc:@scope2/W2*
	container *
shape: *
dtype0*
_output_shapes
: 
�
scope2/W2/AssignAssign	scope2/W2$scope2/W2/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@scope2/W2*
validate_shape(*
_output_shapes
: 
d
scope2/W2/readIdentity	scope2/W2*
T0*
_class
loc:@scope2/W2*
_output_shapes
: 
O

scope2/mulMulPlaceholderscope2/W2/read*
_output_shapes
: *
T0
2
initNoOp^scope1/W1/Assign^scope2/W2/Assign"���1�      ���+	���g(�AJ�%
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.12.02v1.12.0-0-ga6d8ffae09�
L
PlaceholderPlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
*scope1/W1/Initializer/random_uniform/shapeConst*
_class
loc:@scope1/W1*
valueB *
dtype0*
_output_shapes
: 
�
(scope1/W1/Initializer/random_uniform/minConst*
_class
loc:@scope1/W1*
valueB
 *׳ݿ*
dtype0*
_output_shapes
: 
�
(scope1/W1/Initializer/random_uniform/maxConst*
_class
loc:@scope1/W1*
valueB
 *׳�?*
dtype0*
_output_shapes
: 
�
2scope1/W1/Initializer/random_uniform/RandomUniformRandomUniform*scope1/W1/Initializer/random_uniform/shape*
dtype0*
_output_shapes
: *

seed *
T0*
_class
loc:@scope1/W1*
seed2 
�
(scope1/W1/Initializer/random_uniform/subSub(scope1/W1/Initializer/random_uniform/max(scope1/W1/Initializer/random_uniform/min*
T0*
_class
loc:@scope1/W1*
_output_shapes
: 
�
(scope1/W1/Initializer/random_uniform/mulMul2scope1/W1/Initializer/random_uniform/RandomUniform(scope1/W1/Initializer/random_uniform/sub*
T0*
_class
loc:@scope1/W1*
_output_shapes
: 
�
$scope1/W1/Initializer/random_uniformAdd(scope1/W1/Initializer/random_uniform/mul(scope1/W1/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@scope1/W1
�
	scope1/W1
VariableV2*
_class
loc:@scope1/W1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
scope1/W1/AssignAssign	scope1/W1$scope1/W1/Initializer/random_uniform*
T0*
_class
loc:@scope1/W1*
validate_shape(*
_output_shapes
: *
use_locking(
d
scope1/W1/readIdentity	scope1/W1*
T0*
_class
loc:@scope1/W1*
_output_shapes
: 
O

scope1/mulMulPlaceholderscope1/W1/read*
_output_shapes
: *
T0
�
*scope2/W2/Initializer/random_uniform/shapeConst*
_class
loc:@scope2/W2*
valueB *
dtype0*
_output_shapes
: 
�
(scope2/W2/Initializer/random_uniform/minConst*
_class
loc:@scope2/W2*
valueB
 *׳ݿ*
dtype0*
_output_shapes
: 
�
(scope2/W2/Initializer/random_uniform/maxConst*
_class
loc:@scope2/W2*
valueB
 *׳�?*
dtype0*
_output_shapes
: 
�
2scope2/W2/Initializer/random_uniform/RandomUniformRandomUniform*scope2/W2/Initializer/random_uniform/shape*
dtype0*
_output_shapes
: *

seed *
T0*
_class
loc:@scope2/W2*
seed2 
�
(scope2/W2/Initializer/random_uniform/subSub(scope2/W2/Initializer/random_uniform/max(scope2/W2/Initializer/random_uniform/min*
T0*
_class
loc:@scope2/W2*
_output_shapes
: 
�
(scope2/W2/Initializer/random_uniform/mulMul2scope2/W2/Initializer/random_uniform/RandomUniform(scope2/W2/Initializer/random_uniform/sub*
_output_shapes
: *
T0*
_class
loc:@scope2/W2
�
$scope2/W2/Initializer/random_uniformAdd(scope2/W2/Initializer/random_uniform/mul(scope2/W2/Initializer/random_uniform/min*
T0*
_class
loc:@scope2/W2*
_output_shapes
: 
�
	scope2/W2
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@scope2/W2*
	container *
shape: 
�
scope2/W2/AssignAssign	scope2/W2$scope2/W2/Initializer/random_uniform*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@scope2/W2*
validate_shape(
d
scope2/W2/readIdentity	scope2/W2*
T0*
_class
loc:@scope2/W2*
_output_shapes
: 
O

scope2/mulMulPlaceholderscope2/W2/read*
_output_shapes
: *
T0
2
initNoOp^scope1/W1/Assign^scope2/W2/Assign""�
	variables��
[
scope1/W1:0scope1/W1/Assignscope1/W1/read:02&scope1/W1/Initializer/random_uniform:08
[
scope2/W2:0scope2/W2/Assignscope2/W2/read:02&scope2/W2/Initializer/random_uniform:08"�
trainable_variables��
[
scope1/W1:0scope1/W1/Assignscope1/W1/read:02&scope1/W1/Initializer/random_uniform:08
[
scope2/W2:0scope2/W2/Assignscope2/W2/read:02&scope2/W2/Initializer/random_uniform:08���