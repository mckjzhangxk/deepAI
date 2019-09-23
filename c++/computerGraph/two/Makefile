#INCFLAGS = -I /mit/6.837/public/include/vecmath
#INCFLAGS += -I /mit/glut/include
#LINKFLAGS = -L /mit/6.837/public/lib -l vecmath
#LINKFLAGS += -L /mit/glut/lib -lGL -lGLU -lglut -lX11 -lXi
#mINCFLAGS  = -I include/GL
#INCFLAGS += -I include/vecmath
#INCFLAGS += -I ~/vecmath/include

#g++ -I/usr/local/include -I/usr/include/freetype2 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_THREAD_SAFE -D_REENTRANT -o 'main' *.cpp /usr/local/lib/libfltk_gl.a -lGLU -lGL /usr/local/lib/libfltk.a -lXrender -lXcursor -lXfixes -lXext -lXft -lfontconfig -lXinerama -lpthread -ldl -lm -lX11 -lvecmath -lglut

LINKFLAGS  = -lglut -lGL -lGLU
LINKFLAGS += -lvecmath
# LINKFLAGS += /usr/local/lib/libfltk_gl.a 
# LINKFLAGS += /usr/local/lib/libfltk.a

LINKFLAGS += -lfltk_gl 
LINKFLAGS += -lfltk


LINKFLAGS += -lXrender -lXcursor -lXfixes -lXext -lXft -lfontconfig -lXinerama -lpthread -ldl -lm -lX11

CFLAGS    = -g
CFLAGS    += -DSOLN
CFLAGS    += -std=c++11
CC        = g++
SRCS      = bitmap.cpp camera.cpp MatrixStack.cpp modelerapp.cpp modelerui.cpp ModelerView.cpp Joint.cpp SkeletalModel.cpp Mesh.cpp main.cpp
OBJS      = $(SRCS:.cpp=.o)
PROG      = a2

all: $(SRCS) $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LINKFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -c -o $@ $(INCFLAGS) 

depend:
	makedepend $(INCFLAGS) -Y $(SRCS)

clean:
	rm $(OBJS) $(PROG)

bitmap.o: bitmap.h
camera.o: camera.h
Mesh.o: Mesh.h
MatrixStack.o: MatrixStack.h
modelerapp.o: modelerapp.h ModelerView.h modelerui.h bitmap.h camera.h
modelerui.o: modelerui.h ModelerView.h bitmap.h camera.h modelerapp.h
ModelerView.o: ModelerView.h camera.h
SkeletalModel.o: MatrixStack.h ModelerView.h Joint.h modelerapp.h

