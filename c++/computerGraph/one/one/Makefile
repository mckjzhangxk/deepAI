INCFLAGS  = -I /usr/include/GL
INCFLAGS += -I vecmath/include

LINKFLAGS = -lglut -lGL -lGLU
LINKFLAGS += -L vecmath/lib -lvecmath

CFLAGS    = -O2 -w -ansi -DSOLN --std=c++11
CC        = g++
SRCS      = main.cpp parse.cpp curve.cpp surf.cpp camera.cpp
OBJS      = $(SRCS:.cpp=.o)
PROG      = a1
#VECMATH   = vecmath/src/*.cpp
all: $(SRCS) $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(INCFLAGS) $(LINKFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -c -o $@ $(INCFLAGS)

depend:
	makedepend $(INCFLAGS) -Y $(SRCS)

clean:
	rm $(OBJS) $(PROG)
