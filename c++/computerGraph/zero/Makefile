INCFLAGS  = -I /usr/include/GL
INCFLAGS += -I include/vecmath

LINKFLAGS  = -lglut -lGL -lGLU -lvecmath



CFLAGS    = -O2 -std=c++11
CC        = g++
SRCS      = *.cpp
# OBJS      = $(SRCS:.cpp=.o)
PROG      = a0

all: $(SRCS) $(PROG)

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(INCFLAGS) $(LINKFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -c -o $@ $(INCFLAGS)

depend:
	makedepend $(INCFLAGS) -Y $(SRCS)

clean:
	rm $(OBJS) $(PROG)

