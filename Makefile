# This is Makefile to compile the program dependent on openCV and Eigen3 libs
# compiler command
CXX=g++
# linker command
LD=g++

# files removalmakefile for opencv program
RM=/bin/rm -f

#DEBUG mode
#DEBUG=
DEBUG=-DDEBUG

#library to use when compiling
LIBS=$(shell pkg-config --libs opencv) -lm -ldl -lGL -lGLU -lpthread 

# header files
HEADERS=$(shell pkg-config --cflags opencv) 

#library path
LIBS_PATH=

# compiler flags
CXXFLAGS=-g $(HEADERS) $(DEBUG)
# linker flags
LDFLAGS=-g $(LIBS) $(LIBS_PATH) $(DEBUG)

#cpp source files (.cpp)
SRCS=$(wildcard *.cpp)
#object files (.o)
PROG_OBJS=$(patsubst %.cpp,%.o,$(SRCS))

#program's executable
PROG=object_recognition_main

#top-level rule
all: $(PROG)

# link object files into an executable program
$(PROG):$(PROG_OBJS)
	$(LD) $(LDFLAGS) $(PROG_OBJS) -o $(PROG)

#compile cpp source files into object files
%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $<

#clean the program file and the object files
clean:
	$(RM) $(PROG_OBJS) $(PROG)

.PHONY: clean


