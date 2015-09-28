
NVCC_FLAGS :=-O3 -I../../src/include

.PHONY: all lib

all: sample01.exe

sample01.exe: sample01.cc ../../libxmp.a | lib
	nvcc -o $@ ${NVCC_FLAGS} $<  ../../libxmp.a 
	
lib: 
	make -C ../../

clean:
	rm -f  *.exe

