
NVCC_FLAGS :=-O3 -I../../src/include

.PHONY: all lib

all: rsa.exe

rsa.exe: rsa.cc ../../libxmp.a | lib
	nvcc -o $@ ${NVCC_FLAGS} $<  ../../libxmp.a 
	
lib: 
	make -C ../../

clean:
	rm -f  *.exe

