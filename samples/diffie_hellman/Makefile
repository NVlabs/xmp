
NVCC_FLAGS :=-O3 -I../../src/include

.PHONY: all lib

all: dh.exe

dh.exe: diffie_hellman.cc ../../libxmp.a | lib
	nvcc -o $@ ${NVCC_FLAGS} $<  ../../libxmp.a 
	
lib: 
	make -C ../../

clean:
	rm -f  *.exe

