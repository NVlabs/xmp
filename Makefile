
ARCH=  -gencode arch=compute_20,code=\"compute_20,sm_20,sm_30\" \
			-gencode arch=compute_35,code=\"compute_35,sm_35\" \
			-gencode arch=compute_37,code=\"compute_37,sm_37\" \
			-gencode arch=compute_50,code=\"compute_50,sm_50,sm_52\" \

NVCC_FLAGS=-O3 ${ARCH} -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -lineinfo -Xcompiler -rdynamic  -Xptxas -v


.PHONY: libs all samples unit_tests perf_tests clean tests

INC=src/include
INCLUDES=$(wildcard ${INC}/*.h) $(wildcard ${INC}/*/*.h)

libs:  libxmp.a libxmp.so

all: libs samples unit_tests perf_tests

xmp.o: src/xmp.cu ${INCLUDES}
	nvcc ${NVCC_FLAGS} -I${INC} $< -c -o $@

libxmp.a: xmp.o
	nvcc ${NVCC_FLAGS} --lib $^ -o $@ 

libxmp.so: xmp.o
	nvcc ${NVCC_FLAGS} --shared $^ -o $@ 

samples:
	make -C samples

unit_tests:
	make -C unit_tests

perf_tests:
	make -C perf_tests

clean:
	rm -f *.o *.a *.so

tests: libxmp.a
	make -C unit_tests run

make cleanall:  clean
	make -C samples clean
	make -C unit_tests clean
	make -C perf_tests clean
