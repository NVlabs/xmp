
ARCH=  -gencode arch=compute_30,code=\"compute_30,sm_30\" \
			-gencode arch=compute_35,code=\"compute_35,sm_35\" \
			-gencode arch=compute_37,code=\"compute_37,sm_37\" \
			-gencode arch=compute_50,code=\"compute_50,sm_50,sm_52\" \

#ARCH=  -gencode arch=compute_50,code=\"compute_50,sm_50,sm_52\" 

NVCC_FLAGS=--std=c++11 -O3 ${ARCH} -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -lineinfo -Xcompiler -rdynamic  -Xptxas -v


.PHONY: libs all samples unit_tests perf_tests clean tests

INC=src/include
INCLUDES=$(wildcard ${INC}/*.h) $(wildcard ${INC}/*/*.h)
TUNE_INC=$(wildcard src/tune/*.h)

libs:  libxmp.a libxmp.so

all: libs samples unit_tests perf_tests tune

xmp.o: src/xmp.cu ${INCLUDES}
	nvcc ${NVCC_FLAGS} -I${INC} $< -c -o $@

operators.o: src/operators.cu ${INCLUDES} ${TUNE_INC}
	nvcc ${NVCC_FLAGS} -I${INC} $< -c -o $@

instantiations.o: src/instantiations/instantiations.cu ${INCLUDES} 
	nvcc ${NVCC_FLAGS} -I${INC} $< -c -o $@

tune.o: src/tune/tune.cu 
	nvcc ${NVCC_FLAGS} -I${INC} $< -c -o $@

tune: tune.o xmp.o operators.o instantiations.o
	nvcc ${NVCC_FLAGS} $^ -o $@

libxmp.a: xmp.o operators.o instantiations.o
	nvcc ${NVCC_FLAGS} --lib $^ -o $@ 

libxmp.so: xmp.o operators.o instantiations.o
	nvcc ${NVCC_FLAGS} --shared $^ -o $@ 

samples: libs
	make -C samples

unit_tests: libs
	make -C unit_tests

perf_tests: libs
	make -C perf_tests

clean:
	rm -f *.o *.a *.so

tests: libxmp.a
	make -C unit_tests run

make cleanall:  clean
	make -C samples clean
	make -C unit_tests clean
	make -C perf_tests clean
