
ARCH=  -gencode arch=compute_30,code=\"compute_30,sm_30\" \
			-gencode arch=compute_35,code=\"compute_35,sm_35\" \
			-gencode arch=compute_37,code=\"compute_37,sm_37\" \
			-gencode arch=compute_50,code=\"compute_50,sm_50,sm_52\" \

#ARCH=  -gencode arch=compute_50,code=\"compute_50,sm_50,sm_52\" 

NVCC_FLAGS=--std=c++11 -O3 ${ARCH} -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -lineinfo -Xcompiler -rdynamic  -Xptxas -v


.PHONY: libs all samples unit_tests perf_tests clean tests objs

INC=src/include
INCLUDES=$(wildcard ${INC}/*.h) $(wildcard ${INC}/*/*.h)
TUNE_INC=$(wildcard src/tune/*.h)

INST_SRCS=$(wildcard src/instantiations/*.cu)
INST_OBJS=$(INST_SRCS:.cu=.o)

SRC_SRCS=$(wildcard src/*.cu)
SRC_OBJS=$(SRC_SRCS:.cu=.o)

libs:  libxmp.a libxmp.so

all: objs libs samples unit_tests perf_tests tune
	
objs: tune.o xmp.o operators.o ${INST_OBJS} ${SRC_OBJS}


tune: ${INST_OBJS} ${SRC_OBJS} tune.o
	nvcc ${NVCC_FLAGS} $^ -o $@

libxmp.a: ${INST_OBJS} ${SRC_OBJS}
	nvcc ${NVCC_FLAGS} --lib $^ -o $@ 

libxmp.so: ${INST_OBJS} ${SRC_OBJS}
	nvcc ${NVCC_FLAGS} --shared $^ -o $@ 

samples: libs
	make -C samples

unit_tests: libs
	make -C unit_tests

perf_tests: libs
	make -C perf_tests

clean:
	rm -f *.o *.a *.so src/*.o src/instantiations/*.o

tests: libxmp.a
	make -C unit_tests run

make cleanall:  clean
	make -C samples clean
	make -C unit_tests clean
	make -C perf_tests clean

tune.o: src/tune/tune.cu
	nvcc ${NVCC_FLAGS} -I${INC} $< -c -o $@

%.o : %.cu ${INCLUDES}
	nvcc ${NVCC_FLAGS} -I${INC} $< -c -o $@
