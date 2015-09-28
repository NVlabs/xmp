CC=nvcc
GMP_INC_DIR?=/shared/apps/rhel-6.2/tools/gmp-6.0.0a/gcc-4.8.2/include
GMP_LIB_DIR?=/shared/apps/rhel-6.2/tools/gmp-6.0.0a/gcc-4.8.2/lib
GTEST_INC_DIR?=gtest-1.7.0/include
GTEST_LIB_DIR?=gtest-1.7.0/build

CCFLAGS=-I${GMP_INC_DIR} -I${GTEST_INC_DIR} -Xcompiler -fopenmp
LDFLAGS=../libxmp.a ${GTEST_LIB_DIR}/libgtest.a -L ${GMP_LIB_DIR} -l gmp

.PHONY: clean all run lib

all:  test_all

test_all: test.cu ../libxmp.a | lib
	${CC} -o $@ $< ${CCFLAGS} ${LDFLAGS} 

lib: 
	make -C ../

run: test_all
	./test_all

clean: 
	rm -f test_all
