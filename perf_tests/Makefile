GMP_INCLUDE_DIR?=/shared/apps/rhel-6.2/tools/gmp-6.0.0a/gcc-4.8.2/include
GMP_LIB_DIR?=/shared/apps/rhel-6.2/tools/gmp-6.0.0a/gcc-4.8.2/lib

NVCC_FLAGS=-O3 -Xcompiler -fopenmp -I${GMP_INCLUDE_DIR}
LIBS=-L ${GMP_LIB_DIR} -l gmp

.PHONY: lib
perf: main.cu ../libxmp.a | lib
	nvcc ${NVCC_FLAGS} $< -o $@ ../libxmp.a ${LIBS}

lib: 
	make -C ../

clean:
	rm -f perf

