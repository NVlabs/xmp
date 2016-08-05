/***
Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
***/
#include "xmp.h"
#pragma once

#include <stdio.h>

typedef uint32_t xmpLimb_t;
#define xmpNativeOrder (-1)
#define xmpNativeEndian (-1)

#define ROUND_UP(n,d) (((n)+(d)-1)/(d)*(d))
#define DIV_ROUND_UP(n,d) (((n)+(d)-1)/(d))
#define MIN(a,b) (((a)<(b)) ? (a) : (b))
#define MAX(a,b) (((a)<(b)) ? (b) : (a))

//Internal use only, this should not be called in any code as it will exit on a failure instead of returning an error code.
#define cudaCheckError() {                                          \
        cudaDeviceSynchronize(); \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }


#define XMP_CHECK_NE(p,q) \
  if((p)==(q)) return xmpErrorInvalidParameter;

#define XMP_SET_DEVICE(h)                        \
  xmpDeviceSetter __device_setter(h->device);    \
  if(cudaSuccess!=cudaPeekAtLastError())          \
    return xmpErrorCuda;

#define XMP_CHECK_CUDA() \
  if(cudaSuccess!=cudaPeekAtLastError()) \
    return xmpErrorCuda;

const uint32_t XMP_EXECUTION_POLICY_MAX_INDICES_ARRAYS = 4;
struct _xmpExecutionPolicy_t {
  size_t scratch_size_limit;
  uint32_t *indices[XMP_EXECUTION_POLICY_MAX_INDICES_ARRAYS];
  uint32_t indices_count[XMP_EXECUTION_POLICY_MAX_INDICES_ARRAYS];
  xmpAlgorithm_t algorithm;
  _xmpExecutionPolicy_t() {
    scratch_size_limit=0;
    for(int i=0;i<XMP_EXECUTION_POLICY_MAX_INDICES_ARRAYS; i++)
      indices[i]=NULL;
    algorithm=xmpAlgorithmDefault;
  }
};

const _xmpExecutionPolicy_t xmpDefaultExecutionPolicy;

struct _xmpHandle_t {
  cudaStream_t stream;
  int32_t device;
  xmpAllocFunc ha, da;
  xmpFreeFunc hf, df;
  size_t scratchSize;
  void* scratch;
  size_t memorySize;
  size_t tmpOutSize;  //some operators need this to produce inplace output
  void* tmpOut;
  uint32_t arch;
  uint32_t smCount;
  xmpExecutionPolicy_t policy;
};


typedef enum {
  xmpOperationAdd,
  xmpOperationSub,
  xmpOperationSqr,
  xmpOperationMul,
  xmpOperationDiv,
  xmpOperationMod,
  xmpOperationDivMod,
  xmpOperationAR,
  xmpOperationPowm,
  xmpOperationCmp,
  xmpOperationIor,
  xmpOperationAnd,
  xmpOperationXor,
  xmpOperationNot,
  xmpOperationPopc
} xmpOperation_t;

typedef struct _xmpPrecision_t {
  int32_t precisionCount;
  int32_t precision1;
  int32_t precision2;
} xmpPrecision_t;

typedef void (*xmpLauncher_t)(xmpHandle_t *handle, int32_t geometry, int32_t threads, int32_t sharedMem, void *arguments, int32_t count);

typedef struct _xmpDispatchEntry_t {
  xmpOperation_t  operation;
  int32_t         computeCapability;
  int32_t         algorithmIndex;
  xmpPrecision_t  precision;
  int32_t         geometry;
  int32_t         threads;
  xmpLauncher_t   launcher;
  void           *kernel;
} xmpDispatchEntry;

inline const char* getAlgorithmString(xmpAlgorithm_t algorithm) {
    switch(algorithm) {
      case xmpAlgorithmDefault:
        return "Default";
      case xmpAlgorithmDigitMP:
        return "DigitMP";
      case xmpAlgorithmRegMP:
        return "RegMP";
      case xmpAlgorithmDistributedMP:
        return "DistributedMP";
      default:
        return "Unknown";
    }
}
inline void printPolicy(xmpExecutionPolicy_t policy) {
  printf("Execution Policy:\n");
  printf("    Scratch size limit: %ld\n", policy->scratch_size_limit);
  for(int i=0;i<XMP_EXECUTION_POLICY_MAX_INDICES_ARRAYS;i++) {
    if(policy->indices[i]!=NULL) {
      printf("    Indices[%d]: %p:%d\n", i, policy->indices[i],policy->indices_count[i]);
    } else {
      printf("    Indices[%d]: NULL\n", i );
    }
    printf("    Algorithm: %s", getAlgorithmString(policy->algorithm));
  }
}
__global__ void printWordsStrided_kernel(xmpLimb_t* data, int limbs, int stride, int count);
void printWordsStrided(xmpLimb_t* data, int limbs, int stride, int count);
__global__ void printWordsCompact_kernel(xmpLimb_t* data, int limbs, int count);

void printWordsCompact(xmpLimb_t* data, int limbs,  int count);

__global__ void xmpC2S_kernel(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out);
__global__ void xmpS2C_kernel(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out);

xmpError_t xmpSetNecessaryScratchSize(xmpHandle_t handle, size_t bytes);
xmpError_t xmpSetNecessaryOutSize(xmpHandle_t handle, size_t bytes);

inline void xmpC2S(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out, cudaStream_t stream) {
  dim3 threads, blocks;

  //target 128 threads
  threads.x=MIN(32,N);
  threads.y=MIN(DIV_ROUND_UP(128,threads.x),limbs);

  blocks.x=DIV_ROUND_UP(N,threads.x);
  blocks.y=DIV_ROUND_UP(limbs,threads.y);

  //convert from climbs to slimbs
  xmpC2S_kernel<<<blocks,threads,0,stream>>>(N,limbs,stride,in,out);
}

inline void xmpS2C(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out, cudaStream_t stream) {
  dim3 threads, blocks;

  //target 128 threads
  threads.x=MIN(32,limbs);
  threads.y=MIN(DIV_ROUND_UP(128,threads.x),N);

  blocks.x=DIV_ROUND_UP(limbs,threads.x);
  blocks.y=DIV_ROUND_UP(N,threads.y);

  //convert from climbs to slimbs
  xmpS2C_kernel<<<blocks,threads,0,stream>>>(N,limbs,stride,in,out);
}

enum xmpFormat_t { xmpFormatNone, xmpFormatCompact, xmpFormatStrided, xmpFormatBoth };
struct _xmpIntegers_t {
  xmpLimb_t *climbs;
  xmpLimb_t *slimbs;
  uint32_t stride;
  uint32_t nlimbs;
  uint32_t precision;
  uint32_t count;
  int32_t device;
  xmpFormat_t format;

  inline void setFormat(xmpFormat_t format) {
    this->format=format;
  }

  inline xmpFormat_t getFormat() {
    return format;
  }

  inline xmpError_t requireFormat(xmpHandle_t handle, xmpFormat_t format) {
    if(format==this->format || format==xmpFormatNone)
      return xmpErrorSuccess;

    if(this->format==xmpFormatBoth && (format==xmpFormatCompact || format==xmpFormatStrided)) {
      return xmpErrorSuccess;
    }

    if(this->format==xmpFormatCompact && (format==xmpFormatBoth || format==xmpFormatStrided)) {
      //convert from compact to strided
      xmpC2S(count,nlimbs,stride,climbs,slimbs,handle->stream);

      this->format=xmpFormatBoth;
      return xmpErrorSuccess;
    }

    if(this->format==xmpFormatStrided && (format==xmpFormatBoth || format==xmpFormatCompact)) {
      //convert from strided to compact
      xmpS2C(count,nlimbs,stride,slimbs,climbs,handle->stream);

      this->format=xmpFormatBoth;
      return xmpErrorSuccess;
    }

    return xmpErrorInvalidFormat;
  }
};

class xmpDeviceSetter {
  public:
    inline xmpDeviceSetter(int32_t device) {
      cudaGetDevice(&old_device);
      cudaSetDevice(device);
    }
    inline ~xmpDeviceSetter() {
      cudaSetDevice(old_device);
    }
  private:
    int32_t old_device;
    xmpDeviceSetter();
};

