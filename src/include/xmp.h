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
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime_api.h>

#if  defined(_WIN32) || defined(_WIN64)
#define XMPAPI __declspec(dllexport) __stdcall
#else
#define XMPAPI __attribute__ ((visibility ("default")))
#endif


#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
  xmpErrorSuccess,
  xmpErrorInvalidDispatchTable,
  xmpErrorInvalidParameter,
  xmpErrorInvalidMalloc,
  xmpErrorInvalidCudaMalloc,
  xmpErrorInvalidCount,
  xmpErrorInvalidDevice,
  xmpErrorInvalidFormat,
  xmpErrorInvalidPrecision,
  xmpErrorIncreaseScratchLimit,
  xmpErrorUnsupported,
  xmpErrorUnspecified,
  xmpErrorCuda,
} xmpError_t;


typedef enum {
  xmpAlgorithm,
  xmpScratchSizeLimit
} xmpExecutionPolicyParam_t;

typedef enum {
  xmpAlgorithmDefault,
  xmpAlgorithmRegMP,
  xmpAlgorithmDigitMP,
  xmpAlgorithmDistributedMP,
} xmpAlgorithm_t;

union xmpExecutionPolicyValue_t {
  xmpAlgorithm_t algorithm;
  size_t         size;
  xmpExecutionPolicyValue_t(xmpAlgorithm_t a) { algorithm=a; }
  xmpExecutionPolicyValue_t(size_t s) { size=s; }
};

inline const char* xmpGetErrorString (xmpError_t error) {
  switch (error) {
    case xmpErrorSuccess:
      return "xmpErrorSuccess";
    case xmpErrorInvalidDispatchTable:
      return "xmpErrorInvalidDispatchTable";
    case xmpErrorInvalidParameter:
      return "xmpErrorInvalidParameter";
    case xmpErrorInvalidMalloc:
      return "xmpErrorInvalidMalloc";
    case xmpErrorInvalidCudaMalloc:
      return "xmpErrorInvalidCudaMalloc";
    case xmpErrorInvalidCount:
      return "xmpErrorInvalidCount";
    case xmpErrorInvalidDevice:
      return "xmpErrorInvalidDevice";
    case xmpErrorInvalidFormat:
      return "xmpErrorInvalidFormat";
    case xmpErrorInvalidPrecision:
      return "xmpErrorPrecision";
    case xmpErrorIncreaseScratchLimit:
      return "xmpIncreaseScratchLimit";
    case xmpErrorUnsupported:
      return "xmpErrorUnsupported";
    case xmpErrorUnspecified:
      return "xmpErrorUnspecified";
    case xmpErrorCuda:
      return "xmpErrorCuda";
    default:
      return "xmpErrorUnknown";

  };
}


typedef struct _xmpHandle_t * xmpHandle_t;
typedef struct _xmpIntegers_t * xmpIntegers_t;
typedef struct _xmpExecutionPolicy_t * xmpExecutionPolicy_t;

typedef void* (*xmpAllocFunc)(size_t);
typedef void (*xmpFreeFunc)(void*);

//create xmp_handle
xmpError_t XMPAPI xmpHandleCreate(xmpHandle_t *handle);
xmpError_t XMPAPI xmpHandleCreateWithMemoryFunctions(xmpHandle_t *handle,xmpAllocFunc ha, xmpFreeFunc hf, xmpAllocFunc da, xmpFreeFunc df);

//destroy xmp_handle
xmpError_t XMPAPI xmpHandleDestroy(xmpHandle_t handle);

xmpError_t XMPAPI xmpHandleSetStream(xmpHandle_t handle, cudaStream_t stream);
xmpError_t XMPAPI xmpHandleGetStream(xmpHandle_t handle, cudaStream_t *stream);

//set the current execution plicy for the handle
xmpError_t XMPAPI xmpHandleSetExecutionPolicy(xmpHandle_t handle, xmpExecutionPolicy_t policy);

//get memory functions
xmpError_t XMPAPI xmpHandleGetMemoryFunctions(xmpHandle_t handle, xmpAllocFunc *ha, xmpFreeFunc *hf, xmpAllocFunc *da, xmpFreeFunc *df);

//set stream for Cuda operations
xmpError_t XMPAPI xmpHandleSetCudaStream(xmpHandle_t handle, cudaStream_t stream);

//get the stream associated with the handle
xmpError_t XMPAPI xmpHandleGetCudaStream(xmpHandle_t handle, cudaStream_t *stream);

//get the device associated with the handle
xmpError_t XMPAPI xmpHandleGetDevice(xmpHandle_t handle, int32_t *device);

//creates an execution policy
xmpError_t XMPAPI xmpExecutionPolicyCreate(xmpHandle_t handle, xmpExecutionPolicy_t *policy);
//destroys an execution policy
xmpError_t XMPAPI xmpExecutionPolicyDestroy(xmpHandle_t handle, xmpExecutionPolicy_t policy);

//set dynamic indices
xmpError_t XMPAPI xmpExecutionPolicySetIndices(xmpHandle_t handle, xmpExecutionPolicy_t policy, uint32_t which_integer, uint32_t *indices, uint32_t count);
xmpError_t XMPAPI xmpExecutionPolicySetIndicesAsync(xmpHandle_t handle, xmpExecutionPolicy_t policy, uint32_t which_integer, uint32_t *indices, uint32_t count);
xmpError_t XMPAPI xmpExecutionPolicySetParameter(xmpHandle_t handle, xmpExecutionPolicy_t policy, xmpExecutionPolicyParam_t param, xmpExecutionPolicyValue_t val);
xmpError_t XMPAPI xmpExecutionPolicyGetParameter(xmpHandle_t handle, xmpExecutionPolicy_t policy, xmpExecutionPolicyParam_t param, xmpExecutionPolicyValue_t &val);


//allocate array of integers
xmpError_t XMPAPI xmpIntegersCreate(xmpHandle_t handle, xmpIntegers_t *x, uint32_t precision, uint32_t count);
//free an array of integers
xmpError_t XMPAPI xmpIntegersDestroy(xmpHandle_t handle, xmpIntegers_t x);

//get the precision of integers in x
xmpError_t XMPAPI xmpIntegersGetPrecision(xmpHandle_t handle, xmpIntegers_t x, uint32_t *precision);
//get the number of integers in x
xmpError_t XMPAPI xmpIntegersGetCount(xmpHandle_t handle, xmpIntegers_t x, uint32_t *count);

//import integers from in into out
xmpError_t XMPAPI xmpIntegersImport(xmpHandle_t handle, xmpIntegers_t out, uint32_t words, int32_t order, size_t size, int32_t endian, uint32_t nails, void* in, uint32_t count);
xmpError_t XMPAPI xmpIntegersImportAsync(xmpHandle_t handle, xmpIntegers_t out, uint32_t words, int32_t order, size_t size, int32_t endian, uint32_t nails, void* in, uint32_t count);
//export integers from in into out
xmpError_t XMPAPI xmpIntegersExport(xmpHandle_t handle, void* out, uint32_t *words, int32_t order, size_t size, int32_t endian, uint32_t nails, xmpIntegers_t in, uint32_t count);
xmpError_t XMPAPI xmpIntegersExportAsync(xmpHandle_t handle, void* out, uint32_t *words, int32_t order, size_t size, int32_t endian, uint32_t nails, xmpIntegers_t in, uint32_t count);
//copy count integers of size bytes from in into out
xmpError_t XMPAPI xmpIntegersSet(xmpHandle_t handle, xmpIntegers_t out, xmpIntegers_t in, uint32_t count);
xmpError_t XMPAPI xmpIntegersSetAsync(xmpHandle_t handle, xmpIntegers_t out, xmpIntegers_t in, uint32_t count);

/******************************
 * Math APIs
 * ***************************/

//computes s=a+b
xmpError_t XMPAPI xmpIntegersAdd(xmpHandle_t handle, xmpIntegers_t s, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
xmpError_t XMPAPI xmpIntegersAddAsync(xmpHandle_t handle, xmpIntegers_t s, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
//computes d=a-b
xmpError_t XMPAPI xmpIntegersSub(xmpHandle_t handle, xmpIntegers_t d, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
xmpError_t XMPAPI xmpIntegersSubAsync(xmpHandle_t handle, xmpIntegers_t d, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
//computes p=a*b
xmpError_t XMPAPI xmpIntegersMul(xmpHandle_t handle, xmpIntegers_t p, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
xmpError_t XMPAPI xmpIntegersMulAsync(xmpHandle_t handle, xmpIntegers_t p, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
//computes q=floor(a/b)
xmpError_t XMPAPI xmpIntegersDiv(xmpHandle_t handle, xmpIntegers_t q, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
xmpError_t XMPAPI xmpIntegersDivAsync(xmpHandle_t handle, xmpIntegers_t q, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
//computes m=a%b
xmpError_t XMPAPI xmpIntegersMod(xmpHandle_t handle, xmpIntegers_t m, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
xmpError_t XMPAPI xmpIntegersModAsync(xmpHandle_t handle, xmpIntegers_t m, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
//computes q=floor(a/b) and m=a%b
xmpError_t XMPAPI xmpIntegersDivMod(xmpHandle_t handle, xmpIntegers_t q, xmpIntegers_t m, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
xmpError_t XMPAPI xmpIntegersDivModAsync(xmpHandle_t handle, xmpIntegers_t q, xmpIntegers_t m, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
//computes out=base^exp % mod
xmpError_t XMPAPI xmpIntegersPowm(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t base, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t count);
xmpError_t XMPAPI xmpIntegersPowmAsync(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t base, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t count);

/******************************
 * Bitwise APIs
 * ***************************/

//computes c=shift(a,shift)
xmpError_t XMPAPI xmpIntegersShf(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const int32_t* shift, const uint32_t shift_count, uint32_t count);
xmpError_t XMPAPI xmpIntegersShfAsync(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const int32_t* shift, const uint32_t shift_count, uint32_t count);

//computes c=a|b
xmpError_t XMPAPI xmpIntegersIor(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
xmpError_t XMPAPI xmpIntegersIorAsync(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
//computes c=a&b
xmpError_t XMPAPI xmpIntegersAnd(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
xmpError_t XMPAPI xmpIntegersAndAsync(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
//computes c=a^b
xmpError_t XMPAPI xmpIntegersXor(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
xmpError_t XMPAPI xmpIntegersXorAsync(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
//computes c=!a
xmpError_t XMPAPI xmpIntegersNot(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, uint32_t count);
xmpError_t XMPAPI xmpIntegersNotAsync(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, uint32_t count);
//compute c=popc(a)
xmpError_t XMPAPI xmpIntegersPopc(xmpHandle_t handle, uint32_t *c, const xmpIntegers_t a, uint32_t count);
xmpError_t XMPAPI xmpIntegersPopcAsync(xmpHandle_t handle, uint32_t *c, const xmpIntegers_t a, uint32_t count);
//compute c=CMP(a,b),  -1 a is smaller, 0 equal, +1 a is larger
xmpError_t XMPAPI xmpIntegersCmp(xmpHandle_t handle, int32_t *c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);
xmpError_t XMPAPI xmpIntegersCmpAsync(xmpHandle_t handle, int32_t *c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count);



#ifdef __cplusplus
}
#endif
