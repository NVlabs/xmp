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
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define XMP_CHECK_ERROR(fun) \
{                             \
  xmpError_t error=fun;     \
  if(error!=xmpErrorSuccess){ \
    if(error==xmpErrorCuda)   \
      printf("CUDA Error %s, %s:%d\n",cudaGetErrorString(cudaGetLastError()),__FILE__,__LINE__); \
    else  \
      printf("XMP Error %s, %s:%d\n",xmpGetErrorString(error),__FILE__,__LINE__); \
    exit(EXIT_FAILURE); \
  } \
}


int main() {
  uint32_t i,w;
  uint32_t N=15*1024*4;
  int bits=1024;
  xmpIntegers_t base, mod, exp, out;
  uint32_t *b,*m,*e,*o;
  uint32_t limbs=bits/8/sizeof(uint32_t);
  
  size_t bytes=N*bits/8;
  b=(uint32_t*)malloc(bytes);
  o=(uint32_t*)malloc(bytes);
  m=(uint32_t*)malloc(bits/8);
  e=(uint32_t*)malloc(bits/8);

  xmpHandle_t handle;

  cudaSetDevice(0);
  //allocate handle
  XMP_CHECK_ERROR(xmpHandleCreate(&handle));

  //allocate integers
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&base,bits,N));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&out,bits,N));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&exp,bits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&mod,bits,1));

  //initialize base, exp, and mod
  for(i=0;i<N;i++) {
    for(w=0;w<limbs;w++) {
      b[i*limbs+w]=rand();
    }
  }

  for(w=0;w<limbs;w++) {
    m[w]=rand();
    e[w]=rand();
  }
  //make sure modulus is odd
  m[0]|=1;

  //import 
  XMP_CHECK_ERROR(xmpIntegersImport(handle,base,limbs,-1,sizeof(uint32_t),0,0,b,N));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,exp,limbs,-1,sizeof(uint32_t),0,0,e,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,mod,limbs,-1,sizeof(uint32_t),0,0,m,1));

  //call powm
  XMP_CHECK_ERROR(xmpIntegersPowm(handle,out,base,exp,mod,N));
 
  //export
  XMP_CHECK_ERROR(xmpIntegersExport(handle,o,&limbs,-1,sizeof(uint32_t),0,0,out,N));
  
  //use results here

  //free integers
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,base));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,out));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,exp));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,mod));

  //free handle
  XMP_CHECK_ERROR(xmpHandleDestroy(handle));

  free(b);
  free(o);
  free(m);
  free(e);

  printf("sample01 executed sucessfully\n");
  return 0;
}
