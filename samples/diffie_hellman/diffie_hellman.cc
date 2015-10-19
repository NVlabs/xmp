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
#ifndef _WIN32
#include <sys/time.h>
#endif

static double wallclock(void)
{
  double t;
#ifdef _WIN32
  t = clock()/(double)CLOCKS_PER_SEC;
#else
  struct timeval tv;
  struct timezone tz;

  gettimeofday(&tv, &tz);

  t = (double)tv.tv_sec;
  t += ((double)tv.tv_usec)/1000000.0;
#endif
  return t;
}


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

uint32_t rand32() {
  uint32_t lo=rand() & 0xffff;
  uint32_t hi=rand() & 0xffff;

  return (hi<<16)|lo;
}


int main() {
  int count=100000;
  int bits=2048;

  double start,end;

  uint32_t limbs=bits/8/sizeof(uint32_t);
  size_t bytes=limbs*sizeof(uint32_t);

  xmpHandle_t handle;


  //p = public modulus
  //g = public base

  //a = secret integer 1
  //b = secret integer 2

  //ap = public message 1
  //bp = public message 2

  //ar = shared secret key a
  //br = shared secret key b

  xmpIntegers_t p, g;      //public key
  xmpIntegers_t a, b;      //secret integers
  xmpIntegers_t ap, bp;    //unencrypted messages
  xmpIntegers_t ar, br;    //shared secret key

  //host data for initialization
  uint32_t *h_p, *h_g;       //public key
  uint32_t *h_a, *h_b;       //private key

  int32_t *res;            //array for validation

  //using calloc so all allocated memory is zero
  h_p=(uint32_t*)calloc(1,bytes);
  h_g=(uint32_t*)calloc(1,bytes);   
  h_a=(uint32_t*)calloc(1,bytes);
  h_b=(uint32_t*)calloc(1,bytes);

  res=(int32_t*)calloc(count,sizeof(int32_t));

  //allocate handle
  XMP_CHECK_ERROR(xmpHandleCreate(&handle));

  //allocate integers
  //Only using a count of 1 for a,g,a and b to speed up random number generation
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&p,bits,1));  
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&g,bits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&a,bits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&b,bits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&ap,bits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&bp,bits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&ar,bits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&br,bits,count));

  //randomly generate base, modulus, a, and b
  for(int j=0;j<limbs;j++) {
    h_p[j]=rand32();
    h_g[j]=rand32();
    h_a[j]=rand32();
    h_b[j]=rand32();
  }
  //ensure modulus is odd
  h_p[0]|=0x1;

  //import into xmp
  XMP_CHECK_ERROR(xmpIntegersImport(handle,p,limbs,-1,sizeof(uint32_t),0,0,h_p,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,g,limbs,-1,sizeof(uint32_t),0,0,h_g,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,a,limbs,-1,sizeof(uint32_t),0,0,h_a,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,b,limbs,-1,sizeof(uint32_t),0,0,h_b,1));

  start=wallclock();
  //compute both sides here.  Normally you would only do half of this
  //generate public messages
  XMP_CHECK_ERROR(xmpIntegersPowm(handle,ap,g,a,p,count));  //Message 1
  XMP_CHECK_ERROR(xmpIntegersPowm(handle,bp,g,b,p,count));  //Message 2

  //Exchange public messages uncrypted here

  //compute shared key
  XMP_CHECK_ERROR(xmpIntegersPowm(handle,ar,bp,a,p,count));  //Compute shared secret from bp and a
  XMP_CHECK_ERROR(xmpIntegersPowm(handle,br,ap,b,p,count));  //Compute shared secret from ap and b

  end=wallclock();
  //Divide by 2 because we did both parts on the same system.  
  printf("Exchange time: %lg, %d bit throughput: %lg exchanges/second\n", (end-start)/2, bits, count/(end-start)*2);

  //validate
  //ar and br should be the same
  XMP_CHECK_ERROR(xmpIntegersCmp(handle,res,br,ar,count));

  printf("Validating results...\n");
  for(int i=0;i<count;i++) {
    if(res[i]!=0) {
      printf("  Error at index %d\n", i);
      exit(1);
    }
  }

  //free integers
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,p));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,g));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,a));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,b));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,ap));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,bp));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,ar));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,br));

  //free handle
  XMP_CHECK_ERROR(xmpHandleDestroy(handle));

  free(h_p);
  free(h_g);
  free(h_a);
  free(h_b);

  printf("Diffie-Hellman executed successfully\n");
  return 0;
}
