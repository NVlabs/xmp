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
  
  //public key = (n, e)
  //private key = (d)
  //n = modulus
  //e = public exponent
  //d = private exponent
  //c = encrypted message
  //m = message to be encrypted
  //r = decrypted results

  xmpIntegers_t n, e;       //public key
  xmpIntegers_t d;          //private key
  xmpIntegers_t m, c, r;    //messages

  //host data for initialization
  uint32_t *h_n, *h_e;       //public key
  uint32_t *h_d;             //private key
  uint32_t *h_m;             //message to be encrypted

  int32_t *res;            //array for validation

  //using calloc so all allocated memory is zero
  h_n=(uint32_t*)calloc(1,bytes);
  h_e=(uint32_t*)calloc(1,4);                                 //public exponent is usually small

  h_d=(uint32_t*)calloc(1,bytes);
  
  h_m=(uint32_t*)calloc(count,bytes);

  res=(int32_t*)calloc(count,sizeof(int32_t));

  //allocate handle
  XMP_CHECK_ERROR(xmpHandleCreate(&handle));

  //allocate integers
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&n,bits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&e,32,1));         
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&d,bits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&m,bits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&c,bits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&r,bits,count));

  //hard coding with small primes
  //for a real application you should use much larger keys

  //these should be precomputed

  //public key
  h_n[0]=17460671;
  h_e[0]=65537;

  //private key
  h_d[0]=16156673;

  //messages to encrypt
  for(int i=0;i<count;i++) {
    h_m[i*limbs]=rand32()%h_n[0];         //need to restrict message to be smaller than n
  }

  //import into xmp
  XMP_CHECK_ERROR(xmpIntegersImport(handle,n,limbs,-1,sizeof(uint32_t),0,0,h_n,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,e,1,-1,sizeof(uint32_t),0,0,h_e,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,d,limbs,-1,sizeof(uint32_t),0,0,h_d,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,m,limbs,-1,sizeof(uint32_t),0,0,h_m,count));

  //encrypt using Pure RSA  (warning without proper padding this is insecure)
  //c=m ^ e mod n
  start=wallclock();
  XMP_CHECK_ERROR(xmpIntegersPowm(handle,c,m,e,n,count));
  end=wallclock();
  printf("Encryption time: %lg, %d bit throughput: %lg encryptions/second\n", end-start, bits, count/(end-start));
 
  //decrypt
  //r=c^d mod n
  start=wallclock();
  XMP_CHECK_ERROR(xmpIntegersPowm(handle,r,c,d,n,count));
  end=wallclock();
  printf("Decryption time: %lg, %d bit throughput: %lg decryptions/second\n", end-start, bits, count/(end-start));

  //validate
  //r and m should be the same
  XMP_CHECK_ERROR(xmpIntegersCmp(handle,res,m,r,count));

  printf("Validating results...\n");
  for(int i=0;i<count;i++) {
    if(res[i]!=0) {
      printf("  Error at index %d\n", i);
      exit(1);
    }
  }

  //free integers
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,n));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,e));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,d));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,m));

  //free handle
  XMP_CHECK_ERROR(xmpHandleDestroy(handle));

  free(h_n);
  free(h_e);
  free(h_d);

  printf("simple RSA executed successfully\n");
  return 0;
}
