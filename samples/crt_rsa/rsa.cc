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
  int hbits=bits/2;
  double start,end;
  
  uint32_t limbs=bits/8/sizeof(uint32_t);
  uint32_t hlimbs=(limbs+1)/2;  //+1 to round up
  
  size_t bytes=limbs*sizeof(uint32_t);
  size_t hbytes=bytes/2;

  xmpHandle_t handle;
  
  //public key = (n, e)
  //private key = (p, q, dp, dq, (d=p*q)

  //p = prime exponent 1
  //q = prime exponenent 2
  //dp = d mod (p-1)
  //dq = d mod (q-1)
  //cp = q*modInv(q,p)
  //cq = p*modInv(p,q)

  //n = modulus = p*q
  //e = public exponent
  
  //m = message to be encrypted
  //c = encrypted message
  //r = decrypted results
 
  //cm,mp,mq,sq,sp  are all temporaries

  xmpIntegers_t n, e;       //public key
  xmpIntegers_t p, q, d;    //private key
  xmpIntegers_t m, c, r;    //messages
  xmpIntegers_t  dp, dq, cp, cq;

  //these are needed for the Chinese Remainder Theorem
  xmpIntegers_t cm, mp, mq, sq, sp;         //temperaries

  //host data for initialization
  uint32_t *h_n, *h_e;       //public key
  uint32_t *h_p, *h_q;       //private key
  uint32_t *h_m;             //message to be encrypted
  
  //CRT inputs
  uint32_t *h_cp, *h_cq, *h_dp, *h_dq;

  int32_t *results;          //array for validation

  //using calloc so all allocated memory is zero
  h_n=(uint32_t*)calloc(1,bytes);
  h_e=(uint32_t*)calloc(1,4);           //public exponent is usually very small

  h_p=(uint32_t*)calloc(1,hbytes);
  h_q=(uint32_t*)calloc(1,hbytes);
  
  h_m=(uint32_t*)calloc(count,bytes);

  h_dp=(uint32_t*)calloc(1,hbytes);
  h_dq=(uint32_t*)calloc(1,hbytes);
  h_cq=(uint32_t*)calloc(1,bytes);
  h_cp=(uint32_t*)calloc(1,bytes);

  results=(int32_t*)calloc(count,sizeof(int32_t));

  //allocate handle
  XMP_CHECK_ERROR(xmpHandleCreate(&handle));

  //allocate integers
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&n,bits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&e,32,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&p,hbits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&q,hbits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&d,bits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&cp,bits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&cq,bits,1));
  
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&m,bits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&c,bits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&r,bits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&sp,bits+hbits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&sq,bits+hbits,count));

  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&mp,hbits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&mq,hbits,count));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&dp,hbits,1));
  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&dq,hbits,1));

  XMP_CHECK_ERROR(xmpIntegersCreate(handle,&cm,hbits,count));

  //hard coding with small primes
  //for a real application you should use much larger keys

  //these should be precomputed

  //public key
  h_n[0]=17460671;
  h_e[0]=65537;

  //private key
  h_p[0]=4931;
  h_q[0]=3541;

  //CRT inputs
  h_dp[0]=1063;
  h_dq[0]=113;
  h_cp[0]=15212136;
  h_cq[0]=2248536;

  //messages to encrypt
  for(int i=0;i<count;i++) {
    h_m[i*limbs]=rand32()%h_n[0];         //need to restrict message to be smaller than n
  }

  //import into xmp
  XMP_CHECK_ERROR(xmpIntegersImport(handle,n,limbs,-1,sizeof(uint32_t),0,0,h_n,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,e,1,-1,sizeof(uint32_t),0,0,h_e,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,p,hlimbs,-1,sizeof(uint32_t),0,0,h_p,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,q,hlimbs,-1,sizeof(uint32_t),0,0,h_q,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,m,limbs,-1,sizeof(uint32_t),0,0,h_m,count));

  XMP_CHECK_ERROR(xmpIntegersImport(handle,dp,hlimbs,-1,sizeof(uint32_t),0,0,h_dp,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,dq,hlimbs,-1,sizeof(uint32_t),0,0,h_dq,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,cp,limbs,-1,sizeof(uint32_t),0,0,h_cp,1));
  XMP_CHECK_ERROR(xmpIntegersImport(handle,cq,limbs,-1,sizeof(uint32_t),0,0,h_cq,1));

  //encrypt using Pure RSA  (warning without proper padding this is insecure)
  //c=m ^ e mod n
  start=wallclock();
  XMP_CHECK_ERROR(xmpIntegersPowm(handle,c,m,e,n,count));
  end=wallclock();
  printf("Encryption time: %lg, %d bit throughput: %lg encryptions/second\n", end-start, bits, count/(end-start));

  //decrypt with CRT
  start=wallclock();
  //mp=(c mod p)^dp mod p
  XMP_CHECK_ERROR(xmpIntegersMod(handle,cm,c,p,count));       //cm = c mod p
  XMP_CHECK_ERROR(xmpIntegersPowm(handle,mp,cm,dp,p,count));  //mp = cm^dp mod p

  //mq=(c mod q)^dq mod q
  XMP_CHECK_ERROR(xmpIntegersMod(handle,cm,c,q,count));       //cm = c mod q
  XMP_CHECK_ERROR(xmpIntegersPowm(handle,mq,cm,dq,q,count));  //mq = cm^dq mod q

  XMP_CHECK_ERROR(xmpIntegersMul(handle,sp,mp,cp,count));     //sp = mp * cp
  XMP_CHECK_ERROR(xmpIntegersMul(handle,sq,mq,cq,count));     //sq = mq * cq

  XMP_CHECK_ERROR(xmpIntegersAdd(handle,sp,sp,sq,count));      //r=sp+sq
  
  XMP_CHECK_ERROR(xmpIntegersMod(handle,r,sp,n,count));        //r=r mod n
  end=wallclock();
  printf("Decrytpion time: %lg, %d bit throughput: %lg decryptions/second\n", end-start, bits, count/(end-start));

  //validate
  //r and m should be the same
  XMP_CHECK_ERROR(xmpIntegersCmp(handle,results,m,r,count));

  printf("Validating results...\n");
  for(int i=0;i<count;i++) {
    if(results[i]!=0) {
      printf("  Error at index %d\n", i);
      exit(1);
    }
  }

  //free integers
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,n));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,e));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,p));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,q));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,m));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,c));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,cm));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,sp));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,sq));
  

  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,dp));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,dq));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,cp));
  XMP_CHECK_ERROR(xmpIntegersDestroy(handle,cq));

  //free handle
  XMP_CHECK_ERROR(xmpHandleDestroy(handle));

  free(h_n);
  free(h_e);
  free(h_p);
  free(h_q);


  printf("CRT RSA executed successfully\n");
  return 0;
}
