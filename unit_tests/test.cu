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
#include "../src/include/xmp_internal.h"
#include "gtest/gtest.h"
#include <gmp.h>
#include <sstream>
#include <algorithm>

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}
uint32_t rand32() {
  uint32_t lo=rand() & 0xffff; 
  uint32_t hi=rand() & 0xffff; 

  return (hi<<16)|lo; 
}

::std::ostream& operator<<(::std::ostream& os, xmpError_t error) {
    return os << xmpGetErrorString(error);
}

TEST(handleTests,handleCreateDestroy)
{
  xmpHandle_t handle;
  EXPECT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  EXPECT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

void *deviceMalloc(size_t bytes) {
  void* retval;
  if(cudaSuccess!=cudaMalloc(&retval,bytes))
    return 0;
  return retval;
}
void deviceFree(void *ptr) {
  cudaFree(ptr);
}


TEST(handleTests,handleSetGetStream) {
  xmpHandle_t handle;
  xmpHandleCreateWithMemoryFunctions(&handle,malloc,free,deviceMalloc,deviceFree);
  
  cudaStream_t stream;
  cudaStream_t stream2;
  cudaStreamCreate(&stream);
  
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetStream(handle,stream)); 
  ASSERT_EQ(xmpErrorSuccess,xmpHandleGetStream(handle,&stream2));
  ASSERT_EQ(stream,stream2);
  
  xmpHandleDestroy(handle);
  cudaStreamDestroy(stream);
}

TEST(handleTests,handleGetDevice) {
  xmpHandle_t handle;
  xmpHandleCreate(&handle);
  
  int device=-1;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleGetDevice(handle,&device)); 
  ASSERT_EQ(device,0);
  
  xmpHandleDestroy(handle);
}

TEST(handleTests,handleCreateDestroyWithMemoryFunctions) {
  xmpHandle_t handle;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreateWithMemoryFunctions(&handle,malloc,free,deviceMalloc,deviceFree));

  xmpAllocFunc ha,da;
  xmpFreeFunc hf,df;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleGetMemoryFunctions(handle,&ha,&hf,&da,&df));

  EXPECT_EQ(ha,(xmpAllocFunc)malloc);
  EXPECT_EQ(da,(xmpAllocFunc)deviceMalloc);
  EXPECT_EQ(hf,(xmpFreeFunc)free);
  EXPECT_EQ(df,(xmpFreeFunc)deviceFree);

  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST(errorTests,nullTests) {
  xmpHandle_t handle;
  xmpIntegers_t x;
  EXPECT_NE(xmpErrorSuccess,xmpHandleCreate(NULL));
  EXPECT_NE(xmpErrorSuccess,xmpHandleCreateWithMemoryFunctions(NULL,NULL,NULL,NULL,NULL));

  xmpHandleCreate(&handle);

  EXPECT_NE(xmpErrorSuccess,xmpHandleGetStream(handle,NULL));
  EXPECT_NE(xmpErrorSuccess,xmpHandleGetDevice(handle,NULL));

  EXPECT_NE(xmpErrorSuccess,xmpIntegersCreate(handle,NULL,64,100));

  xmpIntegersCreate(handle,&x,64,100);
  
  EXPECT_NE(xmpErrorSuccess,xmpIntegersGetPrecision(handle,x,NULL));
  EXPECT_NE(xmpErrorSuccess,xmpIntegersGetCount(handle,x,NULL));

  EXPECT_NE(xmpErrorSuccess,xmpIntegersImport(handle,x,1,1,1,1,0,NULL,1));
  EXPECT_NE(xmpErrorSuccess,xmpIntegersExport(handle,NULL,NULL,1,1,1,0,x,1));

  xmpIntegersDestroy(handle,x);
  xmpHandleDestroy(handle);
}

TEST(integerTests,xmpIntegersCreateDestroy) {
  xmpHandle_t handle;
  xmpHandleCreate(&handle);
  
  uint32_t N=1000;
  uint32_t p=8192;
  xmpIntegers_t a;
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&a,p,N));
  EXPECT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,a));
  
  xmpHandleDestroy(handle);
}

TEST(integerTests,xmpIntegersQueryFunctions) {
  xmpHandle_t handle;
  xmpIntegers_t a64x1000, a1024x100, a8192x1;
  xmpHandleCreate(&handle);
 
  xmpIntegersCreate(handle,&a64x1000,64,1000);
  xmpIntegersCreate(handle,&a1024x100,1024,100);
  xmpIntegersCreate(handle,&a8192x1,8192,1);

  uint32_t P;
  uint32_t N;

  EXPECT_EQ(xmpErrorSuccess,xmpIntegersGetPrecision(handle,a64x1000,&P));
  EXPECT_EQ(xmpErrorSuccess,xmpIntegersGetCount(handle,a64x1000,&N));
  EXPECT_EQ(P,64);
  EXPECT_EQ(N,1000);
  
  EXPECT_EQ(xmpErrorSuccess,xmpIntegersGetPrecision(handle,a1024x100,&P));
  EXPECT_EQ(xmpErrorSuccess,xmpIntegersGetCount(handle,a1024x100,&N));
  EXPECT_EQ(P,1024);
  EXPECT_EQ(N,100);

  EXPECT_EQ(xmpErrorSuccess,xmpIntegersGetPrecision(handle,a8192x1,&P));
  EXPECT_EQ(xmpErrorSuccess,xmpIntegersGetCount(handle,a8192x1,&N));
  EXPECT_EQ(P,8192);
  EXPECT_EQ(N,1);

  xmpIntegersDestroy(handle,a64x1000);
  xmpIntegersDestroy(handle,a1024x100);
  xmpIntegersDestroy(handle,a8192x1);
  
  xmpHandleDestroy(handle);
}

TEST(errorTests, xmpIntegerFormatErrorTests) {
  xmpHandle_t handle;
  xmpIntegers_t a;
  uint32_t count=100;
  uint32_t precision=128;
  xmpHandleCreate(&handle);
  xmpIntegersCreate(handle,&a,precision,count);
 
  //data doesn't exist on device yet so this should give an error because the format hasn't been set
  EXPECT_NE(xmpErrorSuccess,a->requireFormat(handle,xmpFormatStrided));
  EXPECT_NE(xmpErrorSuccess,a->requireFormat(handle,xmpFormatCompact));
  EXPECT_NE(xmpErrorSuccess,a->requireFormat(handle,xmpFormatBoth));
 
  xmpIntegersDestroy(handle,a);
  xmpHandleDestroy(handle);
}

TEST(integerTests,xmpIntegerFormatConversionRoutines) {
  xmpHandle_t handle;
  xmpIntegers_t a;
  uint32_t count=8;                           //8 integers
  uint32_t precision=128;                     //4 words each
  xmpHandleCreate(&handle);
  xmpIntegersCreate(handle,&a,precision,count);
 
  xmpLimb_t limbs=precision/8/sizeof(xmpLimb_t);
  size_t cbytes=limbs*count*sizeof(xmpLimb_t);
  size_t sbytes=limbs*a->stride*sizeof(xmpLimb_t);
  xmpLimb_t *h=(xmpLimb_t*)malloc(sbytes);
  uint32_t stride=a->stride;

  EXPECT_GE(stride,count);
  EXPECT_EQ(0,(stride*sizeof(xmpLimb_t))%128);  //verify alignment

  for(int i=0;i<count;i++) {
    for(int j=0;j<limbs;j++) {
      h[i*limbs+j]=i*limbs+j;
    }
  }
  
  xmpIntegersImport(handle,a,limbs,xmpNativeOrder,sizeof(xmpLimb_t),xmpNativeEndian,0,h,count);

  EXPECT_EQ(xmpFormatCompact,a->format);
 
  //verify compact storage is correct
  cudaMemcpy(h,a->climbs,cbytes,cudaMemcpyDeviceToHost);
  for(int i=0;i<count;i++) 
    for(int j=0;j<limbs;j++) 
      EXPECT_EQ(i*limbs+j,h[i*limbs+j]);

  cudaMemset(a->slimbs,0,sbytes);
  EXPECT_EQ(xmpErrorSuccess,a->requireFormat(handle,xmpFormatStrided));
  EXPECT_EQ(a->getFormat(),xmpFormatBoth);

  //verify strided format is correct
  cudaMemcpy(h,a->slimbs,sbytes,cudaMemcpyDeviceToHost);
  for(int i=0;i<count;i++) 
    for(int j=0;j<limbs;j++) 
      EXPECT_EQ(i*limbs+j,h[j*stride+i]);

  //set to compact format
  a->setFormat(xmpFormatCompact);
  cudaMemset(a->slimbs,0,sbytes);
  EXPECT_EQ(xmpFormatCompact,a->format);
 
  //require both
  cudaMemset(a->slimbs,0,sbytes);
  EXPECT_EQ(xmpErrorSuccess,a->requireFormat(handle,xmpFormatBoth));
  EXPECT_EQ(a->getFormat(),xmpFormatBoth);
  
  //verify compact storage is correct
  cudaMemcpy(h,a->climbs,cbytes,cudaMemcpyDeviceToHost);
  for(int i=0;i<count;i++) 
    for(int j=0;j<limbs;j++) 
      EXPECT_EQ(i*limbs+j,h[i*limbs+j]);
  
  //verify strided format is correct
  cudaMemcpy(h,a->slimbs,sbytes,cudaMemcpyDeviceToHost);
  for(int i=0;i<count;i++) 
    for(int j=0;j<limbs;j++) 
      EXPECT_EQ(i*limbs+j,h[j*stride+i]);
  
  //set to strided format
  a->setFormat(xmpFormatStrided);
  EXPECT_EQ(xmpFormatStrided,a->format);
  
  cudaMemset(a->climbs,0,cbytes);
  EXPECT_EQ(xmpErrorSuccess,a->requireFormat(handle,xmpFormatCompact));
  EXPECT_EQ(a->getFormat(),xmpFormatBoth);

  //verify compact storage is correct
  cudaMemcpy(h,a->climbs,cbytes,cudaMemcpyDeviceToHost);
  for(int i=0;i<count;i++) 
    for(int j=0;j<limbs;j++) 
      EXPECT_EQ(i*limbs+j,h[i*limbs+j]);

  //set to strided format
  a->setFormat(xmpFormatStrided);
  EXPECT_EQ(xmpFormatStrided,a->format);
  
  cudaMemset(a->climbs,0,cbytes);
  EXPECT_EQ(xmpErrorSuccess,a->requireFormat(handle,xmpFormatBoth));
  EXPECT_EQ(a->getFormat(),xmpFormatBoth);

  //verify compact storage is correct
  cudaMemcpy(h,a->climbs,cbytes,cudaMemcpyDeviceToHost);
  for(int i=0;i<count;i++) 
    for(int j=0;j<limbs;j++) 
      EXPECT_EQ(i*limbs+j,h[i*limbs+j]);

  //verify strided format is correct
  cudaMemcpy(h,a->slimbs,sbytes,cudaMemcpyDeviceToHost);
  for(int i=0;i<count;i++) 
    for(int j=0;j<limbs;j++) 
      EXPECT_EQ(i*limbs+j,h[j*stride+i]);

  free(h);
  xmpIntegersDestroy(handle,a);
  xmpHandleDestroy(handle);
}

TEST(transferTests,xmpIntegersImportExport) {
  xmpHandle_t handle;
  xmpIntegers_t d_a;
  uint32_t *h_a;
  uint32_t *h_b;
  int N=2;
  int P=64;
  uint32_t limbs=P/8/sizeof(uint32_t);
  uint32_t olimbs;

  xmpHandleCreate(&handle);

  h_a=(uint32_t*)malloc(2*sizeof(uint32_t));
  h_b=(uint32_t*)malloc(2*sizeof(uint32_t));
 
  h_a[0*limbs+0]=0x33221100;
  h_a[0*limbs+1]=0x77665544;
  h_a[1*limbs+0]=0x03020100;
  h_a[1*limbs+1]=0x07060504;

  xmpIntegersCreate(handle,&d_a,P,N);

  //Copy in and out with same parameters, values should be unchanged
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,1,sizeof(uint32_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint32_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,-1,sizeof(uint32_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint32_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,1,sizeof(uint32_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint32_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,-1,sizeof(uint32_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint32_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);

  //Copy in and out with different orders (words should flip orders)
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,1,sizeof(uint32_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint32_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x77665544);  EXPECT_EQ(h_b[0*limbs+1],0x33221100);  EXPECT_EQ(h_b[1*limbs+0],0x07060504);  EXPECT_EQ(h_b[1*limbs+1],0x03020100);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,1,sizeof(uint32_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint32_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x77665544);  EXPECT_EQ(h_b[0*limbs+1],0x33221100);  EXPECT_EQ(h_b[1*limbs+0],0x07060504);  EXPECT_EQ(h_b[1*limbs+1],0x03020100);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,-1,sizeof(uint32_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint32_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x77665544);  EXPECT_EQ(h_b[0*limbs+1],0x33221100);  EXPECT_EQ(h_b[1*limbs+0],0x07060504);  EXPECT_EQ(h_b[1*limbs+1],0x03020100);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,-1,sizeof(uint32_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint32_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x77665544);  EXPECT_EQ(h_b[0*limbs+1],0x33221100);  EXPECT_EQ(h_b[1*limbs+0],0x07060504);  EXPECT_EQ(h_b[1*limbs+1],0x03020100);

  //Copy in and out with different endians (bytes in words should flip orders)
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,-1,sizeof(uint32_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint32_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x00112233);  EXPECT_EQ(h_b[0*limbs+1],0x44556677);  EXPECT_EQ(h_b[1*limbs+0],0x00010203);  EXPECT_EQ(h_b[1*limbs+1],0x04050607);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,-1,sizeof(uint32_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint32_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x00112233);  EXPECT_EQ(h_b[0*limbs+1],0x44556677);  EXPECT_EQ(h_b[1*limbs+0],0x00010203);  EXPECT_EQ(h_b[1*limbs+1],0x04050607);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,1,sizeof(uint32_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint32_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x00112233);  EXPECT_EQ(h_b[0*limbs+1],0x44556677);  EXPECT_EQ(h_b[1*limbs+0],0x00010203);  EXPECT_EQ(h_b[1*limbs+1],0x04050607);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,1,sizeof(uint32_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint32_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x00112233);  EXPECT_EQ(h_b[0*limbs+1],0x44556677);  EXPECT_EQ(h_b[1*limbs+0],0x00010203);  EXPECT_EQ(h_b[1*limbs+1],0x04050607);


  //Copy in and out with different endians and different orders (should flip bytes and words)
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,1,sizeof(uint32_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint32_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,-1,sizeof(uint32_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint32_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,-1,sizeof(uint32_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint32_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,1,sizeof(uint32_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint32_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);


  //repeat above tests with 2 byte words
  //Copy in and out with same parameters, values should be unchanged
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,1,sizeof(uint16_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint16_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,-1,sizeof(uint16_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint16_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,1,sizeof(uint16_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint16_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,-1,sizeof(uint16_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint16_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);

  //Copy in and out with different orders (words should flip orders)
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,1,sizeof(uint16_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint16_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x55447766);  EXPECT_EQ(h_b[0*limbs+1],0x11003322);  EXPECT_EQ(h_b[1*limbs+0],0x05040706);  EXPECT_EQ(h_b[1*limbs+1],0x01000302);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,1,sizeof(uint16_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint16_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x55447766);  EXPECT_EQ(h_b[0*limbs+1],0x11003322);  EXPECT_EQ(h_b[1*limbs+0],0x05040706);  EXPECT_EQ(h_b[1*limbs+1],0x01000302);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,-1,sizeof(uint16_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint16_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x55447766);  EXPECT_EQ(h_b[0*limbs+1],0x11003322);  EXPECT_EQ(h_b[1*limbs+0],0x05040706);  EXPECT_EQ(h_b[1*limbs+1],0x01000302);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,-1,sizeof(uint16_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint16_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x55447766);  EXPECT_EQ(h_b[0*limbs+1],0x11003322);  EXPECT_EQ(h_b[1*limbs+0],0x05040706);  EXPECT_EQ(h_b[1*limbs+1],0x01000302);
  
  //Copy in and out with different endians (bytes in words should flip orders)
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,-1,sizeof(uint16_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint16_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x22330011);  EXPECT_EQ(h_b[0*limbs+1],0x66774455);  EXPECT_EQ(h_b[1*limbs+0],0x02030001);  EXPECT_EQ(h_b[1*limbs+1],0x06070405);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,-1,sizeof(uint16_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint16_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x22330011);  EXPECT_EQ(h_b[0*limbs+1],0x66774455);  EXPECT_EQ(h_b[1*limbs+0],0x02030001);  EXPECT_EQ(h_b[1*limbs+1],0x06070405);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,1,sizeof(uint16_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint16_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x22330011);  EXPECT_EQ(h_b[0*limbs+1],0x66774455);  EXPECT_EQ(h_b[1*limbs+0],0x02030001);  EXPECT_EQ(h_b[1*limbs+1],0x06070405);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,1,sizeof(uint16_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint16_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x22330011);  EXPECT_EQ(h_b[0*limbs+1],0x66774455);  EXPECT_EQ(h_b[1*limbs+0],0x02030001);  EXPECT_EQ(h_b[1*limbs+1],0x06070405);
  
  //Copy in and out with different endians and different orders (should flip bytes and words)
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,1,sizeof(uint16_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint16_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,-1,sizeof(uint16_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint16_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,-1,sizeof(uint16_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint16_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*2,1,sizeof(uint16_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint16_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*2,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);

  //repeat tests for 1 byte words 
  //Copy in and out with same parameters, values should be unchanged
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,1,sizeof(uint8_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint8_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,-1,sizeof(uint8_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint8_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,1,sizeof(uint8_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint8_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,-1,sizeof(uint8_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint8_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);

  //Copy in and out with different orders (words should flip orders)
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,1,sizeof(uint8_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint8_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,1,sizeof(uint8_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint8_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,-1,sizeof(uint8_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint8_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,-1,sizeof(uint8_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint8_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);

  //Copy in and out with different endians (order should be unchanged)
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,-1,sizeof(uint8_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint8_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,-1,sizeof(uint8_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint8_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,1,sizeof(uint8_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint8_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,1,sizeof(uint8_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint8_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x33221100);  EXPECT_EQ(h_b[0*limbs+1],0x77665544);  EXPECT_EQ(h_b[1*limbs+0],0x03020100);  EXPECT_EQ(h_b[1*limbs+1],0x07060504);
  
  //Copy in and out with different endians and different orders (should be the same as just doing order)
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,1,sizeof(uint8_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint8_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,-1,sizeof(uint8_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint8_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,-1,sizeof(uint8_t),1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,1,sizeof(uint8_t),-1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs*4,1,sizeof(uint8_t),-1,0,h_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&olimbs,-1,sizeof(uint8_t),1,0,d_a,N)); 
  EXPECT_EQ(limbs*4,olimbs);
  EXPECT_EQ(h_b[0*limbs+0],0x44556677);  EXPECT_EQ(h_b[0*limbs+1],0x00112233);  EXPECT_EQ(h_b[1*limbs+0],0x04050607);  EXPECT_EQ(h_b[1*limbs+1],0x00010203);

  free(h_a);
  free(h_b);
  xmpIntegersDestroy(handle,d_a);
  
  xmpHandleDestroy(handle);
}

#if 0
//TODO DELETE OR CHANGE TO SET
TEST(transferTests,xmpIntegersSelect) {
  xmpHandle_t handle;
  xmpIntegers_t d_a, d_b;
  uint32_t *h_a, *h_b;
  int N=10;
  int P=32;
  uint32_t limbs=P/8/sizeof(uint32_t);
  uint32_t *a_indices, *b_indices;
  uint32_t words;

  xmpHandleCreate(&handle);

  h_a=(uint32_t*)malloc(N*sizeof(uint32_t));
  h_b=(uint32_t*)malloc(N*sizeof(uint32_t));
  a_indices = (uint32_t*)malloc(N*sizeof(uint32_t));
  b_indices = (uint32_t*)malloc(N*sizeof(uint32_t));
  
  for(int i=0;i<N*limbs;i++) h_a[i]=rand32();
  
  xmpIntegersCreate(handle,&d_a,P,N);
  xmpIntegersCreate(handle,&d_b,P,N);

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,d_a,limbs,-1,sizeof(uint32_t),-1,0,h_a,N));

  for(int i=0;i<N;i++) a_indices[i]=i;
  for(int i=0;i<N;i++) b_indices[i]=i;

  //generate random indices
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<N;i++) 
      std::swap(a_indices[i],a_indices[rand32()%N]);
    for(int i=0;i<N;i++) 
      std::swap(b_indices[i],b_indices[rand32()%N]);
  }

  //set indices in xmp
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,d_a,a_indices,N));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,d_b,b_indices,N));
  
  //this should reorder a and store into b
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersSelect(handle,d_b,d_a,N));

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&words,-1,sizeof(uint32_t),-1,0,d_b,N));
  ASSERT_EQ(limbs,words);

  //verify order is correct
  for(int i=0;i<N;i++) {
    ASSERT_EQ(h_a[a_indices[i]],h_b[b_indices[i]]);
  }
  //in place transform
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersSelect(handle,d_a,d_a,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h_b,&words,-1,sizeof(uint32_t),-1,0,d_a,N));
  ASSERT_EQ(limbs,words);
  
  //verify order is correct
  for(int i=0;i<N;i++) {
    ASSERT_EQ(h_a[a_indices[i]],h_b[i]);
  }

  free(h_a); free(h_b); free(a_indices); free(b_indices);
  xmpIntegersDestroy(handle,d_a);
  xmpIntegersDestroy(handle,d_b);
  
  xmpHandleDestroy(handle);
}
#endif


TEST(transferTests,xmpIntegersSet) {
  xmpHandle_t handle;
  xmpIntegers_t a, b;
  xmpHandleCreate(&handle);
  xmpIntegersCreate(handle,&a,64,100);
  xmpIntegersCreate(handle,&b,64,100);
  uint32_t limbs=64/8/sizeof(uint32_t);
  uint32_t *h=(uint32_t*)malloc(limbs*100);

  //need to import in order to set the format flag
  xmpIntegersImport(handle,b,limbs,1,sizeof(uint32_t),1,0,h,100);

  EXPECT_EQ(xmpErrorSuccess,xmpIntegersSet(handle,a,b,100));

  free(h);
  xmpIntegersDestroy(handle,a);
  xmpIntegersDestroy(handle,b);
  xmpHandleDestroy(handle);
}

TEST(transferTests,xmpOrderTest) {
  xmpHandle_t handle;
  xmpIntegers_t a, b, c;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&a,32,2));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&b,32,2));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&c,32,2));

  uint32_t words;
  uint32_t h[2];

  //array a:  1 and 2
  h[0]=1;  h[1]=2;
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,a,1,1,sizeof(uint32_t),0,0,h,2));

  //array b:  3 and 4
  h[0]=3;  h[1]=4;
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,b,1,1,sizeof(uint32_t),0,0,h,2));
 
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersAdd(handle,c,a,b,2));

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,h,&words,1,sizeof(uint32_t),0,0,c,2));
  EXPECT_EQ(h[0],4);
  EXPECT_EQ(h[1],6);

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,a));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,b));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,c));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST(TransferTests,xmpIndexTests) {
  const int N=10;

  xmpHandle_t handle;
  xmpExecutionPolicy_t policy;
  xmpIntegers_t a, b;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyCreate(handle,&policy));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&a,32,N));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&b,32,N));

  uint32_t words;
  uint32_t indices[N];
  uint32_t h[N];
  uint32_t res[N];

  for(int i=0;i<N;i++) {
    h[i] = i;
    indices[i] = i;
  }
  
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<N;i++) 
      std::swap(indices[i],indices[rand32()%N]);
  }

  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,indices,N));

  //enable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,a,1,-1,sizeof(uint32_t),0,0,h,N));

  //disable indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,res,&words,-1,sizeof(uint32_t),0,0,a,N));
  
  //enable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,res,&words,-1,sizeof(uint32_t),0,0,a,N));
  
  for(int i=0;i<N;i++) {
    ASSERT_EQ(res[indices[i]],h[indices[i]]);
  }

  //disable indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImport(handle,a,1,-1,sizeof(uint32_t),0,0,h,N));
  
  //enable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,res,&words,-1,sizeof(uint32_t),0,0,a,N));

  for(int i=0;i<N;i++) {
    ASSERT_EQ(res[indices[i]],h[i]);
  }

  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,NULL,N));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,1,indices,N));

  //should reorder a according to indices and store in b
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersSet(handle,b,a,N));

  //disable indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExport(handle,res,&words,-1,sizeof(uint32_t),0,0,b,N));
  
  for(int i=0;i<N;i++) {
    ASSERT_EQ(res[i],h[indices[i]]);
  }

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,a));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyDestroy(handle,policy));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST(errorTests,xmpImportExportErrors) {
  xmpHandle_t handle;
  xmpIntegers_t a;
  xmpHandleCreate(&handle);
  xmpIntegersCreate(handle,&a,64,100);
  uint32_t limbs=64/8/sizeof(uint32_t);
  uint32_t *h=(uint32_t*)malloc(limbs*100);

  //copy size too big
  EXPECT_NE(xmpErrorSuccess,xmpIntegersImport(handle,a,limbs,1,sizeof(uint32_t),1,0,h,200));
  EXPECT_NE(xmpErrorSuccess,xmpIntegersExport(handle,h,NULL,1,sizeof(uint32_t),1,0,a,200));

  free(h);
  xmpIntegersDestroy(handle,a);
  xmpHandleDestroy(handle);
}

TEST(errorTests,xmpSetErrors) {
  xmpHandle_t handle;
  xmpIntegers_t a, b;
  xmpHandleCreate(&handle);
  xmpIntegersCreate(handle,&a,64,100);
  xmpIntegersCreate(handle,&b,64,200);

  //set with too large of a count
  EXPECT_NE(xmpErrorSuccess,xmpIntegersSet(handle,a,b,200));
  
  xmpIntegersDestroy(handle,a);
  xmpIntegersDestroy(handle,b);
  xmpHandleDestroy(handle);
}

TEST(errorTests,xmpPowmErrors) {
  xmpHandle_t handle;
  xmpIntegers_t c, b, e, m;
  xmpHandleCreate(&handle);
  xmpIntegersCreate(handle,&c,64,100);
  xmpIntegersCreate(handle,&b,64,100);
  xmpIntegersCreate(handle,&m,64,100);
  xmpIntegersCreate(handle,&e,64,100);

  //count too large
  EXPECT_NE(xmpErrorSuccess,xmpIntegersPowmAsync(handle,c,b,e,m,200));
  
  //0 should just return success
  EXPECT_EQ(xmpErrorSuccess,xmpIntegersPowmAsync(handle,c,b,e,m,0));

  xmpIntegersDestroy(handle,c);
  xmpIntegersDestroy(handle,b);
  xmpIntegersDestroy(handle,m);
  xmpIntegersDestroy(handle,e);
  
  xmpIntegersCreate(handle,&m,64,100);
  xmpIntegersCreate(handle,&e,64,100);
  xmpIntegersCreate(handle,&c,128,100);
  xmpIntegersCreate(handle,&b,64,100);
  
  //precision missmatch
  EXPECT_NE(xmpErrorSuccess,xmpIntegersPowmAsync(handle,c,b,e,m,100));

  xmpIntegersDestroy(handle,m);
  xmpIntegersDestroy(handle,e);
  xmpIntegersDestroy(handle,c);
  xmpIntegersDestroy(handle,b);
  xmpHandleDestroy(handle);
}
typedef xmpError_t (*xmpOneInOneOutFunc)(xmpHandle_t, xmpIntegers_t, const xmpIntegers_t, uint32_t);
typedef xmpError_t (*xmpTwoInOneOutFunc)(xmpHandle_t, xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t);
typedef xmpError_t (*xmpThreeInOneOutFunc)(xmpHandle_t, xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t);
typedef xmpError_t (*xmpTwoInTwoOutFunc)(xmpHandle_t, xmpIntegers_t, xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t);

typedef void (*gmpOneInOneOutFunc)(mpz_ptr,mpz_srcptr);
typedef void (*gmpTwoInOneOutFunc)(mpz_ptr,mpz_srcptr,mpz_srcptr);
typedef void (*gmpThreeInOneOutFunc)(mpz_ptr,mpz_srcptr,mpz_srcptr,mpz_srcptr);
typedef void (*gmpTwoInTwoOutFunc)(mpz_ptr,mpz_ptr,mpz_srcptr,mpz_srcptr);

struct NotParams {
  uint32_t N1, N2, N;
  uint32_t P1, P2;

  NotParams(uint32_t P1, uint32_t P2, 
            uint32_t N1, uint32_t N2, uint32_t N) 
            : P1(P1), P2(P2), N1(N1), N2(N2), N(N) {}
  std::string DebugString() {
    std::stringstream s;
    s << "N1: " << N1 << " ";
    s << "N2: " << N2 << " ";
    s << "N: " << N << " ";
    s << "P1: " << P1 << " ";
    s << "P2: " << P2 << " ";
    return s.str();
  }
};
::std::ostream& operator<<(::std::ostream& os, NotParams params) {
    return os << params.DebugString(); 
}

struct ShfParams {
  uint32_t N1, N2, N3, N;
  uint32_t P1, P2;

  ShfParams(uint32_t P1, uint32_t P2, 
            uint32_t N1, uint32_t N2, uint32_t N3, uint32_t N) 
            : P1(P1), P2(P2), N1(N1), N2(N2), N3(N3), N(N) {}
  std::string DebugString() {
    std::stringstream s;
    s << "N1: " << N1 << " ";
    s << "N2: " << N2 << " ";
    s << "N3: " << N3 << " ";
    s << "N: " << N << " ";
    s << "P1: " << P1 << " ";
    s << "P2: " << P2 << " ";
    return s.str();
  }
};
::std::ostream& operator<<(::std::ostream& os, ShfParams params) {
    return os << params.DebugString(); 
}
struct SqrParams {
  uint32_t N1, N2, N;
  uint32_t P1, P2;

  SqrParams(uint32_t P1, uint32_t P2, 
            uint32_t N1, uint32_t N2, uint32_t N) 
            : P1(P1), P2(P2), N1(N1), N2(N2), N(N) {}
  std::string DebugString() {
    std::stringstream s;
    s << "N1: " << N1 << " ";
    s << "N2: " << N2 << " ";
    s << "N: " << N << " ";
    s << "P1: " << P1 << " ";
    s << "P2: " << P2 << " ";
    return s.str();
  }
};
::std::ostream& operator<<(::std::ostream& os, SqrParams params) {
    return os << params.DebugString(); 
}


struct TwoInOneOutParams {
  uint32_t N1, N2, N3, N;
  uint32_t P1, P2, P3;
  xmpTwoInOneOutFunc xfunc;
  gmpTwoInOneOutFunc gfunc;

  TwoInOneOutParams(xmpTwoInOneOutFunc xfunc, gmpTwoInOneOutFunc gfunc, 
                       uint32_t P1, uint32_t P2, uint32_t P3, 
                       uint32_t N1, uint32_t N2, uint32_t N3, uint32_t N) 
                       : xfunc(xfunc), gfunc(gfunc), P1(P1), P2(P2), P3(P3), N1(N1), N2(N2), N3(N3), N(N) {}
  std::string DebugString() {
    std::stringstream s;
    s << "N1: " << N1 << " ";
    s << "N2: " << N2 << " ";
    s << "N3: " << N3 << " ";
    s << "N: " << N << " ";
    s << "P1: " << P1 << " ";
    s << "P2: " << P2 << " ";
    s << "P3: " << P3 << " ";
    return s.str();
  }
};
::std::ostream& operator<<(::std::ostream& os, TwoInOneOutParams params) {
    return os << params.DebugString(); 
}

struct ThreeInOneOutParams {
  uint32_t N1, N2, N3, N4, N;
  uint32_t P1, P2, P3, P4;
  xmpThreeInOneOutFunc xfunc;
  gmpThreeInOneOutFunc gfunc;

  ThreeInOneOutParams(xmpThreeInOneOutFunc xfunc, gmpThreeInOneOutFunc gfunc, 
                       uint32_t P1, uint32_t P2, uint32_t P3, uint32_t P4,
                       uint32_t N1, uint32_t N2, uint32_t N3, uint32_t N4, uint32_t N) 
                       : xfunc(xfunc), gfunc(gfunc), P1(P1), P2(P2), P3(P3), P4(P4), N1(N1), N2(N2), N3(N3), N4(N4), N(N) {}
  std::string DebugString() {
    std::stringstream s;
    s << "N1: " << N1 << " ";
    s << "N2: " << N2 << " ";
    s << "N3: " << N3 << " ";
    s << "N4: " << N4 << " ";
    s << "N: " << N << " ";
    s << "P1: " << P1 << " ";
    s << "P2: " << P2 << " ";
    s << "P3: " << P3 << " ";
    s << "P4: " << P4 << " ";
    return s.str();
  }
};
::std::ostream& operator<<(::std::ostream& os, ThreeInOneOutParams params) {
    return os << params.DebugString(); 
}

struct TwoInTwoOutParams {
  uint32_t N1, N2, N3, N4, N;
  uint32_t P1, P2, P3, P4;
  xmpTwoInTwoOutFunc xfunc;
  gmpTwoInTwoOutFunc gfunc;

  TwoInTwoOutParams(xmpTwoInTwoOutFunc xfunc, gmpTwoInTwoOutFunc gfunc, 
                       uint32_t P1, uint32_t P2, uint32_t P3, uint32_t P4,
                       uint32_t N1, uint32_t N2, uint32_t N3, uint32_t N4, uint32_t N) 
                       : xfunc(xfunc), gfunc(gfunc), P1(P1), P2(P2), P3(P3), P4(P4), N1(N1), N2(N2), N3(N3), N4(N4), N(N) {}
  std::string DebugString() {
    std::stringstream s;
    s << "N1: " << N1 << " ";
    s << "N2: " << N2 << " ";
    s << "N3: " << N3 << " ";
    s << "N4: " << N4 << " ";
    s << "N: " << N << " ";
    s << "P1: " << P1 << " ";
    s << "P2: " << P2 << " ";
    s << "P3: " << P3 << " ";
    s << "P4: " << P4 << " ";
    return s.str();
  }
};
::std::ostream& operator<<(::std::ostream& os, TwoInTwoOutParams params) {
    return os << params.DebugString(); 
}

//specialized because we need to ensure modulous is odd
struct PowmParams {
  uint32_t N1, N2, N3, N4, N;
  uint32_t P1, P2, P3, P4;
  xmpAlgorithm_t algorithm;
  PowmParams(uint32_t P1, uint32_t P2, uint32_t P3, uint32_t P4, uint32_t N1, uint32_t N2, uint32_t N3, uint32_t N4, uint32_t N, xmpAlgorithm_t algorithm) : P1(P1), P2(P2), P3(P3), P4(P4), N1(N1), N2(N2), N3(N3), N4(N4), N(N), algorithm(algorithm) {}
  std::string DebugString() {
    std::stringstream s;
    s << "N1: " << N1 << " ";
    s << "N2: " << N2 << " ";
    s << "N3: " << N3 << " ";
    s << "N4: " << N4 << " ";
    s << "N: " << N << " ";
    s << "P1: " << P1 << " ";
    s << "P2: " << P2 << " ";
    s << "P3: " << P3 << " ";
    s << "P4: " << P4 << " ";
    s << "Algorithm: " << getAlgorithmString(algorithm) << " ";
    return s.str();
  }
};
::std::ostream& operator<<(::std::ostream& os, PowmParams params) {
    return os << params.DebugString(); 
}

//specialized because mpz and xmp different in how they handle negative numbers
struct SubParams {
  uint32_t N1, N2, N3, N;
  uint32_t P1, P2, P3;
  SubParams(uint32_t P1, uint32_t P2, uint32_t P3, uint32_t N1, uint32_t N2, uint32_t N3, uint32_t N) : P1(P1), P2(P2), P3(P3), N1(N1), N2(N2), N3(N3), N(N) {}
  std::string DebugString() {
    std::stringstream s;
    s << "N1: " << N1 << " ";
    s << "N2: " << N2 << " ";
    s << "N3: " << N3 << " ";
    s << "N: " << N << " ";
    s << "P1: " << P1 << " ";
    s << "P2: " << P2 << " ";
    s << "P3: " << P3 << " ";
    return s.str();
  }
};
::std::ostream& operator<<(::std::ostream& os, SubParams params) {
    return os << params.DebugString(); 
}

struct CmpParams {
  uint32_t N1, N2, N;
  uint32_t P1, P2;
  CmpParams(uint32_t P1, uint32_t P2, uint32_t N1, uint32_t N2, uint32_t N) : P1(P1), P2(P2), N1(N1), N2(N2), N(N) {}
  std::string DebugString() {
    std::stringstream s;
    s << "N1: " << N1 << " ";
    s << "N2: " << N2 << " ";
    s << "N: " << N << " ";
    s << "P1: " << P1 << " ";
    s << "P2: " << P2 << " ";
    return s.str();
  }
};
::std::ostream& operator<<(::std::ostream& os, CmpParams params) {
    return os << params.DebugString(); 
}

struct PopcParams {
  uint32_t N1, N;
  uint32_t P1;
  PopcParams(uint32_t P1, uint32_t N1, uint32_t N) : P1(P1), N1(N1), N(N) {}
  std::string DebugString() {
    std::stringstream s;
    s << "N1: " << N1 << " ";
    s << "N: " << N << " ";
    s << "P1: " << P1 << " ";
    return s.str();
  }
};
::std::ostream& operator<<(::std::ostream& os, PopcParams params) {
    return os << params.DebugString(); 
}


class PowmTest : public ::testing::TestWithParam<PowmParams> {};
class SubTest : public ::testing::TestWithParam<SubParams> {};
class CmpTest : public ::testing::TestWithParam<CmpParams> {};
class PopcTest : public ::testing::TestWithParam<PopcParams> {};
class NotTest : public ::testing::TestWithParam<NotParams> {};
class ShfTest : public ::testing::TestWithParam<ShfParams> {};
class SqrTest : public ::testing::TestWithParam<SqrParams> {};
class genericTwoInOneOutTest : public ::testing::TestWithParam<TwoInOneOutParams> {};
class genericTwoInTwoOutTest : public ::testing::TestWithParam<TwoInTwoOutParams> {};

TEST_P(PowmTest,opTests) {
  PowmParams p=GetParam();
  uint32_t cN=p.N1, bN=p.N2, eN=p.N3, mN=p.N4, N=p.N;
  uint32_t cP=p.P1, bP=p.P2, eP=p.P3, mP=p.P4;
  uint32_t climbs=cP/(8*sizeof(uint32_t));
  uint32_t blimbs=bP/(8*sizeof(uint32_t));
  uint32_t elimbs=eP/(8*sizeof(uint32_t));
  uint32_t mlimbs=mP/(8*sizeof(uint32_t));

  //allocate xmp integers
  xmpHandle_t handle;
  xmpExecutionPolicy_t policy;
  xmpIntegers_t x_c, x_b, x_e, x_m;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyCreate(handle,&policy));
  xmpAlgorithm_t algorithm=p.algorithm;

  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetParameter(handle,policy,xmpAlgorithm,algorithm));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_c,cP,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_b,bP,bN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_m,mP,mN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_e,eP,eN));

  //allocate memory on hosts
  uint32_t *h_b, *h_e, *h_m;
  uint32_t *h_res, *d_res;
  h_b=(uint32_t*)malloc(sizeof(uint32_t)*bN*blimbs);
  h_m=(uint32_t*)malloc(sizeof(uint32_t)*mN*mlimbs);
  h_e=(uint32_t*)malloc(sizeof(uint32_t)*eN*elimbs);
  h_res=(uint32_t*)malloc(sizeof(uint32_t)*cN*climbs);
  d_res=(uint32_t*)malloc(sizeof(uint32_t)*cN*climbs);

  //allocate gmp integers
  mpz_t *g_c, *g_b, *g_e, *g_m;

  g_c=(mpz_t*)malloc(sizeof(mpz_t)*cN);
  g_b=(mpz_t*)malloc(sizeof(mpz_t)*bN);
  g_e=(mpz_t*)malloc(sizeof(mpz_t)*eN);
  g_m=(mpz_t*)malloc(sizeof(mpz_t)*mN);

  for(int i=0;i<cN;i++) mpz_init(g_c[i]);
  for(int i=0;i<bN;i++) mpz_init(g_b[i]);
  for(int i=0;i<eN;i++) mpz_init(g_e[i]);
  for(int i=0;i<mN;i++) mpz_init(g_m[i]);

  //generate random data on host
  srand(0);

  for(int i=0;i<bN*blimbs;i++) h_b[i]=rand32();
  for(int i=0;i<eN*elimbs;i++) h_e[i]=rand32();
  for(int i=0;i<mN*mlimbs;i++) h_m[i]=rand32();
  for(int i=0;i<mN;i++) h_m[i*mlimbs]|=0x1;  //ensure least significant bit is odd.

  //import to xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_b,blimbs,-1,sizeof(uint32_t),-1,0,h_b,bN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_e,elimbs,-1,sizeof(uint32_t),-1,0,h_e,eN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_m,mlimbs,-1,sizeof(uint32_t),-1,0,h_m,mN));

  uint32_t *c_indices, *e_indices, *b_indices, *m_indices;
  c_indices=(uint32_t*)malloc(cN*sizeof(uint32_t));
  b_indices=(uint32_t*)malloc(bN*sizeof(uint32_t));
  e_indices=(uint32_t*)malloc(eN*sizeof(uint32_t));
  m_indices=(uint32_t*)malloc(mN*sizeof(uint32_t));

  for(int i=0;i<cN;i++) c_indices[i]=i;
  for(int i=0;i<bN;i++) b_indices[i]=i;
  for(int i=0;i<eN;i++) e_indices[i]=i;
  for(int i=0;i<mN;i++) m_indices[i]=i;

  //generate random indices for a, b and c
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<cN;i++) 
      std::swap(c_indices[i],c_indices[rand32()%cN]);
    for(int i=0;i<bN;i++) 
      std::swap(b_indices[i],b_indices[rand32()%bN]);
    for(int i=0;i<eN;i++) 
      std::swap(e_indices[i],e_indices[rand32()%eN]);
    for(int i=0;i<mN;i++) 
      std::swap(m_indices[i],m_indices[rand32()%mN]);
  }

  //set indices in xmp
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,c_indices,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,1,b_indices,bN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,2,e_indices,eN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,3,m_indices,mN));

  //Enable policy for dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));

  //perform operation the device
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersPowmAsync(handle,x_c,x_b,x_e,x_m,N));
  
  //disable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));
 
  uint32_t nlimbs;
  //export from xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,&nlimbs,-1,sizeof(uint32_t),-1,0,x_c,cN));
  ASSERT_EQ(nlimbs,climbs);
  //import to mpz
#pragma omp parallel for
  for(int i=0;i<bN;i++) mpz_import(g_b[i],blimbs,-1,sizeof(uint32_t),-1,0,&h_b[i*blimbs]);
#pragma omp parallel for
  for(int i=0;i<eN;i++) mpz_import(g_e[i],elimbs,-1,sizeof(uint32_t),-1,0,&h_e[i*elimbs]);
#pragma omp parallel for
  for(int i=0;i<mN;i++) mpz_import(g_m[i],mlimbs,-1,sizeof(uint32_t),-1,0,&h_m[i*mlimbs]);

  //perform operation on the host
#pragma omp parallel for
  for(int i=0;i<N;i++) {
    mpz_t *cc=&g_c[c_indices[i]], *bb, *ee, *mm;
    bb= &g_b[b_indices[i%bN] % bN]; 
    ee= &g_e[e_indices[i%eN] % eN]; 
    mm= &g_m[m_indices[i%mN] % mN ]; 
    mpz_powm(*cc,*bb,*ee,*mm);
  }

  //export from gmp
#pragma omp parallel for
  for(int i=0;i<N;i++) mpz_export(&h_res[c_indices[i]*climbs],NULL,-1,sizeof(uint32_t),-1,0,g_c[c_indices[i]]);

  ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());

  //compare results
  for(int i=0;i<N;i++) {
    for(int j=0;j<climbs;j++) {
      ASSERT_EQ(h_res[c_indices[i]*climbs+j],d_res[c_indices[i]*climbs+j]);
    }
  }

  //in place tests
  if(cN==bN && cP==bP) {

    //Enable policy for dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));

    ASSERT_EQ(xmpErrorSuccess,xmpIntegersPowmAsync(handle,x_b,x_b,x_e,x_m,N));

    //disable dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));

    //export from xmp
    ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,NULL,-1,sizeof(uint32_t),-1,0,x_b,bN));

    ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());

    //compare results
    for(int i=0;i<N;i++) {
      for(int j=0;j<climbs;j++) {
        ASSERT_EQ(h_res[c_indices[i]*climbs+j],d_res[c_indices[i]*climbs+j]);
      }
    }
  }

  //clean up
  for(int i=0;i<cN;i++) mpz_clear(g_c[i]);
  for(int i=0;i<bN;i++) mpz_clear(g_b[i]);
  for(int i=0;i<eN;i++) mpz_clear(g_e[i]);
  for(int i=0;i<mN;i++) mpz_clear(g_m[i]);

  free(c_indices), free(e_indices), free(b_indices), free(m_indices);
  free(h_b); free(h_e); free(h_m); free(h_res); free(d_res);
  free(g_c); free(g_b); free(g_e); free(g_m);

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_m));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_e));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_c));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_b));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyDestroy(handle,policy));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST_P(ShfTest,opTests) {
  ShfParams p=GetParam();
  uint32_t cN=p.N1, aN=p.N2, sN=p.N3, N=p.N;
  uint32_t cP=p.P1, aP=p.P2;
  uint32_t climbs=cP/(8*sizeof(uint32_t));
  uint32_t alimbs=aP/(8*sizeof(uint32_t));
  uint32_t hlimbs=aP*2/(8*sizeof(uint32_t));

  //allocate xmp integers
  xmpHandle_t handle;
  xmpExecutionPolicy_t policy;
  xmpIntegers_t x_c, x_a;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyCreate(handle,&policy));

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_c,cP,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_a,aP,aN));

  //allocate memory on hosts
  uint32_t *h_a;
  uint32_t *d_res, *h_res;
  int32_t *shift;
  h_a=(uint32_t*)malloc(sizeof(uint32_t)*aN*alimbs);
  d_res=(uint32_t*)malloc(sizeof(uint32_t)*N*climbs);
  h_res=(uint32_t*)malloc(sizeof(uint32_t)*N*hlimbs);
  shift=(int32_t*)malloc(sizeof(int32_t)*sN);

  //intitialize resutls to 0 as gmp may not write all of the bits 
  memset(d_res,0,sizeof(uint32_t)*N*climbs);
  memset(h_res,0,sizeof(uint32_t)*N*hlimbs);

  mpz_t *g_c, *g_a;

  g_c=(mpz_t*)malloc(sizeof(mpz_t)*cN);
  g_a=(mpz_t*)malloc(sizeof(mpz_t)*aN);

  for(int i=0;i<cN;i++) mpz_init(g_c[i]);
  for(int i=0;i<aN;i++) mpz_init(g_a[i]);

  //generate random data on host
  srand(0);

  for(int i=0;i<aN*alimbs;i++) h_a[i]=rand32();
  for(int i=0;i<sN;i++) shift[i]=rand32()%(2*aP-2)-aP+1;

  //import to xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_a,alimbs,-1,sizeof(uint32_t),-1,0,h_a,aN));

  uint32_t *a_indices, *c_indices;
  a_indices=(uint32_t*)malloc(aN*sizeof(uint32_t));
  c_indices=(uint32_t*)malloc(cN*sizeof(uint32_t));

  for(int i=0;i<aN;i++) a_indices[i]=i;
  for(int i=0;i<cN;i++) c_indices[i]=i;

  //generate random indices for a, b and c
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<aN;i++) 
      std::swap(a_indices[i],a_indices[rand32()%aN]);
    for(int i=0;i<cN;i++) 
      std::swap(c_indices[i],c_indices[rand32()%cN]);
  }

  //set indices in xmp
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,c_indices,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,1,a_indices,aN));

  //Enable policy for dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));

  //perform operation the device
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersShfAsync(handle,x_c,x_a,shift,sN,N));

  //disable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));

  uint32_t words;
  //export from xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,&words,-1,sizeof(uint32_t),-1,0,x_c,cN));

  //import to mpz
#pragma omp parallel for
  for(int i=0;i<aN;i++) mpz_import(g_a[i],alimbs,-1,sizeof(uint32_t),-1,0,&h_a[i*alimbs]);

  //perform operation on the host
#pragma omp parallel for
  for(int i=0;i<N;i++) {
    mpz_t *cc=&g_c[c_indices[i]], *aa;
    aa= &g_a[a_indices[i%aN] % aN]; 
    if(shift[i]>=0) { //mul
      mpz_mul_2exp(*cc,*aa,shift[i]);
    } else { //div
      mpz_tdiv_q_2exp(*cc,*aa,-shift[i]);
    }
  }

  //export from gmp
#pragma omp parallel for
  for(int i=0;i<N;i++) mpz_export(&h_res[c_indices[i]*hlimbs],NULL,-1,sizeof(uint32_t),-1,0,g_c[c_indices[i]]);

  ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());


  //compare results
  for(int i=0;i<N;i++) {
    for(int j=0;j<climbs;j++) {
      ASSERT_EQ(h_res[c_indices[i]*hlimbs+j],d_res[c_indices[i]*climbs+j]);
    }
  }

  //do in place tests when they make sense
  if(cP==aP && cN==aN) {
    //Enable policy for dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));    

    ASSERT_EQ(xmpErrorSuccess,xmpIntegersShfAsync(handle,x_a,x_a,shift,sN,N));

    //disable dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));

    ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,&words,-1,sizeof(uint32_t),-1,0,x_a,aN));

    ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());

    //compare results
    for(int i=0;i<N;i++) {
      for(int j=0;j<climbs;j++) {
        ASSERT_EQ(h_res[c_indices[i]*hlimbs+j],d_res[c_indices[i]*climbs+j]);
      }
    }
  }

  for(int i=0;i<cN;i++) mpz_clear(g_c[i]);
  for(int i=0;i<aN;i++) mpz_clear(g_a[i]);

  free(a_indices); free(c_indices);
  free(g_c); free(g_a);
  free(h_a); free(h_res); free(d_res);

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_c));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_a));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyDestroy(handle,policy));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST_P(NotTest,opTests) {
  NotParams p=GetParam();
  uint32_t cN=p.N1, aN=p.N2, N=p.N;
  uint32_t cP=p.P1, aP=p.P2;
  uint32_t climbs=cP/(8*sizeof(uint32_t));
  uint32_t alimbs=aP/(8*sizeof(uint32_t));

  //allocate xmp integers
  xmpHandle_t handle;
  xmpExecutionPolicy_t policy;
  xmpIntegers_t x_c, x_a;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyCreate(handle,&policy));

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_c,cP,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_a,aP,aN));

  //allocate memory on hosts
  uint32_t *h_a;
  uint32_t *d_res;
  h_a=(uint32_t*)malloc(sizeof(uint32_t)*aN*alimbs);
  d_res=(uint32_t*)malloc(sizeof(uint32_t)*N*climbs);

  //intitialize resutls to 0 as gmp may not write all of the bits 
  memset(d_res,0,sizeof(uint32_t)*N*climbs);

  //generate random data on host
  srand(0);

  for(int i=0;i<aN*alimbs;i++) h_a[i]=rand32();

  //import to xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_a,alimbs,-1,sizeof(uint32_t),-1,0,h_a,aN));

  uint32_t *a_indices, *c_indices;
  a_indices=(uint32_t*)malloc(aN*sizeof(uint32_t));
  c_indices=(uint32_t*)malloc(cN*sizeof(uint32_t));

  for(int i=0;i<aN;i++) a_indices[i]=i;
  for(int i=0;i<cN;i++) c_indices[i]=i;

  //generate random indices for a, b and c
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<aN;i++) 
      std::swap(a_indices[i],a_indices[rand32()%aN]);
    for(int i=0;i<cN;i++) 
      std::swap(c_indices[i],c_indices[rand32()%cN]);
  }

  //set indices in xmp
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,c_indices,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,1,a_indices,aN));

  //Enable policy for dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy)); 


  //perform operation the device
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersNotAsync(handle,x_c,x_a,N));

  //disable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));

  uint32_t words;
  //export from xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,&words,-1,sizeof(uint32_t),-1,0,x_c,cN));

  ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());

  //compare results
  for(int i=0;i<N;i++) {
    for(int j=0;j<climbs;j++) {
      if(j<alimbs) 
        ASSERT_EQ(~h_a[a_indices[i%aN]%aN*alimbs+j],d_res[c_indices[i]*climbs+j]);
      else 
        ASSERT_EQ(~0,d_res[c_indices[i]*climbs+j]);
    }
  }

  if(cN==aN && cP==aP) {

    //Enable policy for dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));

    ASSERT_EQ(xmpErrorSuccess,xmpIntegersNotAsync(handle,x_a,x_a,N));


    //disable dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));

    ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,&words,-1,sizeof(uint32_t),-1,0,x_a,aN));

    ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());

    //compare results
    for(int i=0;i<N;i++) {
      for(int j=0;j<climbs;j++) {
        if(j<alimbs)
          ASSERT_EQ(~h_a[a_indices[i%aN]%aN*alimbs+j],d_res[c_indices[i]*climbs+j]);
        else 
          ASSERT_EQ(~0,d_res[c_indices[i]*climbs+j]);
      }
    }
  }

  free(a_indices); free(c_indices);
  free(h_a); free(d_res);

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_c));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_a));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyDestroy(handle,policy));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST_P(SqrTest,opTests) {
  SqrParams p=GetParam();
  uint32_t cN=p.N1, aN=p.N2, N=p.N;
  uint32_t cP=p.P1, aP=p.P2;
  uint32_t climbs=cP/(8*sizeof(uint32_t));
  uint32_t alimbs=aP/(8*sizeof(uint32_t));
  uint32_t hlimbs=alimbs*2;
 
  //allocate xmp integers
  xmpHandle_t handle;
  xmpExecutionPolicy_t policy;
  xmpIntegers_t x_c, x_a;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyCreate(handle,&policy));
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_c,cP,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_a,aP,aN));

  //allocate memory on hosts
  uint32_t *h_a;
  uint32_t *h_res, *d_res;
  h_a=(uint32_t*)malloc(sizeof(uint32_t)*aN*alimbs);
  h_res=(uint32_t*)malloc(sizeof(uint32_t)*N*hlimbs);
  d_res=(uint32_t*)malloc(sizeof(uint32_t)*N*climbs);

  //intitialize resutls to 0 as gmp may not write all of the bits 
  memset(h_res,0,sizeof(uint32_t)*N*hlimbs);
  memset(d_res,0,sizeof(uint32_t)*N*climbs);

  //allocate gmp integers
  mpz_t *g_c, *g_a;

  g_c=(mpz_t*)malloc(sizeof(mpz_t)*cN);
  g_a=(mpz_t*)malloc(sizeof(mpz_t)*aN);

  for(int i=0;i<cN;i++) mpz_init(g_c[i]);
  for(int i=0;i<aN;i++) mpz_init(g_a[i]);

  //generate random data on host
  srand(0);

  for(int i=0;i<aN*alimbs;i++) h_a[i]=rand32();

  //import to xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_a,alimbs,-1,sizeof(uint32_t),-1,0,h_a,aN));
  
  uint32_t *c_indices;
  c_indices=(uint32_t*)malloc(cN*sizeof(uint32_t));

  for(int i=0;i<cN;i++) c_indices[i]=i;

  //generate random indices for a, b and c
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<cN;i++) 
      std::swap(c_indices[i],c_indices[rand32()%cN]);
  }
 
  //set indices in xmp
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,c_indices,cN));

  //Enable policy for dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));
  //perform operation the device
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersMul(handle,x_c,x_a,x_a,N));

  //disable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));

  uint32_t words;
  //export from xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,&words,-1,sizeof(uint32_t),-1,0,x_c,cN));
  
  ASSERT_LE(words,climbs);
  //import to mpz
  #pragma omp parallel for
  for(int i=0;i<aN;i++) mpz_import(g_a[i],alimbs,-1,sizeof(uint32_t),-1,0,&h_a[i*alimbs]);

  //perform operation on the host
  #pragma omp parallel for
  for(int i=0;i<N;i++) {
    mpz_t *cc=&g_c[c_indices[i]], *aa;
    aa= &g_a[i % aN]; 
    mpz_mul(*cc,*aa,*aa);
  }

  //perform operation on the host
  size_t gwords;
  //export from gmp
  #pragma omp parallel for
  for(int i=0;i<N;i++) mpz_export(&h_res[c_indices[i]*hlimbs],&gwords,-1,sizeof(uint32_t),-1,0,g_c[c_indices[i]]);
  
  ASSERT_LE(gwords,hlimbs);

  ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());
  //compare results
  for(int i=0;i<N;i++) {
    for(int j=0;j<climbs;j++) {
      ASSERT_EQ(h_res[c_indices[i]*hlimbs+j],d_res[c_indices[i]*climbs+j]);
    }
  }

  //in place tests
  if(aN==cN && aP==cP) {
    //Enable policy for dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));
    
    //perform operation the device
    ASSERT_EQ(xmpErrorSuccess,xmpIntegersMul(handle,x_a,x_a,x_a,N));
  
    //disable dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));
 
    //export from xmp
    ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,&words,-1,sizeof(uint32_t),-1,0,x_a,aN));
    ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());
    ASSERT_EQ(climbs,words);

    //compare results
    for(int i=0;i<N;i++) {
      for(int j=0;j<climbs;j++) {
        ASSERT_EQ(h_res[c_indices[i]*hlimbs+j],d_res[c_indices[i]*climbs+j]);
      }
    }
  }

  //clean up
  for(int i=0;i<cN;i++) mpz_clear(g_c[i]);
  for(int i=0;i<aN;i++) mpz_clear(g_a[i]);

  free(c_indices);

  free(h_a); free(h_res); free(d_res);
  free(g_c); free(g_a); 

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_c));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_a));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyDestroy(handle,policy));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST_P(genericTwoInOneOutTest,opTests) {
  TwoInOneOutParams p=GetParam();
  uint32_t cN=p.N1, aN=p.N2, bN=p.N3, N=p.N;
  uint32_t cP=p.P1, aP=p.P2, bP=p.P3;
  uint32_t climbs=cP/(8*sizeof(uint32_t));
  uint32_t alimbs=aP/(8*sizeof(uint32_t));
  uint32_t blimbs=bP/(8*sizeof(uint32_t));
  uint32_t hlimbs=alimbs+blimbs;
 
  xmpTwoInOneOutFunc xfunc=p.xfunc;
  gmpTwoInOneOutFunc gfunc=p.gfunc;

  //allocate xmp integers
  xmpHandle_t handle;
  xmpExecutionPolicy_t policy;
  xmpIntegers_t x_c, x_a, x_b;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyCreate(handle,&policy));
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_c,cP,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_a,aP,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_b,bP,bN));

  //allocate memory on hosts
  uint32_t *h_a, *h_b;
  uint32_t *h_res, *d_res;
  h_a=(uint32_t*)malloc(sizeof(uint32_t)*aN*alimbs);
  h_b=(uint32_t*)malloc(sizeof(uint32_t)*bN*blimbs);
  h_res=(uint32_t*)malloc(sizeof(uint32_t)*N*hlimbs);
  d_res=(uint32_t*)malloc(sizeof(uint32_t)*N*climbs);

  //intitialize resutls to 0 as gmp may not write all of the bits 
  memset(h_res,0,sizeof(uint32_t)*N*hlimbs);
  memset(d_res,0,sizeof(uint32_t)*N*climbs);

  //allocate gmp integers
  mpz_t *g_c, *g_a, *g_b;

  g_c=(mpz_t*)malloc(sizeof(mpz_t)*cN);
  g_a=(mpz_t*)malloc(sizeof(mpz_t)*aN);
  g_b=(mpz_t*)malloc(sizeof(mpz_t)*bN);

  for(int i=0;i<cN;i++) mpz_init(g_c[i]);
  for(int i=0;i<aN;i++) mpz_init(g_a[i]);
  for(int i=0;i<bN;i++) mpz_init(g_b[i]);

  //generate random data on host
  srand(0);

  for(int i=0;i<aN*alimbs;i++) h_a[i]=rand32();
  for(int i=0;i<bN*blimbs;i++) h_b[i]=rand32();

  //import to xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_a,alimbs,-1,sizeof(uint32_t),-1,0,h_a,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_b,blimbs,-1,sizeof(uint32_t),-1,0,h_b,bN));
  
  uint32_t *a_indices, *b_indices, *c_indices;
  a_indices=(uint32_t*)malloc(aN*sizeof(uint32_t));
  b_indices=(uint32_t*)malloc(bN*sizeof(uint32_t));
  c_indices=(uint32_t*)malloc(cN*sizeof(uint32_t));

  for(int i=0;i<aN;i++) a_indices[i]=i;
  for(int i=0;i<bN;i++) b_indices[i]=i;
  for(int i=0;i<cN;i++) c_indices[i]=i;

  //generate random indices for a, b and c
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<aN;i++) 
      std::swap(a_indices[i],a_indices[rand32()%aN]);
    for(int i=0;i<bN;i++) 
      std::swap(b_indices[i],b_indices[rand32()%bN]);
    for(int i=0;i<cN;i++) 
      std::swap(c_indices[i],c_indices[rand32()%cN]);
  }
 
  //set indices in xmp
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,c_indices,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,1,a_indices,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,2,b_indices,bN));

  //Enable policy for dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));
  //perform operation the device
  ASSERT_EQ(xmpErrorSuccess,xfunc(handle,x_c,x_a,x_b,N));

  //disable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));

  uint32_t words;
  //export from xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,&words,-1,sizeof(uint32_t),-1,0,x_c,cN));
  
  ASSERT_LE(words,climbs);
  //import to mpz
  #pragma omp parallel for
  for(int i=0;i<aN;i++) mpz_import(g_a[i],alimbs,-1,sizeof(uint32_t),-1,0,&h_a[i*alimbs]);
  #pragma omp parallel for
  for(int i=0;i<bN;i++) mpz_import(g_b[i],blimbs,-1,sizeof(uint32_t),-1,0,&h_b[i*blimbs]);

  //perform operation on the host
  #pragma omp parallel for
  for(int i=0;i<N;i++) {
    mpz_t *cc=&g_c[c_indices[i]], *aa, *bb;
    aa= &g_a[a_indices[i%aN] % aN]; 
    bb= &g_b[b_indices[i%bN] % bN]; 
    gfunc(*cc,*aa,*bb);
  }

  //perform operation on the host
  size_t gwords;
  //export from gmp
  #pragma omp parallel for
  for(int i=0;i<N;i++) mpz_export(&h_res[c_indices[i]*hlimbs],&gwords,-1,sizeof(uint32_t),-1,0,g_c[c_indices[i]]);
  
  ASSERT_LE(gwords,alimbs+blimbs);

  ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());
  //compare results
  for(int i=0;i<N;i++) {
    for(int j=0;j<climbs;j++) {
      ASSERT_EQ(h_res[c_indices[i]*hlimbs+j],d_res[c_indices[i]*climbs+j]);
    }
  }

  //in place tests
  if(aN==cN && aP==cP) {
    //Enable policy for dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));
    
    //perform operation the device
    ASSERT_EQ(xmpErrorSuccess,xfunc(handle,x_a,x_a,x_b,N));
  
    //disable dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));
 
    //export from xmp
    ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,&words,-1,sizeof(uint32_t),-1,0,x_a,aN));
    ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());
    ASSERT_EQ(climbs,words);

    //compare results
    for(int i=0;i<N;i++) {
      for(int j=0;j<climbs;j++) {
        ASSERT_EQ(h_res[c_indices[i]*hlimbs+j],d_res[c_indices[i]*climbs+j]);
      }
    }
  }

  //clean up
  for(int i=0;i<cN;i++) mpz_clear(g_c[i]);
  for(int i=0;i<aN;i++) mpz_clear(g_a[i]);
  for(int i=0;i<bN;i++) mpz_clear(g_b[i]);

  free(a_indices); free(b_indices); free(c_indices);

  free(h_a); free(h_b); free(h_res); free(d_res);
  free(g_c); free(g_a); free(g_b); 

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_c));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_a));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_b));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyDestroy(handle,policy));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST_P(genericTwoInTwoOutTest,opTests) {
  TwoInTwoOutParams p=GetParam();
  uint32_t cN=p.N1, dN=p.N2, aN=p.N3, bN=p.N4, N=p.N;
  uint32_t cP=p.P1, dP=p.P2, aP=p.P3, bP=p.P4;
  uint32_t climbs=cP/(8*sizeof(uint32_t));
  uint32_t alimbs=aP/(8*sizeof(uint32_t));
  uint32_t blimbs=bP/(8*sizeof(uint32_t));
  uint32_t dlimbs=dP/(8*sizeof(uint32_t));
  
  xmpTwoInTwoOutFunc xfunc=p.xfunc;
  gmpTwoInTwoOutFunc gfunc=p.gfunc;

  //allocate xmp integers
  xmpHandle_t handle;
  xmpExecutionPolicy_t policy;
  xmpIntegers_t x_c, x_d, x_a, x_b;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyCreate(handle,&policy));
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_c,cP,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_d,dP,dN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_a,aP,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_b,bP,bN));

  //allocate memory on hosts
  uint32_t *h_a, *h_b;
  uint32_t *h_resc, *d_resc, *h_resd, *d_resd;
  h_a=(uint32_t*)malloc(sizeof(uint32_t)*aN*alimbs);
  h_b=(uint32_t*)malloc(sizeof(uint32_t)*bN*blimbs);
  h_resc=(uint32_t*)malloc(sizeof(uint32_t)*N*climbs);
  d_resc=(uint32_t*)malloc(sizeof(uint32_t)*N*climbs);
  h_resd=(uint32_t*)malloc(sizeof(uint32_t)*N*dlimbs);
  d_resd=(uint32_t*)malloc(sizeof(uint32_t)*N*dlimbs);
  
  //intitialize resutls to 0 as gmp may not write all of the bits 
  memset(h_resc,0,sizeof(uint32_t)*N*climbs);
  memset(d_resc,0,sizeof(uint32_t)*N*climbs);
  memset(h_resd,0,sizeof(uint32_t)*N*dlimbs);
  memset(d_resd,0,sizeof(uint32_t)*N*dlimbs);

  //allocate gmp integers
  mpz_t *g_c, *g_d, *g_a, *g_b;

  g_c=(mpz_t*)malloc(sizeof(mpz_t)*cN);
  g_d=(mpz_t*)malloc(sizeof(mpz_t)*dN);
  g_a=(mpz_t*)malloc(sizeof(mpz_t)*aN);
  g_b=(mpz_t*)malloc(sizeof(mpz_t)*bN);

  for(int i=0;i<cN;i++) mpz_init(g_c[i]);
  for(int i=0;i<dN;i++) mpz_init(g_d[i]);
  for(int i=0;i<aN;i++) mpz_init(g_a[i]);
  for(int i=0;i<bN;i++) mpz_init(g_b[i]);

  //generate random data on host
  srand(0);

  for(int i=0;i<aN*alimbs;i++) h_a[i]=rand32();
  for(int i=0;i<bN*blimbs;i++) h_b[i]=rand32();

  //import to xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_a,alimbs,-1,sizeof(uint32_t),-1,0,h_a,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_b,blimbs,-1,sizeof(uint32_t),-1,0,h_b,bN));
  
  uint32_t *a_indices, *b_indices, *c_indices, *d_indices;
  a_indices=(uint32_t*)malloc(aN*sizeof(uint32_t));
  b_indices=(uint32_t*)malloc(bN*sizeof(uint32_t));
  c_indices=(uint32_t*)malloc(cN*sizeof(uint32_t));
  d_indices=(uint32_t*)malloc(dN*sizeof(uint32_t));

  for(int i=0;i<aN;i++) a_indices[i]=i;
  for(int i=0;i<bN;i++) b_indices[i]=i;
  for(int i=0;i<cN;i++) c_indices[i]=i;
  for(int i=0;i<dN;i++) d_indices[i]=i;

  //generate random indices for a, b and c
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<aN;i++) 
      std::swap(a_indices[i],a_indices[rand32()%aN]);
    for(int i=0;i<bN;i++) 
      std::swap(b_indices[i],b_indices[rand32()%bN]);
    for(int i=0;i<cN;i++) 
      std::swap(c_indices[i],c_indices[rand32()%cN]);
    for(int i=0;i<dN;i++) 
      std::swap(d_indices[i],d_indices[rand32()%dN]);
  }
  
  //set indices in xmp
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,c_indices,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,1,d_indices,dN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,2,a_indices,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,3,b_indices,bN));

  //Enable policy for dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));
  
  //perform operation the device
  ASSERT_EQ(xmpErrorSuccess,xfunc(handle,x_c,x_d,x_a,x_b,N));
 
  //disable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));
 
  //export from xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_resc,NULL,-1,sizeof(uint32_t),-1,0,x_c,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_resd,NULL,-1,sizeof(uint32_t),-1,0,x_d,dN));
  
  //import to mpz
  #pragma omp parallel for
  for(int i=0;i<aN;i++) mpz_import(g_a[i],alimbs,-1,sizeof(uint32_t),-1,0,&h_a[i*alimbs]);
  #pragma omp parallel for
  for(int i=0;i<bN;i++) mpz_import(g_b[i],blimbs,-1,sizeof(uint32_t),-1,0,&h_b[i*blimbs]);
  
  //perform operation on the host
  #pragma omp parallel for
  for(int i=0;i<N;i++) {
    mpz_t *cc=&g_c[c_indices[i]], *aa, *bb, *dd=&g_d[d_indices[i]];
    aa= &g_a[a_indices[i%aN] % aN]; 
    bb= &g_b[b_indices[i%bN] % bN]; 
    gfunc(*cc,*dd,*aa,*bb);
  }

  //export from gmp
  #pragma omp parallel for
  for(int i=0;i<N;i++) mpz_export(&h_resc[c_indices[i]*climbs],NULL,-1,sizeof(uint32_t),-1,0,g_c[c_indices[i]]);
  
  #pragma omp parallel for
  for(int i=0;i<N;i++) mpz_export(&h_resd[d_indices[i]*climbs],NULL,-1,sizeof(uint32_t),-1,0,g_d[d_indices[i]]);
  
  ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());
  
  //compare results
  for(int i=0;i<N;i++) {
    for(int j=0;j<climbs;j++) {
      ASSERT_EQ(h_resc[c_indices[i]*climbs+j],d_resc[c_indices[i]*climbs+j]);
    }
  }
  for(int i=0;i<N;i++) {
    for(int j=0;j<dlimbs;j++) {
      ASSERT_EQ(h_resd[d_indices[i]*dlimbs+j],d_resd[d_indices[i]*dlimbs+j]);
    }
  }

  //clean up
  for(int i=0;i<cN;i++) mpz_clear(g_c[i]);
  for(int i=0;i<dN;i++) mpz_clear(g_d[i]);
  for(int i=0;i<aN;i++) mpz_clear(g_a[i]);
  for(int i=0;i<bN;i++) mpz_clear(g_b[i]);

  free(a_indices); free(b_indices); free(c_indices); free(d_indices); 

  free(h_a); free(h_b); free(h_resc); free(h_resd); free(d_resc); free(d_resd);
  free(g_c); free(g_d); free(g_a); free(g_b); 

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_c));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_d));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_a));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_b));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyDestroy(handle,policy));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST_P(CmpTest,opTests) {
  CmpParams p=GetParam();
  uint32_t aN=p.N1, bN=p.N2, N=p.N;
  uint32_t aP=p.P1, bP=p.P2;
  uint32_t alimbs=aP/(8*sizeof(uint32_t));
  uint32_t blimbs=bP/(8*sizeof(uint32_t));

  //allocate xmp integers
  xmpHandle_t handle;
  xmpExecutionPolicy_t policy;
  xmpIntegers_t x_a, x_b;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyCreate(handle,&policy));
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_a,aP,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_b,bP,bN));

  //allocate memory on hosts
  uint32_t *h_a, *h_b;
  int32_t *h_res, *d_res;

  h_a=(uint32_t*)malloc(sizeof(uint32_t)*aN*alimbs);
  h_b=(uint32_t*)malloc(sizeof(uint32_t)*bN*blimbs);
  h_res=(int32_t*)malloc(sizeof(uint32_t)*N);
  d_res=(int32_t*)malloc(sizeof(uint32_t)*N);

  //intitialize resutls to 0 as gmp may not write all of the bits 
  memset(h_res,0,sizeof(uint32_t)*N);
  memset(d_res,0,sizeof(uint32_t)*N);

  //allocate gmp integers
  mpz_t *g_a, *g_b;

  g_a=(mpz_t*)malloc(sizeof(mpz_t)*aN);
  g_b=(mpz_t*)malloc(sizeof(mpz_t)*bN);

  for(int i=0;i<aN;i++) mpz_init(g_a[i]);
  for(int i=0;i<bN;i++) mpz_init(g_b[i]);

  //generate random data on host
  srand(0);

  for(int i=0;i<aN*alimbs;i++) h_a[i]=rand32();
  for(int i=0;i<bN*blimbs;i++) h_b[i]=rand32();

  //import to xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_a,alimbs,-1,sizeof(uint32_t),-1,0,h_a,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_b,blimbs,-1,sizeof(uint32_t),-1,0,h_b,bN));
  
  uint32_t *a_indices, *b_indices;
  a_indices=(uint32_t*)malloc(aN*sizeof(uint32_t));
  b_indices=(uint32_t*)malloc(bN*sizeof(uint32_t));

  for(int i=0;i<aN;i++) a_indices[i]=i;
  for(int i=0;i<bN;i++) b_indices[i]=i;

  //generate random indices for a, b and c
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<aN;i++) 
      std::swap(a_indices[i],a_indices[rand32()%aN]);
    for(int i=0;i<bN;i++) 
      std::swap(b_indices[i],b_indices[rand32()%bN]);
  }
  
  //set indices in xmp
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,a_indices,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,1,b_indices,bN));

  //Enable policy for dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));

  //perform operation the device
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCmpAsync(handle,d_res,x_a,x_b,N));

  //disable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));
 
  //import to mpz
  #pragma omp parallel for
  for(int i=0;i<aN;i++) mpz_import(g_a[i],alimbs,-1,sizeof(uint32_t),-1,0,&h_a[i*alimbs]);
  #pragma omp parallel for
  for(int i=0;i<bN;i++) mpz_import(g_b[i],blimbs,-1,sizeof(uint32_t),-1,0,&h_b[i*blimbs]);
  
  //perform operation on the host
  #pragma omp parallel for
  for(int i=0;i<N;i++) {
    mpz_t *aa, *bb;
    aa= &g_a[a_indices[i%aN] % aN]; 
    bb= &g_b[b_indices[i%bN] % bN]; 
    h_res[i]=mpz_cmp(*aa,*bb);
  }

  ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());
  
  //compare results
  for(int i=0;i<N;i++) {
    ASSERT_EQ(sgn(h_res[i]),sgn(d_res[i]));
  }

  //clean up
  for(int i=0;i<aN;i++) mpz_clear(g_a[i]);
  for(int i=0;i<bN;i++) mpz_clear(g_b[i]);
  
  free(a_indices); free(b_indices);

  free(h_a); free(h_b); free(h_res); free(d_res);
  free(g_a); free(g_b); 

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_a));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_b));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyDestroy(handle,policy));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST_P(PopcTest,opTests) {
  PopcParams p=GetParam();
  uint32_t aN=p.N1, N=p.N;
  uint32_t aP=p.P1;
  uint32_t alimbs=aP/(8*sizeof(uint32_t));

  //allocate xmp integers
  xmpHandle_t handle;
  xmpExecutionPolicy_t policy;
  xmpIntegers_t x_a;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyCreate(handle,&policy));
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_a,aP,aN));

  //allocate memory on hosts
  uint32_t *h_a;
  uint32_t *h_res, *d_res;

  h_a=(uint32_t*)malloc(sizeof(uint32_t)*aN*alimbs);
  h_res=(uint32_t*)malloc(sizeof(uint32_t)*N);
  d_res=(uint32_t*)malloc(sizeof(uint32_t)*N);

  //intitialize resutls to 0 as gmp may not write all of the bits 
  memset(h_res,0,sizeof(uint32_t)*N);
  memset(d_res,0,sizeof(uint32_t)*N);

  //allocate gmp integers
  mpz_t *g_a;
  
  g_a=(mpz_t*)malloc(sizeof(mpz_t)*aN);

  for(int i=0;i<aN;i++) mpz_init(g_a[i]);

  //generate random data on host
  srand(0);

  for(int i=0;i<aN*alimbs;i++) h_a[i]=rand32();

  //import to xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_a,alimbs,-1,sizeof(uint32_t),-1,0,h_a,aN));
  
  uint32_t *a_indices;
  a_indices=(uint32_t*)malloc(aN*sizeof(uint32_t));

  for(int i=0;i<aN;i++) a_indices[i]=i;

  //generate random indices for a, b and c
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<aN;i++) 
      std::swap(a_indices[i],a_indices[rand32()%aN]);
  }
  
  //set indices in xmp
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,a_indices,aN));

  //Enable policy for dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));

  //perform operation the device
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersPopcAsync(handle,d_res,x_a,N));

  //disable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));
 
  //import to mpz
  #pragma omp parallel for
  for(int i=0;i<aN;i++) mpz_import(g_a[i],alimbs,-1,sizeof(uint32_t),-1,0,&h_a[i*alimbs]);
  
  //perform operation on the host
  #pragma omp parallel for
  for(int i=0;i<N;i++) {
    mpz_t *aa;
    aa= &g_a[a_indices[i%aN] % aN]; 
    h_res[i]=mpz_popcount(*aa);
  }

  ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());
  
  //compare results
  for(int i=0;i<N;i++) {
    ASSERT_EQ(h_res[i],d_res[i]);
  }

  //clean up
  for(int i=0;i<aN;i++) mpz_clear(g_a[i]);
  
  free(a_indices);

  free(h_a); free(h_res); free(d_res);
  free(g_a); 

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_a));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyDestroy(handle,policy));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

TEST_P(SubTest,opTests) {
  SubParams p=GetParam();
  uint32_t cN=p.N1, aN=p.N2, bN=p.N3, N=p.N;
  uint32_t cP=p.P1, aP=p.P2, bP=p.P3;
  uint32_t climbs=cP/(8*sizeof(uint32_t));
  uint32_t alimbs=aP/(8*sizeof(uint32_t));
  uint32_t blimbs=bP/(8*sizeof(uint32_t));

  //allocate xmp integers
  xmpHandle_t handle;
  xmpExecutionPolicy_t policy;
  xmpIntegers_t x_c, x_a, x_b;
  ASSERT_EQ(xmpErrorSuccess,xmpHandleCreate(&handle));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyCreate(handle,&policy));
  
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_c,cP,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_a,aP,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersCreate(handle,&x_b,bP,bN));

  //allocate memory on hosts
  uint32_t *h_a, *h_b;
  uint32_t *h_res, *d_res;
  h_a=(uint32_t*)malloc(sizeof(uint32_t)*aN*alimbs);
  h_b=(uint32_t*)malloc(sizeof(uint32_t)*bN*blimbs);
  h_res=(uint32_t*)malloc(sizeof(uint32_t)*cN*climbs);
  d_res=(uint32_t*)malloc(sizeof(uint32_t)*cN*climbs);

  //intitialize resutls to 0 as gmp may not write all of the bits 
  memset(h_res,0,sizeof(uint32_t)*cN*climbs);
  memset(d_res,0,sizeof(uint32_t)*cN*climbs);

  //allocate gmp integers
  mpz_t *g_c, *g_a, *g_b, *g_t;

  g_c=(mpz_t*)malloc(sizeof(mpz_t)*cN);
  g_a=(mpz_t*)malloc(sizeof(mpz_t)*aN);
  g_b=(mpz_t*)malloc(sizeof(mpz_t)*bN);
  g_t=(mpz_t*)malloc(sizeof(mpz_t)*N);

  for(int i=0;i<cN;i++) mpz_init(g_c[i]);
  for(int i=0;i<aN;i++) mpz_init(g_a[i]);
  for(int i=0;i<bN;i++) mpz_init(g_b[i]);
  for(int i=0;i<N;i++) mpz_init(g_t[i]);

  //generate random data on host
  srand(0);

  for(int i=0;i<aN*alimbs;i++) h_a[i]=rand32();
  for(int i=0;i<bN*blimbs;i++) h_b[i]=rand32();

  //import to xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_a,alimbs,-1,sizeof(uint32_t),-1,0,h_a,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersImportAsync(handle,x_b,blimbs,-1,sizeof(uint32_t),-1,0,h_b,bN));
  
  uint32_t *a_indices, *b_indices, *c_indices;
  a_indices=(uint32_t*)malloc(aN*sizeof(uint32_t));
  b_indices=(uint32_t*)malloc(bN*sizeof(uint32_t));
  c_indices=(uint32_t*)malloc(cN*sizeof(uint32_t));

  for(int i=0;i<aN;i++) a_indices[i]=i;
  for(int i=0;i<bN;i++) b_indices[i]=i;
  for(int i=0;i<cN;i++) c_indices[i]=i;

  //generate random indices for a, b and c
  //shuffle indices
  for(int j=0;j<10;j++) {
    for(int i=0;i<aN;i++) 
      std::swap(a_indices[i],a_indices[rand32()%aN]);
    for(int i=0;i<bN;i++) 
      std::swap(b_indices[i],b_indices[rand32()%bN]);
    for(int i=0;i<cN;i++) 
      std::swap(c_indices[i],c_indices[rand32()%cN]);
  }
 
  //set indices in xmp
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,0,c_indices,cN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,1,a_indices,aN));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicySetIndices(handle,policy,2,b_indices,bN));

  //Enable policy for dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));

  //perform operation the device
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersSubAsync(handle,x_c,x_a,x_b,N));

  //disable dynamic indexing
  ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));
 
  //export from xmp
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,NULL,-1,sizeof(uint32_t),-1,0,x_c,cN));
  
  //import to mpz
  #pragma omp parallel for
  for(int i=0;i<aN;i++) mpz_import(g_a[i],alimbs,-1,sizeof(uint32_t),-1,0,&h_a[i*alimbs]);
  #pragma omp parallel for
  for(int i=0;i<bN;i++) mpz_import(g_b[i],blimbs,-1,sizeof(uint32_t),-1,0,&h_b[i*blimbs]);
  
  //perform operation on the host
  #pragma omp parallel for
  for(int i=0;i<N;i++) {
    mpz_t *cc=&g_c[c_indices[i]], *aa, *bb, *tt=&g_t[i];
    aa= &g_a[a_indices[i%aN] % aN]; 
    bb= &g_b[b_indices[i%bN] % bN]; 
    mpz_sub(*cc,*aa,*bb);
    if(mpz_sgn(*cc)==-1) {
      mpz_set_ui(*tt,1);
      mpz_mul_2exp(*tt,*tt,cP);
      mpz_add(*cc,*cc,*tt);
    }
  }

  //export from gmp
  #pragma omp parallel for
  for(int i=0;i<N;i++) mpz_export(&h_res[c_indices[i]*climbs],NULL,-1,sizeof(uint32_t),-1,0,g_c[c_indices[i]]);
  
  ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());
  
  //compare results
  for(int i=0;i<N;i++) {
    for(int j=0;j<climbs;j++) {
      ASSERT_EQ(h_res[c_indices[i]*climbs+j],d_res[c_indices[i]*climbs+j]);
    }
  }

  if(cN==aN && cP==aP) {
    //Enable policy for dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,policy));
    
    //perform operation the device
    ASSERT_EQ(xmpErrorSuccess,xmpIntegersSubAsync(handle,x_a,x_a,x_b,N));
  
    //disable dynamic indexing
    ASSERT_EQ(xmpErrorSuccess,xmpHandleSetExecutionPolicy(handle,NULL));
 
    //export from xmp
    ASSERT_EQ(xmpErrorSuccess,xmpIntegersExportAsync(handle,d_res,NULL,-1,sizeof(uint32_t),-1,0,x_a,aN));
  
    ASSERT_EQ(cudaSuccess,cudaDeviceSynchronize());
    //compare results
    for(int i=0;i<N;i++) {
      for(int j=0;j<climbs;j++) {
        ASSERT_EQ(h_res[c_indices[i]*climbs+j],d_res[c_indices[i]*climbs+j]);
      }
    }
  }

  //clean up
  for(int i=0;i<cN;i++) mpz_clear(g_c[i]);
  for(int i=0;i<aN;i++) mpz_clear(g_a[i]);
  for(int i=0;i<bN;i++) mpz_clear(g_b[i]);
  for(int i=0;i<N;i++) mpz_clear(g_t[i]);
  
  free(a_indices); free(b_indices); free(c_indices); 

  free(h_a); free(h_b); free(h_res); free(d_res);
  free(g_c); free(g_a); free(g_b), free(g_t); 

  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_c));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_a));
  ASSERT_EQ(xmpErrorSuccess,xmpIntegersDestroy(handle,x_b));
  ASSERT_EQ(xmpErrorSuccess,xmpExecutionPolicyDestroy(handle,policy));
  ASSERT_EQ(xmpErrorSuccess,xmpHandleDestroy(handle));
}

int main(int argc, char **argv) {

  int nDevice=-1;
  cudaGetDeviceCount(&nDevice);

  if(nDevice<=0) {
    printf("Error no cuda device found.  Aborting tests\n");
    exit(EXIT_FAILURE);
  }
  
  printf("Running all tests\n");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

int N=10000;
int M=100;
INSTANTIATE_TEST_CASE_P(ShfTests, ShfTest, ::testing::Values( 
        ShfParams(32,32,N,N,N,N),
        ShfParams(64,64,N,N,N,N),
        ShfParams(128,128,N,N,N,N),
        ShfParams(256,256,N,N,N,N),
        ShfParams(512,512,N,N,N,N),
        
        ShfParams(64,32,N,N,N,N),
        ShfParams(128,64,N,N,N,N),
        ShfParams(256,128,N,N,N,N),
        ShfParams(512,256,N,N,N,N),
      
        ShfParams(32,32,N,M,N,N),
        ShfParams(64,64,N,M,N,N),
        ShfParams(128,128,N,M,N,N),
        ShfParams(256,256,N,M,N,N),
        ShfParams(512,512,N,M,N,N),
      
        ShfParams(160,160,N,N,N,N),
        ShfParams(192,192,N,N,N,N),
        ShfParams(224,224,N,N,N,N),
        ShfParams(288,288,N,N,N,N),
        ShfParams(320,320,N,N,N,N),
        ShfParams(352,352,N,N,N,N),
        ShfParams(384,384,N,N,N,N)
      ));
INSTANTIATE_TEST_CASE_P(SqrTests, SqrTest, ::testing::Values( 
        SqrParams(32,32,N,N,N),
        SqrParams(64,64,N,N,N),
        SqrParams(128,128,N,N,N),
        SqrParams(256,256,N,N,N),
        SqrParams(512,512,N,N,N),
        
        SqrParams(64,32,N,N,N),
        SqrParams(128,64,N,N,N),
        SqrParams(256,128,N,N,N),
        SqrParams(512,256,N,N,N),
      
        SqrParams(32,32,N,M,N),
        SqrParams(64,64,N,M,N),
        SqrParams(128,128,N,M,N),
        SqrParams(256,256,N,M,N),
        SqrParams(512,512,N,M,N),
      
        SqrParams(160,160,N,N,N),
        SqrParams(192,192,N,N,N),
        SqrParams(224,224,N,N,N),
        SqrParams(288,288,N,N,N),
        SqrParams(320,320,N,N,N),
        SqrParams(352,352,N,N,N),
        SqrParams(384,384,N,N,N)
      ));
INSTANTIATE_TEST_CASE_P(NotTests, NotTest, ::testing::Values( 
        NotParams(32,32,N,N,N),
        NotParams(64,64,N,N,N),
        NotParams(128,128,N,N,N),
        NotParams(256,256,N,N,N),
        NotParams(512,512,N,N,N),
        
        NotParams(64,32,N,N,N),
        NotParams(128,64,N,N,N),
        NotParams(256,128,N,N,N),
        NotParams(512,256,N,N,N),
      
        NotParams(32,32,N,M,N),
        NotParams(64,64,N,M,N),
        NotParams(128,128,N,M,N),
        NotParams(256,256,N,M,N),
        NotParams(512,512,N,M,N),
      
      
        NotParams(160,160,N,N,N),
        NotParams(192,192,N,N,N),
        NotParams(224,224,N,N,N),
        NotParams(288,288,N,N,N),
        NotParams(320,320,N,N,N),
        NotParams(352,352,N,N,N),
        NotParams(384,384,N,N,N)
      ));
INSTANTIATE_TEST_CASE_P(CmpTests, CmpTest, ::testing::Values( 
        CmpParams(32,32,N,N,N),
        CmpParams(64,64,N,N,N),
        CmpParams(128,128,N,N,N),
        CmpParams(256,256,N,N,N),
        CmpParams(512,512,N,N,N),
        
        CmpParams(64,32,N,N,N),
        CmpParams(128,64,N,N,N),
        CmpParams(256,128,N,N,N),
        CmpParams(512,256,N,N,N),
      
        CmpParams(32,32,N,M,N),
        CmpParams(64,64,N,M,N),
        CmpParams(128,128,N,M,N),
        CmpParams(256,256,N,M,N),
        CmpParams(512,512,N,M,N),
      
      
        CmpParams(160,160,N,N,N),
        CmpParams(192,192,N,N,N),
        CmpParams(224,224,N,N,N),
        CmpParams(288,288,N,N,N),
        CmpParams(320,320,N,N,N),
        CmpParams(352,352,N,N,N),
        CmpParams(384,384,N,N,N)
      ));
INSTANTIATE_TEST_CASE_P(PopcTests, PopcTest, ::testing::Values( 
        PopcParams(32,N,N),
        PopcParams(64,N,N),
        PopcParams(128,N,N),
        PopcParams(256,N,N),
        PopcParams(512,N,N),
        
        PopcParams(160,N,N),
        PopcParams(192,N,N),
        PopcParams(224,N,N),
        PopcParams(288,N,N),
        PopcParams(320,N,N),
        PopcParams(352,N,N),
        PopcParams(384,N,N)
      ));
INSTANTIATE_TEST_CASE_P(AndTests, genericTwoInOneOutTest, ::testing::Values( 
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,32,32,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,64,64,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,128,128,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,256,256,256,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,512,512,512,N,N,N,N),
        
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,64,64,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,128,128,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,256,256,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,512,512,256,N,N,N,N),
      
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,32,32,32,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,64,64,64,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,128,128,128,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,256,256,256,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,512,512,512,N,N,M,N),
      
      
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,160,160,160,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,192,192,192,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,224,224,224,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,288,288,288,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,320,320,320,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,352,352,352,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAndAsync,mpz_and,384,384,384,N,N,N,N)
      ));
INSTANTIATE_TEST_CASE_P(IorTests, genericTwoInOneOutTest, ::testing::Values( 
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,32,32,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,64,64,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,128,128,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,256,256,256,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,512,512,512,N,N,N,N),
        
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,64,64,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,128,128,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,256,256,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,512,512,256,N,N,N,N),
      
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,32,32,32,N,N,M,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,64,64,64,N,N,M,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,128,128,128,N,N,M,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,256,256,256,N,N,M,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,512,512,512,N,N,M,N),
      
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,160,160,160,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,192,192,192,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,224,224,224,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,288,288,288,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,320,320,320,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,352,352,352,N,N,N,N),
        TwoInOneOutParams(xmpIntegersIorAsync,mpz_ior,384,384,384,N,N,N,N)
      ));
INSTANTIATE_TEST_CASE_P(XorTests, genericTwoInOneOutTest, ::testing::Values( 
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,32,32,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,64,64,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,128,128,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,256,256,256,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,512,512,512,N,N,N,N),
        
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,64,64,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,128,128,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,256,256,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,512,512,256,N,N,N,N),
      
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,32,32,32,N,N,M,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,64,64,64,N,N,M,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,128,128,128,N,N,M,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,256,256,256,N,N,M,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,512,512,512,N,N,M,N),
      
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,160,160,160,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,192,192,192,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,224,224,224,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,288,288,288,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,320,320,320,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,352,352,352,N,N,N,N),
        TwoInOneOutParams(xmpIntegersXorAsync,mpz_xor,384,384,384,N,N,N,N)
      ));



INSTANTIATE_TEST_CASE_P(AddTests, genericTwoInOneOutTest, ::testing::Values( 
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,32+32,32,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,64+32,64,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,128+32,128,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,256+32,256,256,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,512+32,512,512,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,1024+32,1024,1024,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,2048+32,2048,2048,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,4096+32,4096,4096,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,8192+32,8192,8192,N,N,N,N),
      
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,32+32,32,32,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,64+32,64,64,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,128+32,128,128,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,256+32,256,256,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,512+32,512,512,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,1024+32,1024,1024,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,2048+32,2048,2048,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,4096+32,4096,4096,N,N,M,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,8192+32,8192,8192,N,N,M,N),
      
      
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,160+32,160,160,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,192+32,192,192,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,224+32,224,224,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,288+32,288,288,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,320+32,320,320,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,352+32,352,352,N,N,N,N),
        TwoInOneOutParams(xmpIntegersAddAsync,mpz_add,384+32,384,384,N,N,N,N)
      ));

  
 INSTANTIATE_TEST_CASE_P(SubTests, SubTest, ::testing::Values( 
      SubParams(32,32,32,N,N,N,N),
      SubParams(64,64,64,N,N,N,N),
      SubParams(128,128,128,N,N,N,N),
      SubParams(256,256,256,N,N,N,N),
      SubParams(512,512,512,N,N,N,N),
      SubParams(1024,1024,1024,N,N,N,N),
      SubParams(2048,2048,2048,N,N,N,N),
      SubParams(4096,4096,4096,N,N,N,N),
      SubParams(1024,1024,1024,N,N,N,N),
      
      SubParams(32,32,32,N,N,M,N),
      SubParams(64,64,64,N,N,M,N),
      SubParams(128,128,128,N,N,M,N),
      SubParams(256,256,256,N,N,M,N),
      SubParams(512,512,512,N,N,M,N),
      SubParams(1024,1024,1024,N,N,M,N),
      SubParams(2048,2048,2048,N,N,M,N),
      SubParams(4096,4096,4096,N,N,M,N),
      SubParams(1024,1024,1024,N,N,M,N),
      
      
      SubParams(160,160,160,N,N,N,N),
      SubParams(192,192,192,N,N,N,N),
      SubParams(224,224,224,N,N,N,N),
      SubParams(288,288,288,N,N,N,N),
      SubParams(320,320,320,N,N,N,N),
      SubParams(352,352,352,N,N,N,N),
      SubParams(384,384,384,N,N,N,N)
      ));
 

INSTANTIATE_TEST_CASE_P(MulTests, genericTwoInOneOutTest, ::testing::Values( 
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*32,32,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*64,64,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*128,128,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*256,256,256,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*512,512,512,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*1024,1024,1024,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*2048,2048,2048,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*4096,4096,4096,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*8192,8192,8192,N,N,N,N),
      
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*32,32,32,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*64,64,64,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*128,128,128,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*256,256,256,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*512,512,512,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*1024,1024,1024,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*2048,2048,2048,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*4096,4096,4096,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*8192,8192,8192,N,N,M,N),
      
      
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*160,160,160,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*192,192,192,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*224,224,224,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*288,288,288,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*320,320,320,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*352,352,352,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2*384,384,384,N,N,N,N),
        
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,32,32,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,64,64,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,128,128,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,256,256,256,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,512,512,512,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,1024,1024,1024,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2048,2048,2048,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,4096,4096,4096,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,8192,8192,8192,N,N,N,N),
      
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,32,32,32,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,64,64,64,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,128,128,128,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,256,256,256,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,512,512,512,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,1024,1024,1024,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,2048,2048,2048,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,4096,4096,4096,N,N,M,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,8192,8192,8192,N,N,M,N),
      
      
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,160,160,160,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,192,192,192,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,224,224,224,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,288,288,288,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,320,320,320,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,352,352,352,N,N,N,N),
        TwoInOneOutParams(xmpIntegersMulAsync,mpz_mul,384,384,384,N,N,N,N)
      ));
 
INSTANTIATE_TEST_CASE_P(DivTests, genericTwoInOneOutTest, ::testing::Values( 
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,32,32,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,64,64,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,128,128,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,256,256,256,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,512,512,512,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,1024,1024,1024,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,2048,2048,2048,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,4096,4096,4096,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,8192,8192,8192,N,N,M,N),
      
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,32,32,32,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,64,64,64,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,128,128,128,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,256,256,256,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,512,512,512,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,1024,1024,1024,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,2048,2048,2048,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,4096,4096,4096,N,N,M,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,8192,8192,8192,N,N,M,N),
      
      
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,160,160,160,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,192,192,192,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,224,224,224,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,288,288,288,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,320,320,320,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,352,352,352,N,N,N,N),
        TwoInOneOutParams(xmpIntegersDivAsync,mpz_div,384,384,384,N,N,N,N)
      ));

INSTANTIATE_TEST_CASE_P(ModTests, genericTwoInOneOutTest, ::testing::Values( 
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,32,32,32,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,64,64,64,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,128,128,128,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,256,256,256,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,512,512,512,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,1024,1024,1024,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,2048,2048,2048,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,4096,4096,4096,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,8192,8192,8192,N,N,N,N),
      
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,32,32,32,N,N,M,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,64,64,64,N,N,M,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,128,128,128,N,N,M,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,256,256,256,N,N,M,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,512,512,512,N,N,M,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,1024,1024,1024,N,N,M,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,2048,2048,2048,N,N,M,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,4096,4096,4096,N,N,M,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,8192,8192,8192,N,N,M,N),
      
      
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,160,160,160,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,192,192,192,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,224,224,224,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,288,288,288,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,320,320,320,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,352,352,352,N,N,N,N),
        TwoInOneOutParams(xmpIntegersModAsync,mpz_mod,384,384,384,N,N,N,N)
      ));

INSTANTIATE_TEST_CASE_P(DivModTests, genericTwoInTwoOutTest, ::testing::Values( 
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,32,32,32,32,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,64,64,64,64,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,128,128,128,128,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,256,256,256,256,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,512,512,512,512,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,1024,1024,1024,1024,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,2048,2048,2048,2048,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,4096,4096,4096,4096,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,8192,8192,8192,8192,N,N,N,N,N),
        
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,32,32,32,32,N,N,N,M,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,64,64,64,64,N,N,N,M,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,128,128,128,128,N,N,N,M,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,256,256,256,256,N,N,N,M,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,512,512,512,512,N,N,N,M,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,1024,1024,1024,1024,N,N,N,M,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,2048,2048,2048,2048,N,N,N,M,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,4096,4096,4096,4096,N,N,N,M,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,8192,8192,8192,8192,N,N,N,M,N),
      
      
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,160,160,160,160,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,192,192,192,192,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,224,224,224,224,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,288,288,288,288,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,320,320,320,320,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,352,352,352,352,N,N,N,N,N),
        TwoInTwoOutParams(xmpIntegersDivModAsync,mpz_divmod,384,384,384,384,N,N,N,N,N)
      ));
 
INSTANTIATE_TEST_CASE_P(DistributedPowmTests, PowmTest, ::testing::Values( 
      PowmParams(128,128,128,128,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(128,128,128,128,N,N,2*N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(128,128,128,128,N,N,1,1,N,xmpAlgorithmDistributedMP),
      PowmParams(128,128,128,128,N,N,M,M,N,xmpAlgorithmDistributedMP),
      PowmParams(128,128,128,128,N,N,M,1,N,xmpAlgorithmDistributedMP),
      PowmParams(128,128,128,128,N,1,M,N,N,xmpAlgorithmDistributedMP),
      PowmParams(128,128,128,128,N,N,N,N,M,xmpAlgorithmDistributedMP),
      PowmParams(128,128,128,128,N,N,N,N,N,xmpAlgorithmDistributedMP),
      
      PowmParams(32,32,32,32,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(32,32,32,32,N,N,1,1,N,xmpAlgorithmDistributedMP),
      PowmParams(64,64,64,64,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(64,64,64,64,N,N,1,1,N,xmpAlgorithmDistributedMP),
      PowmParams(128,128,128,128,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(128,128,128,128,N,N,1,1,N,xmpAlgorithmDistributedMP),
      PowmParams(256,256,256,256,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(256,256,256,256,N,N,1,1,N,xmpAlgorithmDistributedMP),
      PowmParams(512,512,512,512,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(512,512,512,512,N,N,1,1,N,xmpAlgorithmDistributedMP),
      
      PowmParams(1056,1056,1056,1056,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(2080,2080,2080,2080,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(6144,6144,6144,6144,N,N,1,1,N,xmpAlgorithmDistributedMP),
      PowmParams(6144,6144,6144,6144,N,N,N,N,N,xmpAlgorithmDistributedMP),

      PowmParams(1024,1024,1024,1024,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(1024,1024,1024,1024,N,N,1,1,N,xmpAlgorithmDistributedMP),
      PowmParams(2048,2048,2048,2048,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(2048,2048,2048,2048,N,N,1,1,N,xmpAlgorithmDistributedMP),
      PowmParams(4096,4096,4096,4096,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(4096,4096,4096,4096,N,N,1,1,N,xmpAlgorithmDistributedMP),
      PowmParams(8192,8192,8192,8192,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(8192,8192,8192,8192,N,N,1,1,N,xmpAlgorithmDistributedMP),
      

      PowmParams(160,160,160,160,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(192,192,192,192,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(224,224,224,224,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(288,288,288,288,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(320,320,320,320,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(352,352,352,352,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(384,384,384,384,N,N,N,N,N,xmpAlgorithmDistributedMP),
      PowmParams(544,544,544,544,N,N,N,N,N,xmpAlgorithmDistributedMP)
      
      ));
 INSTANTIATE_TEST_CASE_P(RegMPPowmTests, PowmTest, ::testing::Values( 
      PowmParams(128,128,128,128,N,N,2*N,N,N,xmpAlgorithmRegMP),
      PowmParams(128,128,128,128,N,N,1,1,N,xmpAlgorithmRegMP),
      PowmParams(128,128,128,128,N,N,M,M,N,xmpAlgorithmRegMP),
      PowmParams(128,128,128,128,N,N,M,1,N,xmpAlgorithmRegMP),
      PowmParams(128,128,128,128,N,1,M,N,N,xmpAlgorithmRegMP),
      PowmParams(128,128,128,128,N,N,N,N,M,xmpAlgorithmRegMP),
      
      PowmParams(32,32,32,32,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(32,32,32,32,N,N,1,1,N,xmpAlgorithmRegMP),
      PowmParams(64,64,64,64,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(64,64,64,64,N,N,1,1,N,xmpAlgorithmRegMP),
      PowmParams(128,128,128,128,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(128,128,128,128,N,N,1,1,N,xmpAlgorithmRegMP),
      PowmParams(256,256,256,256,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(256,256,256,256,N,N,1,1,N,xmpAlgorithmRegMP),
      PowmParams(512,512,512,512,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(512,512,512,512,N,N,1,1,N,xmpAlgorithmRegMP),

      PowmParams(160,160,160,160,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(192,192,192,192,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(224,224,224,224,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(288,288,288,288,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(320,320,320,320,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(352,352,352,352,N,N,N,N,N,xmpAlgorithmRegMP),
      PowmParams(384,384,384,384,N,N,N,N,N,xmpAlgorithmRegMP)
      
      ));

 INSTANTIATE_TEST_CASE_P(DigitMPPowmTests, PowmTest, ::testing::Values( 
      PowmParams(1056,1056,1056,1056,N,N,N,N,N,xmpAlgorithmDigitMP),
      PowmParams(2080,2080,2080,2080,N,N,N,N,N,xmpAlgorithmDigitMP),
      PowmParams(6144,6144,6144,6144,N,N,1,1,N,xmpAlgorithmDigitMP),
      PowmParams(6144,6144,6144,6144,N,N,N,N,N,xmpAlgorithmDigitMP),

      PowmParams(512,512,512,512,N,N,N,N,N,xmpAlgorithmDigitMP),
      PowmParams(512,512,512,512,N,N,1,1,N,xmpAlgorithmDigitMP),
      PowmParams(1024,1024,1024,1024,N,N,N,N,N,xmpAlgorithmDigitMP),
      PowmParams(1024,1024,1024,1024,N,N,1,1,N,xmpAlgorithmDigitMP),
      PowmParams(2048,2048,2048,2048,N,N,N,N,N,xmpAlgorithmDigitMP),
      PowmParams(2048,2048,2048,2048,N,N,1,1,N,xmpAlgorithmDigitMP),
      PowmParams(4096,4096,4096,4096,N,N,N,N,N,xmpAlgorithmDigitMP),
      PowmParams(4096,4096,4096,4096,N,N,1,1,N,xmpAlgorithmDigitMP),
      PowmParams(8192,8192,8192,8192,N,N,N,N,N,xmpAlgorithmDigitMP),
      PowmParams(8192,8192,8192,8192,N,N,1,1,N,xmpAlgorithmDigitMP),
 
      PowmParams(544,544,544,544,N,N,N,N,N,xmpAlgorithmDigitMP)
      

      ));
