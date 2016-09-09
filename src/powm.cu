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

#include <operators.h>
#include "powm_operators.h"

//prevent instantiation of these here....
extern template xmpError_t internalPowmRegMP<128,6,4,0,0>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmRegMP<128,6,8,0,0>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmRegMP<128,6,12,0,0>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmRegMP<128,6,16,0,0>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,4,1>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,2,2>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,8,1>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,4,2>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,2,4>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,4,3>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,2,6>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,16,1>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,8,2>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,4,4>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,2,8>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,8,3>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,4,6>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,32,1>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,16,2>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,8,4>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,4,8>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,16,3>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,8,6>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,32,2>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,16,4>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,8,8>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,32,3>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,16,6>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,32,4>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,16,8>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,32,6>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmWarpDistributedMP<128,6,32,8>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t* );
extern template xmpError_t internalPowmDigitMP<128,6,8>(xmpHandle_t,xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, const xmpIntegers_t, uint32_t, uint32_t, xmpLimb_t*, uint32_t*, uint32_t*);

uint32_t xmpPowmPrecisions[]={128,256,512,768,1024,1536,2048,3072,4096,6144,8192};
//uint32_t xmpPowmPrecisions[]={128,256,512,768,1024,1536,2048,3072,4096,6144};
uint32_t xmpPowmPrecisionsCount = sizeof(xmpPowmPrecisions)/sizeof(uint32_t);

xmpPowmAlgorithm xmpPowmAlgorithms[] = {
  //ThreeN
  xmpPowmAlgorithm(xmpAlgorithmRegMP,(xmpPowmFunc)internalPowmRegMP<128,6,4,0,0>,1,128),
  xmpPowmAlgorithm(xmpAlgorithmRegMP,(xmpPowmFunc)internalPowmRegMP<128,6,8,0,0>,129,256),
  xmpPowmAlgorithm(xmpAlgorithmRegMP,(xmpPowmFunc)internalPowmRegMP<128,6,12,0,0>,257,384),
  xmpPowmAlgorithm(xmpAlgorithmRegMP,(xmpPowmFunc)internalPowmRegMP<128,6,16,0,0>,385,512),

  //Distributed
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,4,1>,1,128),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,2,2>,1,128),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,8,1>,129,256),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,4,2>,129,256),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,2,4>,129,256),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,4,3>,257,384),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,2,6>,257,384),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,16,1>,385,512),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,8,2>,385,512),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,4,4>,385,512),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,2,8>,385,512),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,8,3>,513,768),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,4,6>,513,768),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,32,1>,767,1024),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,16,2>,767,1024),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,8,4>,767,1024),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,4,8>,767,1024),
#if 1
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,16,3>,1025,1536),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,8,6>,1025,1536),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,32,2>,1537,2048),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,16,4>,1537,2048),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,8,8>,1537,2048),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,32,3>,2049,3072),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,16,6>,2049,3072),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,32,4>,3073,4096),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,16,8>,3073,4096),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,32,6>,4097,6144),
  xmpPowmAlgorithm(xmpAlgorithmDistributedMP,(xmpPowmFunc)internalPowmWarpDistributedMP<128,6,32,8>,6145,8192),
#endif
  //Digitized
  xmpPowmAlgorithm(xmpAlgorithmDigitMP,(xmpPowmFunc)internalPowmDigitMP<128,6,8>,512,uint32_t(-1)),
};
uint32_t xmpPowmAlgorithmsCount = sizeof(xmpPowmAlgorithms)/sizeof(xmpPowmAlgorithm);

struct LaunchParameters
{
  int32_t alg_index;
  uint32_t count;
  float latency;
  LaunchParameters(int32_t alg_index, uint32_t count, float latency) : alg_index(alg_index), count(count), latency(latency) {}
  LaunchParameters() : alg_index(-1), count(0), latency(99999999999) {}
};

#include "tune/tune_maxwell.h"
#include "tune/tune_kepler.h"

LaunchParameters getFastestPowmLaunch(xmpHandle_t handle, xmpAlgorithm_t alg, uint32_t max_waves, uint32_t count, const Latency *lookup, uint32_t tbl_count) {
  LaunchParameters params;
  float max_throughput=0;
  params.alg_index=-1;

  int i=0;
  while(i<tbl_count) {
    
    Latency lat = lookup[i];
    uint32_t alg_index=lat.alg_index;
    if(alg!=xmpAlgorithmDefault && alg!=xmpPowmAlgorithms[lat.alg_index].alg) {
      //skip this algorithm
      while(i<tbl_count && alg_index==lookup[i].alg_index) 
        i++;
      continue;
    }

    uint32_t lcount = lat.instances_per_sm*handle->smCount;   //instances per wave of blocks
    float latency;

    if(count<=lcount) {
      //no splitting, compute all in a single wave

      //search for smallest size that is greater than or equal to count
      while(i+1<tbl_count && alg_index==lookup[i+1].alg_index && lookup[i+1].instances_per_sm*handle->smCount>=count) {
        i++;
      }
      
      latency = lookup[i].latency;
      lcount = count;

    } else {
      //possible splitting
      uint32_t WAVES=count/lcount;    //number of waves to launch
      WAVES=MIN(WAVES,max_waves);
      lcount = WAVES*lcount;          //number of instances to launch
      uint32_t tail_count=count-lcount;
      //compute cost of tail
      
      //if tail exists
      if(tail_count>0) {
        LaunchParameters tail_launch= getFastestPowmLaunch(handle,alg,max_waves,tail_count,lookup,tbl_count);

        //if algorithm for the tail is the same combine into a single call
        if(WAVES<max_waves && tail_launch.alg_index==lat.alg_index) {
          lcount=count;
        }
        latency = WAVES*lat.latency+tail_launch.latency;
      } else {
        latency = WAVES*lat.latency;
      }
    } 
    
    float throughput = count/latency;

    //maximize throughput across all algorithms
    if(throughput>max_throughput) {
      params.alg_index=alg_index;
      params.count=lcount;
      params.latency=latency;
      max_throughput=throughput;
    }
    //printf("alg: %d, count: %d, lcount: %d, throughput: %f, latency: %f, max_throughput: %f\n",params.alg_index,count,lcount,throughput,latency,max_throughput);
    
    //skip remaining entries for this algorithm
    while(i<tbl_count && alg_index==lookup[i].alg_index) 
      i++;
  }
  return params;
}

LaunchParameters getPowmLaunchParameters(xmpHandle_t handle, uint32_t precision, uint32_t count, xmpAlgorithm_t alg ) {
  const Latency * lookup;
  uint32_t tbl_count;
  

  int idx;
  if(precision>xmpPowmPrecisions[xmpPowmPrecisionsCount-1]) {
    if(alg!=xmpAlgorithmDefault && alg!=xmpAlgorithmDigitMP)
      return LaunchParameters();

    LaunchParameters params;
    //force it to digitized
    params.alg_index=xmpPowmAlgorithmsCount-1;
    //run at full count (assumes count isn't so big that it overflows CUDA)
    params.count=count;
  
    return params;
  }

  idx=0;
  for(int i=0;i<xmpPowmPrecisionsCount;i++) {
    if(precision<=xmpPowmPrecisions[i]) {
      idx=i;
      break;
    }
  }

  if(handle->arch<50) {
    lookup=powm_tbl_kepler[idx];
    tbl_count=powm_tbl_kepler_counts[idx];
  }else {
    lookup=powm_tbl_maxwell[idx];
    tbl_count=powm_tbl_maxwell_counts[idx];
  }

  //hueristic to reduce the number of waves at max
  uint32_t max_waves=8192/precision;
  max_waves=MIN(max_waves,16);
  max_waves=MAX(max_waves,1);


  LaunchParameters params = getFastestPowmLaunch(handle, alg, max_waves, count, lookup, tbl_count);
  
  //printf("POWM: precison: %d, count: %d, lcount: %d,  alg: %d\n", precision, count, params.count, params.alg_index);
  return params;
}


//computes out=base^exp % mod for count integers
xmpError_t XMPAPI xmpIntegersPowm(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t count) {
  xmpError_t error=xmpIntegersPowmAsync(handle,out,a,exp,mod,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

xmpError_t XMPAPI xmpIntegersPowmAsync(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t count) {
  int                  device=handle->device;
  xmpExecutionPolicy_t policy=handle->policy;
  //verify out, base, exp, mod devices all match handle device
  if(out->device!=device || a->device!=device || exp->device!=device || mod->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  int32_t precision=out->precision;

  if(out->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(out->precision!=precision || a->precision!=precision || mod->precision!=precision)
    return xmpErrorInvalidPrecision;

  xmpAlgorithm_t alg = policy->algorithm;

  bool inplace=false;
  xmpLimb_t *out_buffer=out->slimbs;
  size_t out_buffer_size=0;
  if(out==a || out==exp || out==mod) {
    inplace=true;
    out_buffer_size=out->stride*out->nlimbs*sizeof(xmpLimb_t);
    xmpError_t error=xmpSetNecessaryOutSize(handle, out_buffer_size);
    if(error!=xmpErrorSuccess)
      return error;
    out_buffer=(xmpLimb_t*)handle->tmpOut;
  }

  uint32_t start=0;
  while(start<count) {
    LaunchParameters params=getPowmLaunchParameters(handle,precision,count-start,alg);
    if(params.alg_index==-1)
      return xmpErrorUnsupported;
    uint32_t lcount=params.count;
    xmpPowmAlgorithm algorithm=xmpPowmAlgorithms[params.alg_index];
    xmpError_t error=algorithm.pfunc(handle, out, a, exp, mod, start, lcount, out_buffer, NULL, NULL);
    if(error!=xmpErrorSuccess) {
      return error;
    }
    start+=lcount;
  }

  if(inplace) {
    cudaMemcpyAsync(out->slimbs,out_buffer,out_buffer_size,cudaMemcpyDeviceToDevice,handle->stream);
  }

  return xmpErrorSuccess;



}

