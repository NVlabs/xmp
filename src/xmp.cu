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
#include "xmp_internal.h"
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h> 

void *xmpDeviceMalloc(size_t bytes) {
  void* retval;
  if(cudaSuccess!=cudaMalloc(&retval,bytes))
    return 0;
  return retval;
}
void xmpDeviceFree(void *ptr) {
  cudaFree(ptr);
}

//create xmp_handle
xmpError_t XMPAPI xmpHandleCreate(xmpHandle_t *handle) {
  return xmpHandleCreateWithMemoryFunctions(handle,malloc,free,xmpDeviceMalloc,xmpDeviceFree);
}

xmpError_t XMPAPI xmpHandleCreateWithMemoryFunctions(xmpHandle_t *handle,xmpAllocFunc ha, xmpFreeFunc hf, xmpAllocFunc da, xmpFreeFunc df) {
  XMP_CHECK_NE(handle,NULL);
  
  if(ha==NULL) ha=malloc;
  if(hf==NULL) hf=free;
  if(da==NULL) da=xmpDeviceMalloc;
  if(df==NULL) df=xmpDeviceFree;

  *handle=(_xmpHandle_t*)ha(sizeof(_xmpHandle_t));

  if(*handle==0)
    return xmpErrorInvalidMalloc;

  (*handle)->stream=0;
  (*handle)->scratch=NULL;
  (*handle)->scratchSize=0;
  (*handle)->tmpOutSize=0;
  (*handle)->tmpOut=NULL;
  (*handle)->ha=ha;
  (*handle)->hf=hf;
  (*handle)->da=da;
  (*handle)->df=df;

  (*handle)->policy=const_cast<xmpExecutionPolicy_t>(&xmpDefaultExecutionPolicy);

  if(cudaSuccess!=cudaGetDevice(&((*handle)->device)))
    return xmpErrorInvalidDevice;

  //verify device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,(*handle)->device);

  if(prop.major<2)
    return xmpErrorInvalidDevice;

  (*handle)->memorySize=prop.totalGlobalMem;
  (*handle)->arch=prop.major*10+prop.minor;
  (*handle)->smCount=prop.multiProcessorCount;

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}


//destroy xmp_handle
xmpError_t XMPAPI xmpHandleDestroy(xmpHandle_t handle) {
  XMP_SET_DEVICE(handle);

  handle->df(handle->scratch);
  handle->df(handle->tmpOut);

  //free handle
  handle->hf(handle); 
  
  if(cudaSuccess!=cudaPeekAtLastError())
    return xmpErrorCuda;

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//increases scratch size if necessary
xmpError_t xmpSetNecessaryScratchSize(xmpHandle_t handle, size_t bytes) {
  size_t maxSize=0;
  
  XMP_SET_DEVICE(handle);

  if(handle->policy!=NULL)
    maxSize=handle->policy->scratch_size_limit;
  if(maxSize==0)
    maxSize=handle->memorySize/4*3;
  if(bytes>maxSize)
    return xmpErrorIncreaseScratchLimit;
    
  if(handle->scratchSize<bytes)  {
    if(handle->scratch!=0) 
      //free existing scratch
      handle->df(handle->scratch);
    //allocate scratch
    handle->scratch=handle->da(bytes);
    handle->scratchSize=bytes;
  }
  if(handle->scratch==0)
    return xmpErrorInvalidCudaMalloc;
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
//increases output size if necessary
xmpError_t xmpSetNecessaryOutSize(xmpHandle_t handle, size_t bytes) {
  size_t maxSize=0;
  
  XMP_SET_DEVICE(handle);

  if(handle->tmpOutSize<bytes)  {
    if(handle->tmpOut!=0) 
      //free existing scratch
      handle->df(handle->tmpOut);
    //allocate scratch
    handle->tmpOut=handle->da(bytes);
    handle->tmpOutSize=bytes;
  }
  if(handle->scratch==0)
    return xmpErrorInvalidCudaMalloc;
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//get memory functions
xmpError_t XMPAPI xmpHandleGetMemoryFunctions(xmpHandle_t handle, xmpAllocFunc *ha, xmpFreeFunc *hf, xmpAllocFunc *da, xmpFreeFunc *df) {
  if(ha!=NULL) *ha=handle->ha;
  if(hf!=NULL) *hf=handle->hf;
  if(da!=NULL) *da=handle->da;
  if(hf!=NULL) *df=handle->df;

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//set stream for CUDA operations
xmpError_t XMPAPI xmpHandleSetStream(xmpHandle_t handle, cudaStream_t stream) {
  //TODO check that the stream and handle device match (not supported in CUDA yet)

  //TODO set dependcy between old stream and new stream
  handle->stream=stream;

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}


//get the stream associated with the handle
xmpError_t XMPAPI xmpHandleGetStream(xmpHandle_t handle, cudaStream_t *stream) {
  XMP_CHECK_NE(stream,NULL);

  *stream=handle->stream;

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//get the device associated with the handle
xmpError_t XMPAPI xmpHandleGetDevice(xmpHandle_t handle, int32_t *device) {
  XMP_CHECK_NE(device,NULL);

  *device=handle->device;

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

xmpError_t XMPAPI xmpHandleSetExecutionPolicy(xmpHandle_t handle, xmpExecutionPolicy_t policy) {
  if(policy!=NULL)
    handle->policy=policy;
  else
    handle->policy=const_cast<xmpExecutionPolicy_t>(&xmpDefaultExecutionPolicy);

  return xmpErrorSuccess;
}

//creates an execution policy
xmpError_t XMPAPI xmpExecutionPolicyCreate(xmpHandle_t handle, xmpExecutionPolicy_t *policy) {
  XMP_CHECK_NE(policy,NULL);
  XMP_SET_DEVICE(handle);
  
  //allocate policy
  *policy=( _xmpExecutionPolicy_t*)handle->ha(sizeof(_xmpExecutionPolicy_t));
  if(*policy==0)
    return xmpErrorInvalidMalloc;
 
  (*policy)->scratch_size_limit=0;
  for(int i=0;i<XMP_EXECUTION_POLICY_MAX_INDICES_ARRAYS;i++)
    (*policy)->indices[i]=NULL;

  (*policy)->algorithm=xmpAlgorithmDefault;

  return xmpErrorSuccess;
}
//destroys an execution policy
xmpError_t XMPAPI xmpExecutionPolicyDestroy(xmpHandle_t handle, xmpExecutionPolicy_t policy) {
  XMP_SET_DEVICE(handle);

  //free indices
  for(int i=0;i<XMP_EXECUTION_POLICY_MAX_INDICES_ARRAYS;i++) {
    if(policy->indices[i]!=NULL) 
      handle->df(policy->indices[i]);
  }

  //free handle
  handle->hf(policy);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//set dynamic indices
xmpError_t XMPAPI xmpExecutionPolicySetIndicesAsync(xmpHandle_t handle, xmpExecutionPolicy_t policy, uint32_t which_integer, uint32_t *indices, uint32_t count) {
  XMP_SET_DEVICE(handle);
  
  
  if(which_integer>=XMP_EXECUTION_POLICY_MAX_INDICES_ARRAYS) {
    return xmpErrorInvalidParameter;
  }


  uint32_t **pindices=&policy->indices[which_integer];
  uint32_t *pcount=&policy->indices_count[which_integer];

  //if the parameter is null clear existing indices
  if(indices==NULL) {
    if(*pindices!=NULL) {
      handle->df(*pindices);
      *pindices=NULL;
    }
  } else {
    //if already allocated but a different size
    if(*pindices!=NULL && *pcount!=count) {
      //free old indices
      handle->df(*pindices);
      *pindices=NULL;
    }

    //if not allocated
    if(*pindices==NULL) {
      //allocate new indices
      *pindices=(uint32_t*)handle->da(sizeof(uint32_t)*count);
      *pcount=count;
      if(*pindices==NULL)  
        return xmpErrorInvalidCudaMalloc;
    }

    cudaMemcpyAsync(*pindices,indices,sizeof(uint32_t)*count,cudaMemcpyDefault,handle->stream);
  
    *pcount=count;
  }


  return xmpErrorSuccess;
}
  
xmpError_t XMPAPI xmpExecutionPolicySetIndices(xmpHandle_t handle, xmpExecutionPolicy_t policy, uint32_t which_integer, uint32_t *indices, uint32_t count) {
  xmpError_t error=xmpExecutionPolicySetIndicesAsync(handle,policy,which_integer,indices,count);
  cudaStreamSynchronize(handle->stream);
  return error;
}

xmpError_t XMPAPI xmpExecutionPolicySetParameter(xmpHandle_t handle, xmpExecutionPolicy_t policy, xmpExecutionPolicyParam_t param, xmpExecutionPolicyValue_t val) {
  switch(param) {
    case xmpAlgorithm:
      policy->algorithm=val.algorithm;
      break;
    case xmpScratchSizeLimit:
      policy->scratch_size_limit=val.size;
      break;
    default:
      return xmpErrorInvalidParameter;
  }
  return xmpErrorSuccess;
}

xmpError_t XMPAPI xmpExecutionPolicyGetParameter(xmpHandle_t handle, xmpExecutionPolicy_t policy, xmpExecutionPolicyParam_t param, xmpExecutionPolicyValue_t &val) {
  switch(param) {
    case xmpAlgorithm:
      val.algorithm=policy->algorithm;
      break;
    case xmpScratchSizeLimit:
      val.size=policy->scratch_size_limit;
      break;
    default:
      return xmpErrorInvalidParameter;
  }
  return xmpErrorSuccess;
}

//allocate array of integers
xmpError_t XMPAPI xmpIntegersCreate(xmpHandle_t handle, xmpIntegers_t *x, uint32_t precision, uint32_t count) {
  XMP_CHECK_NE(x,NULL);
  XMP_SET_DEVICE(handle);
 
  if(count==0 || count>0xfffffffc)
    return xmpErrorInvalidCount;

  //allocate integer handle
  *x=(_xmpIntegers_t*)handle->ha(sizeof(_xmpIntegers_t));
  
  if(*x==0)
    return xmpErrorInvalidMalloc;

  uint32_t bits_per_limb=sizeof(xmpLimb_t)*8;
  
  if(precision%(sizeof(uint32_t)*8)!=0)
    return xmpErrorUnsupported;

  //precision=ROUND_UP(precision,bits_per_limb);
  uint32_t stride=ROUND_UP(count,128/sizeof(xmpLimb_t));  //round up to 128 byte boundaries
  uint32_t nlimbs=precision/bits_per_limb;
   
  (*x)->count=count;
  (*x)->precision=precision;
  (*x)->nlimbs=nlimbs;
  (*x)->device=handle->device;
  (*x)->format=xmpFormatNone;
  (*x)->stride=stride;

  //allocate array of integers on the device
  (*x)->climbs=(xmpLimb_t*)handle->da(sizeof(xmpLimb_t)*nlimbs*count);
  (*x)->slimbs=(xmpLimb_t*)handle->da(sizeof(xmpLimb_t)*nlimbs*stride);
  if((*x)->climbs==0 || (*x)->slimbs==0 )
    return xmpErrorInvalidCudaMalloc;

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//free an array of integers
xmpError_t XMPAPI xmpIntegersDestroy(xmpHandle_t handle, xmpIntegers_t x) {
  XMP_SET_DEVICE(handle);

  //free array of integers using cudaFree here
  handle->df(x->climbs);
  handle->df(x->slimbs);

  //free integer handle
  handle->hf(x);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//get the device associated with the handle
xmpError_t XMPAPI xmpIntegersGetPrecision(xmpHandle_t handle, xmpIntegers_t x, uint32_t *precision) {
  XMP_CHECK_NE(precision,NULL);

  *precision=x->precision;

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
//get the device associated with the handle
xmpError_t XMPAPI xmpIntegersGetCount(xmpHandle_t handle, xmpIntegers_t x, uint32_t *count) {
  XMP_CHECK_NE(count,NULL);

  *count=x->count;

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

int32_t query_endianess() {
  int32_t num=1;
  if(*(char *)&num == 1)
    return -1;
  else
    return 1;
}

__device__ inline uint8_t byte_swap(uint8_t w) { return w; }
__device__ inline uint16_t byte_swap(uint16_t w) { 
  asm("{"
      ".reg .b8 %wlo; \n"
      ".reg .b8 %whi; \n"
      "mov.b16 {%wlo,%whi}, %0; \n"
      "mov.b16 %0, {%whi,%wlo}; \n"
      "}" : "+h"(w) );
  return w; 
}
__device__ inline uint32_t byte_swap(uint32_t w) { 
  asm("{" 
    "prmt.b32    %0, %0, %0, 0x0123; \n"
      "}" : "+r"(w) );  
  return w; 
}
__device__ inline uint64_t byte_swap(uint64_t w) {
  asm("{"
    ".reg .u32 %alo; \n"
    ".reg .u32 %ahi; \n"
    ".reg .u32 %blo; \n"
    ".reg .u32 %bhi; \n"
    "mov.b64     {%alo,%ahi}, %0;   \n"
    "prmt.b32    %bhi, %alo, %ahi, 0x0123; \n"
    "prmt.b32    %blo, %alo, %ahi, 0x4567; \n"
    "mov.b64     %0,   {%blo,%bhi}; \n"
  "}" : "+l"(w) );
  return w; 
}

__global__ void printWordsStrided_kernel(xmpLimb_t* data, int limbs, int stride, int count) {
  for(int i=0;i<count;i++) {
    printf("i=%d\n    ",i);
    for(int j=limbs-1;j>=0;j--) {
      printf("%08x",data[j*stride+i]);
    }
    printf("\n");
  }
}

void printWordsStrided(xmpLimb_t* data, int limbs, int stride, int count) {
  printWordsStrided_kernel<<<1,1>>>(data,limbs,stride,count);
  cudaDeviceSynchronize();
}

__global__ void printWordsCompact_kernel(xmpLimb_t* data, int limbs, int count) {
  for(int i=0;i<count;i++) {
    printf("i=%d\n    ",i);
    for(int j=limbs-1;j>=0;j--) {
      printf("%08x",data[i*limbs+j]);
    }
    printf("\n");
  }
}

void printWordsCompact(xmpLimb_t* data, int limbs, int count) {
  printWordsCompact_kernel<<<1,1>>>(data,limbs,count);
  cudaDeviceSynchronize();
}



//transforms an array of data.  Can reverse the order, endian, and zero out the top nails bits of each word.
template<class word_t>
__global__ void xmpTransform(word_t *output, word_t *input, uint32_t count, uint32_t words, int32_t order, int32_t endian, uint32_t nails, uint32_t *out_indices, uint32_t *in_indices, uint32_t out_indices_count, uint32_t in_indices_count) {
  for(uint32_t i=blockIdx.y*blockDim.y+threadIdx.y;i<count;i+=blockDim.y*gridDim.y) {
    for(uint32_t j=blockIdx.x*blockDim.x+threadIdx.x;j<words;j+=blockDim.x*gridDim.x) {
      
      uint32_t in_idx=i, out_idx=i;

      if(NULL!=out_indices) out_idx=out_indices[out_idx%out_indices_count];
      if(NULL!=in_indices) in_idx=in_indices[in_idx%in_indices_count];

      //Read in the order we want to store
      uint32_t offset= (order==xmpNativeOrder) ? j : words-j-1;
      uint32_t idx=in_idx*words + offset;

      //read word
      word_t w=input[idx];
      
      //byte swap if big endian
      if(endian!=xmpNativeEndian) w=byte_swap(w);

      //apply nails
      word_t mask=word_t(-1)>>nails;
      w&=mask;
      
      //write in least significant first ordering
      output[out_idx*words+j]=w;
    }
  }
}


xmpError_t inline xmpIntegersImportInternal(xmpHandle_t handle, xmpIntegers_t out, uint32_t words, int32_t order, size_t size, int32_t endian, int32_t nails, void* in, uint32_t count, bool async) {
  XMP_CHECK_NE(in,NULL);

  xmpExecutionPolicy_t policy=handle->policy;

  //verify handle device and out device match
  int32_t device=handle->device;
  if(out->device!=device)
    return xmpErrorInvalidDevice;

  XMP_SET_DEVICE(handle);

  if(endian==0) endian=query_endianess();

  if(size!=1 && size!=2 && size!=4 && size!=8)
    return xmpErrorInvalidParameter;

  if(count==0 || count>out->count || words==0 || (order!=1 && order!=-1) ||  (endian!=1 && endian !=-1))
    return xmpErrorInvalidParameter;

  if(words*size*8!=out->precision)
    return xmpErrorInvalidPrecision;

  size_t bytes=count*words*size;
  if(endian==xmpNativeEndian && order==xmpNativeOrder && nails==0 && policy->indices[0]==NULL) {
    //common case, count & precision match, little endian, nails=0, no dynamic indexing,  no temporary memory needed just copy in
    if(cudaSuccess!=cudaMemcpyAsync(out->climbs,in,bytes,cudaMemcpyDefault,handle->stream))
      return xmpErrorCuda;
  } else {
    //check if we know where this pointer came from, if not assume host
    cudaPointerAttributes attrib;
    cudaError_t error=cudaPointerGetAttributes(&attrib,in);
    if(error!=cudaSuccess) {
      if(error==cudaErrorInvalidValue) {
        cudaGetLastError();  //reset to cudaSuccess
        attrib.memoryType=cudaMemoryTypeHost;
      } else {
        return xmpErrorCuda;
      }
    }
    
    void* src=in;
    if(attrib.memoryType==cudaMemoryTypeHost) {

      xmpError_t e=xmpSetNecessaryScratchSize(handle,bytes);
      if(e!=xmpErrorSuccess) return e;
      
      src=handle->scratch;

      //copy down to temporary memory
      if(cudaSuccess!=cudaMemcpyAsync(src,in,bytes,cudaMemcpyDefault,handle->stream))
        return xmpErrorCuda;
    }

    //x = words
    //y = count
    dim3 blocks,threads;
    threads.x=MIN(words,128);           //Use 1 thread per word (max 128)
    threads.y=DIV_ROUND_UP(128,threads.x);  //block size = ~128 threads
    blocks.x=DIV_ROUND_UP(words,threads.x);
    blocks.y=DIV_ROUND_UP(count,threads.y);
 
    //unpack from temporary memory
    switch(size) {
      case 1:
        xmpTransform<<<blocks,threads,0,handle->stream>>>((uint8_t*)out->climbs,(uint8_t*)src,count,words,order,endian,nails,NULL,policy->indices[0],NULL,policy->indices_count[0]);
        break;
      case 2:
        xmpTransform<<<blocks,threads,0,handle->stream>>>((uint16_t*)out->climbs,(uint16_t*)src,count,words,order,endian,nails,NULL,policy->indices[0],NULL,policy->indices_count[0]);
        break;
      case 4:
        xmpTransform<<<blocks,threads,0,handle->stream>>>((uint32_t*)out->climbs,(uint32_t*)src,count,words,order,endian,nails,NULL,policy->indices[0],NULL,policy->indices_count[0]);
        break;
      case 8:
        xmpTransform<<<blocks,threads,0,handle->stream>>>((uint64_t*)out->climbs,(uint64_t*)src,count,words,order,endian,nails,NULL,policy->indices[0],NULL,policy->indices_count[0]);
        break;
      default:
        return xmpErrorInvalidParameter;
    };
  }
  out->setFormat(xmpFormatCompact);
  if(!async) cudaStreamSynchronize(handle->stream);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

xmpError_t XMPAPI xmpIntegersImport(xmpHandle_t handle, xmpIntegers_t out, uint32_t words, int32_t order, size_t size, int32_t endian, uint32_t nails, void* in, uint32_t count) {
  return xmpIntegersImportInternal(handle,out,words,order,size,endian,nails,in,count,false);
}
xmpError_t XMPAPI xmpIntegersImportAsync(xmpHandle_t handle, xmpIntegers_t out, uint32_t words, int32_t order, size_t size, int32_t endian, uint32_t nails, void* in, uint32_t count) {
  return xmpIntegersImportInternal(handle,out,words,order,size,endian,nails,in,count,true);
}
//export count integers of size bytes from in into out
xmpError_t inline xmpIntegersExportInternal(xmpHandle_t handle, void* out, uint32_t *words, int32_t order, size_t size, int32_t endian, uint32_t nails, xmpIntegers_t in, uint32_t count, bool async) {
  XMP_CHECK_NE(out,NULL);

  xmpExecutionPolicy_t policy=handle->policy;
  
  //verify handle device and in device match
  int32_t device=handle->device;
  if(in->device!=device)
    return xmpErrorInvalidDevice;
  
  XMP_SET_DEVICE(handle);

  if(endian==0) endian=query_endianess();
  
  if(size!=1 && size!=2 && size!=4 && size!=8)
    return xmpErrorInvalidParameter;

  if(count==0 || count>in->count || (order!=1 && order!=-1) ||  (endian!=1 && endian !=-1))
    return xmpErrorInvalidParameter;
  
  if(xmpErrorSuccess!=in->requireFormat(handle,xmpFormatCompact))
    return xmpErrorInvalidFormat;

  uint32_t limbs=in->nlimbs;
  uint32_t w = limbs * (uint32_t)sizeof(xmpLimb_t) / size;
  size_t bytes=count*limbs*sizeof(xmpLimb_t);
  if(endian==xmpNativeEndian && order==xmpNativeOrder && nails==0  && policy->indices[0]==NULL) {
    //common case, naitve endian and order, , nails=0, no dynamic indexing, no temporary memory needed just copy in
    if(cudaSuccess!=cudaMemcpyAsync(out,in->climbs,bytes,cudaMemcpyDefault,handle->stream))
      return xmpErrorCuda;
  } else {

    //check if we know where this pointer came from, if not assume host
    cudaPointerAttributes attrib;
    cudaError_t error=cudaPointerGetAttributes(&attrib,out);
    if(error!=cudaSuccess) {
      if(error==cudaErrorInvalidValue) {
        cudaGetLastError();  //reset to cudaSuccess
        attrib.memoryType=cudaMemoryTypeHost;
      } else {
        return xmpErrorCuda;
      }
    }

    void* dst=out;
    if(attrib.memoryType==cudaMemoryTypeHost) {
      xmpError_t e=xmpSetNecessaryScratchSize(handle,bytes);
      if(e!=xmpErrorSuccess) return e;
      dst=handle->scratch;
    }

    //x = words
    //y = count
    dim3 blocks,threads;
    threads.x=MIN(w,128);           //Use 1 thread per word (max 128)
    threads.y=DIV_ROUND_UP(128,threads.x);  //block size = ~128 threads
    blocks.x=DIV_ROUND_UP(w,threads.x);
    blocks.y=DIV_ROUND_UP(count,threads.y);

    //pack to temporary memory
    switch(size) {
      case 1:
        xmpTransform<<<blocks,threads,0,handle->stream>>>((uint8_t*)dst,(uint8_t*)in->climbs,count,w,order,endian,nails,policy->indices[0],NULL,policy->indices_count[0],NULL);
        break;
      case 2:
        xmpTransform<<<blocks,threads,0,handle->stream>>>((uint16_t*)dst,(uint16_t*)in->climbs,count,w,order,endian,nails,policy->indices[0],NULL,policy->indices_count[0],NULL);
        break;
      case 4:
        xmpTransform<<<blocks,threads,0,handle->stream>>>((uint32_t*)dst,(uint32_t*)in->climbs,count,w,order,endian,nails,policy->indices[0],NULL,policy->indices_count[0],NULL);
        break;
      case 8:
        xmpTransform<<<blocks,threads,0,handle->stream>>>((uint64_t*)dst,(uint64_t*)in->climbs,count,w,order,endian,nails,policy->indices[0],NULL,policy->indices_count[0],NULL);
        break;
      default:
        return xmpErrorInvalidParameter;
    };
   
    if(attrib.memoryType==cudaMemoryTypeHost) {
      //copy up from temporary memory
      if(cudaSuccess!=cudaMemcpyAsync(out,dst,bytes,cudaMemcpyDefault,handle->stream))
        return xmpErrorCuda;
    }
  }

  if(words!=0) *words=w;
  if(!async) cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

xmpError_t XMPAPI xmpIntegersExport(xmpHandle_t handle, void* out, uint32_t *words, int32_t order, size_t size, int32_t endian, uint32_t nails, xmpIntegers_t in, uint32_t count) {
  return xmpIntegersExportInternal(handle,out,words,order,size,endian,nails,in,count,false);
}

xmpError_t XMPAPI xmpIntegersExportAsync(xmpHandle_t handle, void* out, uint32_t *words, int32_t order, size_t size, int32_t endian, uint32_t nails, xmpIntegers_t in, uint32_t count) {
  return xmpIntegersExportInternal(handle,out,words,order,size,endian,nails,in,count,true);
}


struct ReorderParams {
  uint32_t *out_limbs, *in_limbs;             //limb data
  uint32_t *out_indices, *in_indices;             //index arrays
  uint32_t out_indices_count, in_indices_count;   //index array count length
  uint32_t in_count;                          //actual length of the integer array
  uint32_t out_words, in_words;               //precision 
  uint32_t count;                             //number of elements to copy
};
__global__ void xmpReorder_compact_kernel(ReorderParams params) {
  uint32_t count=           params.count;
  uint32_t *in_index=       params.in_indices;
  uint32_t in_index_count=  params.in_indices_count;
  uint32_t in_count=        params.in_count;
  uint32_t *in_limbs=       params.in_limbs;
  uint32_t  in_words=       params.in_words;
  uint32_t *out_index=      params.out_indices;
  uint32_t out_index_count= params.out_indices_count;
  uint32_t *out_limbs=      params.out_limbs;
  uint32_t  out_words=      params.out_words;


  for(int idx=blockIdx.y*blockDim.y+threadIdx.y;idx<count;idx+=blockDim.y*gridDim.y) {
    for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<out_words;j+=blockDim.x*gridDim.x) {
      uint32_t i= (NULL==in_index) ? idx : in_index[idx%in_index_count]%in_count;
      uint32_t o= (NULL==out_index) ? idx : out_index[idx%out_index_count];
      //read with indirection
      uint32_t val = (j<in_words) ? in_limbs[i*in_words+j] : 0;
      //write with indirection
      out_limbs[o*out_words+j]=val;
    }
  }
}


xmpError_t XMPAPI xmpIntegersSet(xmpHandle_t handle, xmpIntegers_t out, xmpIntegers_t in, uint32_t count) {
  xmpError_t error=xmpIntegersSetAsync(handle,out,in,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaError_t cerror=cudaStreamSynchronize(handle->stream);
  if(cerror==cudaSuccess)
    return xmpErrorSuccess;
  else
    return xmpErrorCuda;
}
//copy count integers of size bytes from in into out
xmpError_t XMPAPI xmpIntegersSetAsync(xmpHandle_t handle, xmpIntegers_t out, xmpIntegers_t in, uint32_t count) {
  //verify handle device and in or out device match
  int32_t device=handle->device;
  if(in->device!=device && out->device!=device)
    return xmpErrorInvalidDevice;

  XMP_SET_DEVICE(handle);

  if(in->precision!=out->precision)
    return xmpErrorInvalidPrecision;

  if(out->count<count || in->count<count)
    return xmpErrorInvalidCount;
  
  size_t bytes=in->count*in->nlimbs*sizeof(xmpLimb_t);
  size_t sbytes=in->count*in->stride*sizeof(xmpLimb_t);
  
  xmpExecutionPolicy_t policy=handle->policy;

  if(policy->indices[0]==NULL && policy->indices[1]==NULL && in->count == out->count) {
    xmpFormat_t format=in->getFormat();

    switch(format) {
      case xmpFormatCompact:
        if(cudaSuccess!=cudaMemcpyAsync(out->climbs,in->climbs,bytes,cudaMemcpyDefault,handle->stream))
          return xmpErrorCuda;
        break;
      case xmpFormatStrided:
        if(cudaSuccess!=cudaMemcpyAsync(out->slimbs,in->slimbs,sbytes,cudaMemcpyDefault,handle->stream))
          return xmpErrorCuda;
        break;
      case xmpFormatBoth:
        if(cudaSuccess!=cudaMemcpyAsync(out->climbs,in->climbs,bytes,cudaMemcpyDefault,handle->stream))
          return xmpErrorCuda;
        if(cudaSuccess!=cudaMemcpyAsync(out->slimbs,in->slimbs,sbytes,cudaMemcpyDefault,handle->stream))
          return xmpErrorCuda;
        break;
      case xmpFormatNone:
        return xmpErrorInvalidFormat;
    }
    out->setFormat(format);
  } else {
    //currently only support working in compact format
    in->requireFormat(handle,xmpFormatCompact);

    //create scratch memory (handles, different devices and in-place transfers)
    xmpError_t e=xmpSetNecessaryScratchSize(handle,in->nlimbs*count*sizeof(xmpLimb_t));
    if(e!=xmpErrorSuccess) return e;
    uint32_t *dst=(uint32_t*)handle->scratch;

    //copy to temporary memory on target device
    if(cudaSuccess!=cudaMemcpyAsync(dst, in->climbs, bytes, cudaMemcpyDefault, handle->stream)) {
      return xmpErrorCuda;
    }

    ReorderParams params;

    params.out_limbs=out->climbs;
    params.out_indices=policy->indices[0];
    params.out_indices_count=policy->indices_count[0];
    params.in_limbs=dst;
    params.in_indices=policy->indices[1];
    params.in_indices_count=policy->indices_count[1];
    params.in_count=in->count;
    params.out_words=out->nlimbs;
    params.in_words=in->nlimbs;
    params.count=count;
    
    dim3 blocks, threads;

    threads.x=MIN(out->nlimbs,128);       //1 thread per limb, max of 128
    threads.y=DIV_ROUND_UP(128,threads.x);  //target block size is 128, fill in block size with instances if necessary
    blocks.x=DIV_ROUND_UP(out->nlimbs,threads.x);  //remaining limbs get covered in multiple blocks
    blocks.y=DIV_ROUND_UP(out->count,threads.y);    //handle remainig instances with more blocks

    xmpReorder_compact_kernel<<<blocks,threads>>>(params);

    out->setFormat(xmpFormatCompact);
  }

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//x along N
//y along limbs
__global__ void xmpC2S_kernel(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out) {
  //outer dimension = N
  //inner dimension = limbs
  
  //read strided in inner dimension`
  //write coalesced in outer dimension
  for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    for(uint32_t j=blockIdx.y*blockDim.y+threadIdx.y;j<limbs;j+=blockDim.y*gridDim.y) {
      out[j*stride + i] = in[i*limbs + j];
    }
  }
}

//x along limbs
//y along N
__global__ void xmpS2C_kernel(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out) {
  //outer dimension = limbs
  //inner dimension = N

  //read strided in inner dimension
  //write coalesced in outer dimension
  for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<limbs;i+=blockDim.x*gridDim.x) {
    for(uint32_t j=blockIdx.y*blockDim.y+threadIdx.y;j<N;j+=blockDim.y*gridDim.y) {
      out[j*limbs + i] = in[i*stride + j];
    }
  }
}


