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

//compute c=CMP(a,b),  -1 a is smaller, 0 equal, +1 a is larger
xmpError_t XMPAPI xmpIntegersCmp(xmpHandle_t handle, int32_t *c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  xmpError_t error=xmpIntegersCmpAsync(handle,c,a,b,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersCmpAsync(xmpHandle_t handle, int32_t *c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  int32_t         negate;
  xmpIntegers_t   l, s;
  xmpError_t      error;
  cmp_arguments_t cmp_arguments;
  xmpExecutionPolicy_t policy=handle->policy;

  if(a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  error=xmpSetNecessaryScratchSize(handle, count*sizeof(int32_t));
  if(error!=xmpErrorSuccess)
    return error;

  if(a->precision>=b->precision) {
    negate=1;
    l=a;
    s=b;
  }
  else {
    negate=-1;
    l=b;
    s=a;
  }

  //check if we know where this pointer came from, if not assume host
  cudaPointerAttributes attrib;
  cudaError_t cerror=cudaPointerGetAttributes(&attrib,c);
  if(cerror!=cudaSuccess) {
    if(cerror==cudaErrorInvalidValue) {
      cudaGetLastError();  //reset to cudaSuccess
      attrib.memoryType=cudaMemoryTypeHost;
    }
    else {
      return xmpErrorCuda;
    }
  }

  int32_t *dst=c;
  if(attrib.memoryType==cudaMemoryTypeHost) {
    xmpError_t error=xmpSetNecessaryScratchSize(handle, count*sizeof(int32_t));
    if(error!=xmpErrorSuccess)
      return error;

    dst=(int32_t*)handle->scratch;
  }

  l->requireFormat(handle, xmpFormatStrided);
  s->requireFormat(handle, xmpFormatStrided);

  cmp_arguments.out_data=dst;
  cmp_arguments.a_data=l->slimbs;
  cmp_arguments.b_data=s->slimbs;
  cmp_arguments.a_len=DIV_ROUND_UP(l->precision, 32);
  cmp_arguments.a_stride=l->stride;
  cmp_arguments.a_count=l->count;
  cmp_arguments.b_len=DIV_ROUND_UP(s->precision, 32);
  cmp_arguments.b_stride=s->stride;
  cmp_arguments.b_count=s->count;
  cmp_arguments.negate=negate;
  cmp_arguments.a_indices=policy->indices[0];
  cmp_arguments.b_indices=policy->indices[1];
  cmp_arguments.a_indices_count=policy->indices_count[0];
  cmp_arguments.b_indices_count=policy->indices_count[1];

  configureActiveBlocks(handle, blocks, threads, strided_compare_kernel<GSL>);
  strided_compare_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(cmp_arguments, count);

  if(attrib.memoryType==cudaMemoryTypeHost) {
    cudaMemcpyAsync(c,dst,sizeof(int32_t)*count,cudaMemcpyDefault,handle->stream);
  }
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//computes c=shift(a,shift)
xmpError_t XMPAPI xmpIntegersShf(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const int32_t* shift, const uint32_t shift_count, uint32_t count) {
  xmpError_t error=xmpIntegersShfAsync(handle,c,a,shift,shift_count,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersShfAsync(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const int32_t* shift, const uint32_t shift_count, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  shf_arguments_t shf_arguments;
  xmpExecutionPolicy_t policy=handle->policy;

  XMP_CHECK_NE(shift,NULL);

  if(c->device!=device || a->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(c->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  a->requireFormat(handle, xmpFormatStrided);

  //check if we know where this pointer came from, if not assume host
  cudaPointerAttributes attrib;
  cudaError_t error=cudaPointerGetAttributes(&attrib,shift);
  if(error!=cudaSuccess) {
    if(error==cudaErrorInvalidValue) {
      cudaGetLastError();  //reset to cudaSuccess
      attrib.memoryType=cudaMemoryTypeHost;
    } else {
      return xmpErrorCuda;
    }
  }

  uint32_t scount=MIN(shift_count,count);
  int32_t *src=const_cast<int32_t*>(shift);

  size_t scratch_size_out=0, scratch_size_shift=0;
  if(a==c)
    scratch_size_out=a->stride*a->nlimbs*sizeof(xmpLimb_t);
  if(attrib.memoryType==cudaMemoryTypeHost)
    scratch_size_shift=scount*sizeof(int32_t);

  xmpError_t xerror=xmpSetNecessaryScratchSize(handle, scratch_size_out+scratch_size_shift);
  if(xerror!=xmpErrorSuccess)
    return xerror;

  xmpLimb_t *dst=  (a==c) ? (xmpLimb_t*) handle->scratch : c->slimbs;

  if(attrib.memoryType==cudaMemoryTypeHost) {
    src=(int32_t*)((char*)handle->scratch+scratch_size_out);
    cudaMemcpyAsync(src,shift,scratch_size_shift,cudaMemcpyHostToDevice,handle->stream);
  }

  shf_arguments.out_data=(xmpLimb_t *)dst;
  shf_arguments.a_data=a->slimbs;
  shf_arguments.shift_data=(int32_t *)src;
  shf_arguments.out_len=DIV_ROUND_UP(c->precision, 32);
  shf_arguments.out_stride=c->stride;
  shf_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  shf_arguments.a_stride=a->stride;
  shf_arguments.a_count=a->count;
  shf_arguments.shift_count=scount;
  shf_arguments.out_indices=policy->indices[0];
  shf_arguments.a_indices=policy->indices[1];
  shf_arguments.a_indices_count=policy->indices_count[1];

  configureActiveBlocks(handle, blocks, threads, strided_shf_kernel<GSL>);
  strided_shf_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(shf_arguments, count);

  if(a==c) {
    cudaMemcpyAsync(c->slimbs,dst,scratch_size_out,cudaMemcpyDeviceToDevice,handle->stream);
  }

  c->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}


//computes c=a|b
xmpError_t XMPAPI xmpIntegersIor(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  xmpError_t error=xmpIntegersIorAsync(handle,c,a,b,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersIorAsync(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  xmpIntegers_t   l, s;
  ior_arguments_t ior_arguments;
  xmpExecutionPolicy_t policy=handle->policy;

  if(c->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(c->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(c->precision<MAX(a->precision, b->precision))
    return xmpErrorInvalidPrecision;

  if(a->precision>=b->precision) {
    l=a;
    s=b;
  }
  else {
    l=b;
    s=a;
  }

  l->requireFormat(handle, xmpFormatStrided);
  s->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  ior_arguments.out_data=c->slimbs;
  ior_arguments.out_len=DIV_ROUND_UP(c->precision, 32);
  ior_arguments.out_stride=c->stride;
  ior_arguments.a_data=l->slimbs;
  ior_arguments.a_len=DIV_ROUND_UP(l->precision, 32);
  ior_arguments.a_stride=l->stride;
  ior_arguments.a_count=l->count;
  ior_arguments.b_data=s->slimbs;
  ior_arguments.b_len=DIV_ROUND_UP(s->precision, 32);
  ior_arguments.b_stride=s->stride;
  ior_arguments.b_count=s->count;
  ior_arguments.out_indices=policy->indices[0];
  ior_arguments.a_indices=policy->indices[1];
  ior_arguments.b_indices=policy->indices[2];
  ior_arguments.a_indices_count=policy->indices_count[1];
  ior_arguments.b_indices_count=policy->indices_count[2];


  //if c is indexed and in-place we need to work in scratch memory
  bool inplace=(c==a || c==b);
  size_t out_size=c->stride*c->nlimbs*sizeof(xmpLimb_t);

  if(inplace) {
    xmpError_t error;
    error=xmpSetNecessaryScratchSize(handle, out_size);
    if(error!=xmpErrorSuccess)
      return error;
    ior_arguments.out_data=(xmpLimb_t*)handle->scratch;
  }

  configureActiveBlocks(handle, blocks, threads, strided_ior_kernel<GSL>);
  strided_ior_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(ior_arguments, count);


  if(inplace) {
    cudaMemcpyAsync(c->slimbs,ior_arguments.out_data,out_size,cudaMemcpyDeviceToDevice,handle->stream);
  }

  c->setFormat(xmpFormatStrided);
  return xmpErrorSuccess;
}
//computes c=a&b
xmpError_t XMPAPI xmpIntegersAnd(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  xmpError_t error=xmpIntegersAndAsync(handle,c,a,b,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersAndAsync(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  xmpIntegers_t   l, s;
  and_arguments_t and_arguments;
  xmpExecutionPolicy_t policy=handle->policy;

  if(c->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(c->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(c->precision<MAX(a->precision, b->precision))
    return xmpErrorInvalidPrecision;

  if(a->precision>=b->precision) {
    l=a;
    s=b;
  }
  else {
    l=b;
    s=a;
  }

  l->requireFormat(handle, xmpFormatStrided);
  s->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  and_arguments.out_data=c->slimbs;
  and_arguments.out_len=DIV_ROUND_UP(c->precision, 32);
  and_arguments.out_stride=c->stride;
  and_arguments.a_data=l->slimbs;
  and_arguments.a_len=DIV_ROUND_UP(l->precision, 32);
  and_arguments.a_stride=l->stride;
  and_arguments.a_count=l->count;
  and_arguments.b_data=s->slimbs;
  and_arguments.b_len=DIV_ROUND_UP(s->precision, 32);
  and_arguments.b_stride=s->stride;
  and_arguments.b_count=s->count;
  and_arguments.out_indices=policy->indices[0];
  and_arguments.a_indices=policy->indices[1];
  and_arguments.b_indices=policy->indices[2];
  and_arguments.a_indices_count=policy->indices_count[1];
  and_arguments.b_indices_count=policy->indices_count[2];

  //if c is indexed and in-place we need to work in scratch memory
  bool inplace=(c==a || c==b);
  size_t out_size=c->stride*c->nlimbs*sizeof(xmpLimb_t);

  if(inplace) {
    xmpError_t error;
    error=xmpSetNecessaryScratchSize(handle, out_size);
    if(error!=xmpErrorSuccess)
      return error;
    and_arguments.out_data=(xmpLimb_t*)handle->scratch;
  }

  configureActiveBlocks(handle, blocks, threads, strided_and_kernel<GSL>);
  strided_and_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(and_arguments, count);

  if(inplace) {
    cudaMemcpyAsync(c->slimbs,and_arguments.out_data,out_size,cudaMemcpyDeviceToDevice,handle->stream);
  }

  c->setFormat(xmpFormatStrided);
  return xmpErrorSuccess;
}
//computes c=a^b
xmpError_t XMPAPI xmpIntegersXor(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  xmpError_t error=xmpIntegersXorAsync(handle,c,a,b,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersXorAsync(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  xmpIntegers_t   l, s;
  xor_arguments_t xor_arguments;
  xmpExecutionPolicy_t policy=handle->policy;

  if(c->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(c->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(c->precision<MAX(a->precision, b->precision))
    return xmpErrorInvalidPrecision;

  if(a->precision>=b->precision) {
    l=a;
    s=b;
  }
  else {
    l=b;
    s=a;
  }

  l->requireFormat(handle, xmpFormatStrided);
  s->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  xor_arguments.out_data=c->slimbs;
  xor_arguments.out_len=DIV_ROUND_UP(c->precision, 32);
  xor_arguments.out_stride=c->stride;
  xor_arguments.a_data=l->slimbs;
  xor_arguments.a_len=DIV_ROUND_UP(l->precision, 32);
  xor_arguments.a_stride=l->stride;
  xor_arguments.a_count=l->count;
  xor_arguments.b_data=s->slimbs;
  xor_arguments.b_len=DIV_ROUND_UP(s->precision, 32);
  xor_arguments.b_stride=s->stride;
  xor_arguments.b_count=s->count;
  xor_arguments.out_indices=policy->indices[0];
  xor_arguments.a_indices=policy->indices[1];
  xor_arguments.b_indices=policy->indices[2];
  xor_arguments.a_indices_count=policy->indices_count[1];
  xor_arguments.b_indices_count=policy->indices_count[2];

  //if c is indexed and in-place we need to work in scratch memory
  bool inplace=(c==a || c==b);
  size_t out_size=c->stride*c->nlimbs*sizeof(xmpLimb_t);

  if(inplace) {
    xmpError_t error;
    error=xmpSetNecessaryScratchSize(handle, out_size);
    if(error!=xmpErrorSuccess)
      return error;
    xor_arguments.out_data=(xmpLimb_t*)handle->scratch;
  }

  configureActiveBlocks(handle, blocks, threads, strided_xor_kernel<GSL>);
  strided_xor_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(xor_arguments, count);

  if(inplace) {
    cudaMemcpyAsync(c->slimbs,xor_arguments.out_data,out_size,cudaMemcpyDeviceToDevice,handle->stream);
  }

  c->setFormat(xmpFormatStrided);
  return xmpErrorSuccess;
}
//computes c=!a
xmpError_t XMPAPI xmpIntegersNot(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, uint32_t count) {
  xmpError_t error=xmpIntegersNotAsync(handle,c,a,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersNotAsync(xmpHandle_t handle, xmpIntegers_t c, const xmpIntegers_t a, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  not_arguments_t not_arguments;
  xmpExecutionPolicy_t policy=handle->policy;

  if(c->device!=device || a->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(c->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(c->precision<a->precision)
    return xmpErrorInvalidPrecision;

  a->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  not_arguments.out_data=c->slimbs;
  not_arguments.out_len=DIV_ROUND_UP(c->precision, 32);
  not_arguments.out_stride=c->stride;
  not_arguments.a_data=a->slimbs;
  not_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  not_arguments.a_stride=a->stride;
  not_arguments.a_count=a->count;
  not_arguments.out_indices=policy->indices[0];
  not_arguments.a_indices=policy->indices[1];
  not_arguments.a_indices_count=policy->indices_count[1];

  //if c is in-place we need to work in scratch memory
  bool inplace=(c==a);
  size_t out_size=c->stride*c->nlimbs*sizeof(xmpLimb_t);

  if(inplace) {
    xmpError_t error;
    error=xmpSetNecessaryScratchSize(handle, out_size);
    if(error!=xmpErrorSuccess)
      return error;
    not_arguments.out_data=(xmpLimb_t*)handle->scratch;
  }


  configureActiveBlocks(handle, blocks, threads, strided_not_kernel<GSL>);
  strided_not_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(not_arguments, count);

  if(inplace) {
    cudaMemcpyAsync(c->slimbs,not_arguments.out_data,out_size,cudaMemcpyDeviceToDevice,handle->stream);
  }

  c->setFormat(xmpFormatStrided);
  return xmpErrorSuccess;
}
//compute c=popc(a)
xmpError_t XMPAPI xmpIntegersPopc(xmpHandle_t handle, uint32_t *c, const xmpIntegers_t a, uint32_t count) {
  xmpError_t error=xmpIntegersPopcAsync(handle,c,a,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersPopcAsync(xmpHandle_t handle, uint32_t *c, const xmpIntegers_t a, uint32_t count) {
  dim3             blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int              device=handle->device;
  xmpError_t       error;
  popc_arguments_t popc_arguments;
  xmpExecutionPolicy_t policy=handle->policy;

  if(a->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  if(a->count<count)
    return xmpErrorInvalidCount;

  XMP_SET_DEVICE(handle);

  //check if we know where this pointer came from, if not assume host
  cudaPointerAttributes attrib;
  cudaError_t cerror=cudaPointerGetAttributes(&attrib,c);
  if(cerror!=cudaSuccess) {
    if(cerror==cudaErrorInvalidValue) {
      cudaGetLastError();  //reset to cudaSuccess
      attrib.memoryType=cudaMemoryTypeHost;
    } else {
      return xmpErrorCuda;
    }
  }

  uint32_t *dst=c;
  if(attrib.memoryType==cudaMemoryTypeHost) {
    error=xmpSetNecessaryScratchSize(handle, count*sizeof(uint32_t));
    if(error!=xmpErrorSuccess)
      return error;
    dst=(uint32_t *)handle->scratch;
  }

  a->requireFormat(handle, xmpFormatStrided);

  popc_arguments.out_data=(uint32_t *)dst;
  popc_arguments.a_data=a->slimbs;
  popc_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  popc_arguments.a_stride=a->stride;
  popc_arguments.a_count=a->count;
  popc_arguments.a_indices=policy->indices[0];
  popc_arguments.a_indices_count=policy->indices_count[0];

  configureActiveBlocks(handle, blocks, threads, strided_popc_kernel<GSL>);
  strided_popc_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(popc_arguments, count);

  if(attrib.memoryType==cudaMemoryTypeHost) {
    cudaMemcpyAsync(c,dst,sizeof(uint32_t)*count,cudaMemcpyDefault,handle->stream);
  }
  return xmpErrorSuccess;
}
