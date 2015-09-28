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

#include "xmp_internal.h"
#include "arguments.h"
#include "implementations.h"

#define DIGIT 8
#define GSL true

template<class T>
void configureActiveBlocks(xmpHandle_t handle, dim3 &blocks, dim3 threads, T *kernel) {
  int         maxBlocks;
  cudaError_t error;

  if(GSL) {
    error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, kernel, threads.x, 0);
    if(error==cudaSuccess && blocks.x>maxBlocks*handle->smCount)
      blocks.x=maxBlocks*handle->smCount;
  }
}


//computes s=a+b
xmpError_t XMPAPI xmpIntegersAdd(xmpHandle_t handle, xmpIntegers_t s, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  xmpError_t error=xmpIntegersAddAsync(handle,s,a,b,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersAddAsync(xmpHandle_t handle, xmpIntegers_t s, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  add_arguments_t add_arguments;

  if(s->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(s->count<count)
    return xmpErrorInvalidCount;

  if(s->precision<MAX(a->precision,b->precision))
    return xmpErrorInvalidPrecision;

  XMP_CHECK_CUDA();

  a->requireFormat(handle, xmpFormatStrided);
  b->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  add_arguments.out_data=s->slimbs;
  add_arguments.out_len=DIV_ROUND_UP(s->precision, 32);
  add_arguments.out_stride=s->stride;
  add_arguments.a_data=a->slimbs;
  add_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  add_arguments.a_stride=a->stride;
  add_arguments.a_count=a->count;
  add_arguments.b_data=b->slimbs;
  add_arguments.b_len=DIV_ROUND_UP(b->precision, 32);
  add_arguments.b_stride=b->stride;
  add_arguments.b_count=b->count;

  if(a->precision<=128 && b->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_add_kernel<GSL, 4>);
    regmp_add_kernel<GSL, 4><<<blocks, threads, 0, handle->stream>>>(add_arguments, count);
  }
  else if(a->precision<=256 && b->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_add_kernel<GSL, 8>);
    regmp_add_kernel<GSL, 8><<<blocks, threads, 0, handle->stream>>>(add_arguments, count);
  }
  else if(a->precision<=384 && b->precision<=384) {
    configureActiveBlocks(handle, blocks, threads, regmp_add_kernel<GSL, 12>);
    regmp_add_kernel<GSL, 12><<<blocks, threads, 0, handle->stream>>>(add_arguments, count);
  }
  else if(a->precision<=512 && b->precision<=512) {
    configureActiveBlocks(handle, blocks, threads, regmp_add_kernel<GSL, 16>);
    regmp_add_kernel<GSL, 16><<<blocks, threads, 0, handle->stream>>>(add_arguments, count);
  }
  else {
    configureActiveBlocks(handle, blocks, threads, digitmp_add_kernel<GSL, DIGIT>);
    digitmp_add_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(add_arguments, count);
  }

  s->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//computes d=a-b
xmpError_t XMPAPI xmpIntegersSub(xmpHandle_t handle, xmpIntegers_t d, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  xmpError_t error=xmpIntegersSubAsync(handle,d,a,b,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersSubAsync(xmpHandle_t handle, xmpIntegers_t d, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  sub_arguments_t sub_arguments;

  if(d->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(d->count<count)
    return xmpErrorInvalidCount;

  if(d->precision<MAX(a->precision,b->precision))
    return xmpErrorInvalidPrecision;

  a->requireFormat(handle, xmpFormatStrided);
  b->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  sub_arguments.out_data=d->slimbs;
  sub_arguments.out_len=DIV_ROUND_UP(d->precision, 32);
  sub_arguments.out_stride=d->stride;
  sub_arguments.a_data=a->slimbs;
  sub_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  sub_arguments.a_stride=a->stride;
  sub_arguments.a_count=a->count;
  sub_arguments.b_data=b->slimbs;
  sub_arguments.b_len=DIV_ROUND_UP(b->precision, 32);
  sub_arguments.b_stride=b->stride;
  sub_arguments.b_count=b->count;

  if(a->precision<=128 && b->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_sub_kernel<GSL, 4>);
    regmp_sub_kernel<GSL, 4><<<blocks, threads, 0, handle->stream>>>(sub_arguments, count);
  }
  else if(a->precision<=256 && b->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_sub_kernel<GSL, 8>);
    regmp_sub_kernel<GSL, 8><<<blocks, threads, 0, handle->stream>>>(sub_arguments, count);
  }
  else if(a->precision<=384 && b->precision<=384) {
    configureActiveBlocks(handle, blocks, threads, regmp_sub_kernel<GSL, 12>);
    regmp_sub_kernel<GSL, 12><<<blocks, threads, 0, handle->stream>>>(sub_arguments, count);
  }
  else if(a->precision<=512 && b->precision<=512) {
    configureActiveBlocks(handle, blocks, threads, regmp_sub_kernel<GSL, 16>);
    regmp_sub_kernel<GSL, 16><<<blocks, threads, 0, handle->stream>>>(sub_arguments, count);
  }
  else {
    configureActiveBlocks(handle, blocks, threads, digitmp_sub_kernel<GSL, DIGIT>);
    digitmp_sub_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(sub_arguments, count);
  }

  d->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
//computes p=a*a -- private but called from xmpIntegersMulAsync
xmpError_t xmpIntegersSqrAsync(xmpHandle_t handle, xmpIntegers_t p, const xmpIntegers_t a, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  sqr_arguments_t sqr_arguments;

  if(p->device!=device || a->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(p->count<count)
    return xmpErrorInvalidCount;

  if(p->precision<a->precision)
    return xmpErrorInvalidPrecision;

  bool       inplace=(p==a);
  size_t     out_size=p->stride*p->nlimbs*sizeof(xmpLimb_t);
  xmpLimb_t *dst=p->slimbs;

  if(inplace) {
    xmpError_t error;

    error=xmpSetNecessaryScratchSize(handle, out_size);
    if(error!=xmpErrorSuccess)
      return error;
    dst=(xmpLimb_t*)handle->scratch;
  }

  a->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  sqr_arguments.out_data=dst;
  sqr_arguments.out_len=DIV_ROUND_UP(p->precision, 32);
  sqr_arguments.out_stride=p->stride;
  sqr_arguments.a_data=a->slimbs;
  sqr_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  sqr_arguments.a_stride=a->stride;
  sqr_arguments.a_count=a->count;

  if(a->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_sqr_kernel<GSL, 2>);
    regmp_sqr_kernel<GSL, 2><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
  }
  else if(a->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_sqr_kernel<GSL, 4>);
    regmp_sqr_kernel<GSL, 4><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
  }
  else if(a->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_sqr_kernel<GSL, 8>);
    regmp_sqr_kernel<GSL, 8><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
  }
  else if(a->precision<=384) {
    configureActiveBlocks(handle, blocks, threads, regmp_sqr_kernel<GSL, 12>);
    regmp_sqr_kernel<GSL, 12><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
  }
  else if(a->precision<=512) {
    configureActiveBlocks(handle, blocks, threads, regmp_sqr_kernel<GSL, 16>);
    regmp_sqr_kernel<GSL, 16><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
  }
  else {
    configureActiveBlocks(handle, blocks, threads, digitmp_sqr_kernel<GSL, DIGIT>);
    digitmp_sqr_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
  }

  if(inplace)
    cudaMemcpyAsync(p->slimbs,dst,out_size,cudaMemcpyDeviceToDevice,handle->stream);

  p->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

//computes p=a*b
xmpError_t XMPAPI xmpIntegersMul(xmpHandle_t handle, xmpIntegers_t p, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  xmpError_t error=xmpIntegersMulAsync(handle,p,a,b,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersMulAsync(xmpHandle_t handle, xmpIntegers_t p, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  xmpIntegers_t   l, s;
  mul_arguments_t mul_arguments;

  if(a==b)
    return xmpIntegersSqrAsync(handle, p, a, count);

  if(p->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(p->count<count)
    return xmpErrorInvalidCount;

  if(p->precision<MAX(a->precision,b->precision))
    return xmpErrorInvalidPrecision;

  if(a->precision>=b->precision) {
    l=a;
    s=b;
  }
  else {
    l=b;
    s=a;
  }

  bool       inplace=(p==a || p==b);
  size_t     out_size=p->stride*p->nlimbs*sizeof(xmpLimb_t);
  xmpLimb_t *dst=p->slimbs;

  if(inplace) {
    xmpError_t error=xmpSetNecessaryScratchSize(handle, out_size);
    if(error!=xmpErrorSuccess)
      return error;
    dst=(xmpLimb_t*)handle->scratch;
  }

  l->requireFormat(handle, xmpFormatStrided);
  s->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  mul_arguments.out_data=dst;
  mul_arguments.out_len=DIV_ROUND_UP(p->precision, 32);
  mul_arguments.out_stride=p->stride;
  mul_arguments.a_data=l->slimbs;
  mul_arguments.a_len=DIV_ROUND_UP(l->precision, 32);
  mul_arguments.a_stride=l->stride;
  mul_arguments.a_count=l->count;
  mul_arguments.b_data=s->slimbs;
  mul_arguments.b_len=DIV_ROUND_UP(s->precision, 32);
  mul_arguments.b_stride=s->stride;
  mul_arguments.b_count=s->count;

  // multiply is a very common operator, so we have many sizes
  if(l->precision<=64 && s->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 2, 2>);
    regmp_mul_kernel<GSL, 2, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=128 && s->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 4, 2>);
    regmp_mul_kernel<GSL, 4, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=192 && s->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 6, 2>);
    regmp_mul_kernel<GSL, 6, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=256 && s->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 8, 2>);
    regmp_mul_kernel<GSL, 8, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=384 && s->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 12, 2>);
    regmp_mul_kernel<GSL, 12, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=512 && s->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 16, 2>);
    regmp_mul_kernel<GSL, 16, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=128 && s->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 4, 4>);
    regmp_mul_kernel<GSL, 4, 4><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=256 && s->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 8, 4>);
    regmp_mul_kernel<GSL, 8, 4><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=256 && s->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 8, 8>);
    regmp_mul_kernel<GSL, 8, 8><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=384 && s->precision<=192) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 12, 6>);
    regmp_mul_kernel<GSL, 12, 6><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=384 && s->precision<=384) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 12, 12>);
    regmp_mul_kernel<GSL, 12, 12><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=512 && s->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 16, 8>);
    regmp_mul_kernel<GSL, 16, 8><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else if(l->precision<=512 && s->precision<=512) {
    configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 16, 16>);
    regmp_mul_kernel<GSL, 16, 16><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }
  else {
    configureActiveBlocks(handle, blocks, threads, digitmp_mul_kernel<GSL, DIGIT>);
    digitmp_mul_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
  }

  if(inplace)
    cudaMemcpyAsync(p->slimbs,dst,out_size,cudaMemcpyDeviceToDevice,handle->stream);

  p->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
//computes q=floor(a/b)
xmpError_t XMPAPI xmpIntegersDiv(xmpHandle_t handle, xmpIntegers_t q, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  xmpError_t error=xmpIntegersDivAsync(handle,q,a,b,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersDivAsync(xmpHandle_t handle, xmpIntegers_t q, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  xmpError_t      error;
  div_arguments_t div_arguments;

  if(q->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(q->count<count)
    return xmpErrorInvalidCount;

  if(q->precision<a->precision)
    return xmpErrorInvalidPrecision;

  a->requireFormat(handle, xmpFormatStrided);
  b->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  div_arguments.out_data=q->slimbs;
  div_arguments.out_len=DIV_ROUND_UP(q->precision, 32);
  div_arguments.out_stride=q->stride;
  div_arguments.a_data=a->slimbs;
  div_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  div_arguments.a_stride=a->stride;
  div_arguments.a_count=a->count;
  div_arguments.b_data=b->slimbs;
  div_arguments.b_len=DIV_ROUND_UP(b->precision, 32);
  div_arguments.b_stride=b->stride;
  div_arguments.b_count=b->count;
  div_arguments.scratch=NULL;

  if(a->precision<=64 && b->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 2, 2>);
    regmp_div_kernel<GSL, 2, 2><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
  }
  else if(a->precision<=128 && b->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 4, 2>);
    regmp_div_kernel<GSL, 4, 2><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
  }
  else if(a->precision<=128 && b->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 4, 4>);
    regmp_div_kernel<GSL, 4, 4><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
  }
  else if(a->precision<=256 && b->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 8, 4>);
    regmp_div_kernel<GSL, 8, 4><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
  }
  else if(a->precision<=256 && b->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 8, 8>);
    regmp_div_kernel<GSL, 8, 8><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
  }
  else if(a->precision<=384 && b->precision<=192) {
    configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 12, 6>);
    regmp_div_kernel<GSL, 12, 6><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
  }
  else if(a->precision<=384 && b->precision<=384) {
    configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 12, 12>);
    regmp_div_kernel<GSL, 12, 12><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
  }
  else if(a->precision<=512 && b->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 16, 8>);
    regmp_div_kernel<GSL, 16, 8><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
  }
  else if(a->precision<=512 && b->precision<=512) {
    configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 16, 16>);
    regmp_div_kernel<GSL, 16, 16><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
  }
  else {
    int32_t digits=DIV_ROUND_UP(a->precision, DIGIT*32) + DIV_ROUND_UP(b->precision, DIGIT*32) + 2;
    size_t  bytes=digits*DIGIT*sizeof(xmpLimb_t);

    // FIX FIX FIX - need to set up scratch in all cases, not just digitized

    bytes=bytes*ROUND_UP(count, GEOMETRY);

    error=xmpSetNecessaryScratchSize(handle, bytes);
    if(error!=xmpErrorSuccess)
      return error;

    div_arguments.scratch=(xmpLimb_t *)handle->scratch;

    configureActiveBlocks(handle, blocks, threads, digitmp_div_kernel<GSL, DIGIT>);
    digitmp_div_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
  }

  q->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
//computes m=a%b
xmpError_t XMPAPI xmpIntegersMod(xmpHandle_t handle, xmpIntegers_t m, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  xmpError_t error=xmpIntegersModAsync(handle,m,a,b,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersModAsync(xmpHandle_t handle, xmpIntegers_t m, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  xmpError_t      error;
  mod_arguments_t mod_arguments;

  if(m->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(m->count<count)
    return xmpErrorInvalidCount;

  if(m->precision<b->precision)
    return xmpErrorInvalidPrecision;

  a->requireFormat(handle, xmpFormatStrided);
  b->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  mod_arguments.out_data=m->slimbs;
  mod_arguments.out_len=DIV_ROUND_UP(m->precision, 32);
  mod_arguments.out_stride=m->stride;
  mod_arguments.a_data=a->slimbs;
  mod_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  mod_arguments.a_stride=a->stride;
  mod_arguments.a_count=a->count;
  mod_arguments.b_data=b->slimbs;
  mod_arguments.b_len=DIV_ROUND_UP(b->precision, 32);
  mod_arguments.b_stride=b->stride;
  mod_arguments.b_count=b->count;
  mod_arguments.scratch=NULL;

  if(a->precision<=64 && b->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 2, 2>);
    regmp_mod_kernel<GSL, 2, 2><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
  }
  else if(a->precision<=128 && b->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 4, 2>);
    regmp_mod_kernel<GSL, 4, 2><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
  }
  else if(a->precision<=128 && b->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 4, 4>);
    regmp_mod_kernel<GSL, 4, 4><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
  }
  else if(a->precision<=256 && b->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 8, 4>);
    regmp_mod_kernel<GSL, 8, 4><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
  }
  else if(a->precision<=256 && b->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 8, 8>);
    regmp_mod_kernel<GSL, 8, 8><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
  }
  else if(a->precision<=384 && b->precision<=192) {
    configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 12, 6>);
    regmp_mod_kernel<GSL, 12, 6><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
  }
  else if(a->precision<=384 && b->precision<=384) {
    configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 12, 12>);
    regmp_mod_kernel<GSL, 12, 12><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
  }
  else if(a->precision<=512 && b->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 16, 8>);
    regmp_mod_kernel<GSL, 16, 8><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
  }
  else if(a->precision<=512 && b->precision<=512) {
    configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 16, 16>);
    regmp_mod_kernel<GSL, 16, 16><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
  }
  else {
    int32_t digits=DIV_ROUND_UP(a->precision, DIGIT*32) + DIV_ROUND_UP(b->precision, DIGIT*32) + 2;
    size_t  bytes=digits*DIGIT*sizeof(xmpLimb_t);

    bytes=bytes*ROUND_UP(count, GEOMETRY);

    error=xmpSetNecessaryScratchSize(handle, bytes);
    if(error!=xmpErrorSuccess)
      return error;

    mod_arguments.scratch=(xmpLimb_t *)handle->scratch;

    configureActiveBlocks(handle, blocks, threads, digitmp_mod_kernel<GSL, DIGIT>);
    digitmp_mod_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
  }

  m->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
//computes q=floor(a/b) and m=a%b
xmpError_t XMPAPI xmpIntegersDivMod(xmpHandle_t handle, xmpIntegers_t q, xmpIntegers_t m, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  xmpError_t error=xmpIntegersDivModAsync(handle,q,m,a,b,count);
  if(error!=xmpErrorSuccess)
    return error;
  XMP_SET_DEVICE(handle);
  cudaStreamSynchronize(handle->stream);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
xmpError_t XMPAPI xmpIntegersDivModAsync(xmpHandle_t handle, xmpIntegers_t q, xmpIntegers_t m, const xmpIntegers_t a, const xmpIntegers_t b, uint32_t count) {
  dim3               blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int                device=handle->device;
  xmpError_t         error;
  divmod_arguments_t divmod_arguments;

  if(q->device!=device || m->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(m->count<count || q->count<count)
    return xmpErrorInvalidCount;

  if(q->precision<a->precision || m->precision<b->precision)
    return xmpErrorInvalidPrecision;

  a->requireFormat(handle, xmpFormatStrided);
  b->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  divmod_arguments.q_data=q->slimbs;
  divmod_arguments.q_len=DIV_ROUND_UP(q->precision, 32);
  divmod_arguments.q_stride=q->stride;
  divmod_arguments.m_data=m->slimbs;
  divmod_arguments.m_len=DIV_ROUND_UP(m->precision, 32);
  divmod_arguments.m_stride=m->stride;
  divmod_arguments.a_data=a->slimbs;
  divmod_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  divmod_arguments.a_stride=a->stride;
  divmod_arguments.a_count=a->count;
  divmod_arguments.b_data=b->slimbs;
  divmod_arguments.b_len=DIV_ROUND_UP(b->precision, 32);
  divmod_arguments.b_stride=b->stride;
  divmod_arguments.b_count=b->count;
  divmod_arguments.scratch=NULL;

  if(a->precision<=64 && b->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 2, 2>);
    regmp_divmod_kernel<GSL, 2, 2><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
  }
  else if(a->precision<=128 && b->precision<=64) {
    configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 4, 2>);
    regmp_divmod_kernel<GSL, 4, 2><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
  }
  else if(a->precision<=128 && b->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 4, 4>);
    regmp_divmod_kernel<GSL, 4, 4><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
  }
  else if(a->precision<=256 && b->precision<=128) {
    configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 8, 4>);
    regmp_divmod_kernel<GSL, 8, 4><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
  }
  else if(a->precision<=256 && b->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 8, 8>);
    regmp_divmod_kernel<GSL, 8, 8><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
  }
  else if(a->precision<=384 && b->precision<=192) {
    configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 12, 6>);
    regmp_divmod_kernel<GSL, 12, 6><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
  }
  else if(a->precision<=384 && b->precision<=384) {
    configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 12, 12>);
    regmp_divmod_kernel<GSL, 12, 12><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
  }
  else if(a->precision<=512 && b->precision<=256) {
    configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 16, 8>);
    regmp_divmod_kernel<GSL, 16, 8><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
  }
  else if(a->precision<=512 && b->precision<=512) {
    configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 16, 16>);
    regmp_divmod_kernel<GSL, 16, 16><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
  }
  else {
    int32_t digits=DIV_ROUND_UP(a->precision, DIGIT*32) + DIV_ROUND_UP(b->precision, DIGIT*32) + 2;
    size_t  bytes=digits*DIGIT*sizeof(xmpLimb_t);

    bytes=bytes*ROUND_UP(count, GEOMETRY);

    error=xmpSetNecessaryScratchSize(handle, bytes);
    if(error!=xmpErrorSuccess)
      return error;

    divmod_arguments.scratch=(xmpLimb_t *)handle->scratch;

    configureActiveBlocks(handle, blocks, threads, digitmp_divmod_kernel<GSL, DIGIT>);
    digitmp_divmod_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
  }

  q->setFormat(xmpFormatStrided);
  m->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
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
  int              device=handle->device;
  int              bits, windowBits;
  size_t           windowBytes;
  xmpError_t       error;
  ar_arguments_t   ar_arguments;
  powm_arguments_t powm_arguments;

  //verify out, base, exp, mod devices all match handle device
  if(out->device!=device || a->device!=device || exp->device!=device || mod->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  int32_t precision=out->precision;

  if(out->count<count)
    return xmpErrorInvalidCount;

  if(out->precision!=precision || a->precision!=precision || mod->precision!=precision)
    return xmpErrorInvalidPrecision;

  // these are not tuned
  bits=exp->precision;
  if(bits<400)
    windowBits=4;
  else if(bits<800)
    windowBits=5;
  else if(bits<1600)
    windowBits=6;
  else if(bits<=4096)
    windowBits=7;
  else
    windowBits=8;

  // modBytes and windowBytes size must match _words
  if(precision<=128)
    windowBytes=((1<<windowBits)+4)*4*4;
  else if(precision<=256)
    windowBytes=((1<<windowBits)+4)*8*4;
  else if(precision<=384)
    windowBytes=((1<<windowBits)+4)*12*4;
  else if(precision<=512)
    windowBytes=((1<<windowBits)+4)*16*4;
  else
    windowBytes=((1<<windowBits)+4)*ROUND_UP(DIV_ROUND_UP(precision, 32), DIGIT)*4;

  windowBytes*=ROUND_UP(count, GEOMETRY);

  error=xmpSetNecessaryScratchSize(handle, windowBytes);
  if(error!=xmpErrorSuccess)
    return error;

  a->requireFormat(handle, xmpFormatStrided);
  mod->requireFormat(handle, xmpFormatStrided);
  exp->requireFormat(handle, xmpFormatStrided);

  // package up the arguments
  ar_arguments.window_data=(xmpLimb_t *)handle->scratch;
  ar_arguments.window_bits=windowBits;
  ar_arguments.a_data=a->slimbs;
  ar_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  ar_arguments.a_stride=a->stride;
  ar_arguments.a_count=a->count;
  ar_arguments.mod_data=mod->slimbs;
  ar_arguments.mod_len=DIV_ROUND_UP(mod->precision, 32);
  ar_arguments.mod_stride=mod->stride;
  ar_arguments.mod_count=mod->count;

  if(precision<=128) {
    dim3 blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);

    configureActiveBlocks(handle, blocks, threads, regmp_ar_kernel<GSL, 4>);
    regmp_ar_kernel<GSL, 4><<<blocks, threads, 0, handle->stream>>>(ar_arguments, count);
  }
  else if(precision<=256) {
    dim3 blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);

    configureActiveBlocks(handle, blocks, threads, regmp_ar_kernel<GSL, 8>);
    regmp_ar_kernel<GSL, 8><<<blocks, threads, 0, handle->stream>>>(ar_arguments, count);
  }
  else if(precision<=384) {
    dim3 blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);

    configureActiveBlocks(handle, blocks, threads, regmp_ar_kernel<GSL, 12>);
    regmp_ar_kernel<GSL, 12><<<blocks, threads, 0, handle->stream>>>(ar_arguments, count);
  }
  else if(precision<=512) {
    dim3 blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);

    configureActiveBlocks(handle, blocks, threads, regmp_ar_kernel<GSL, 16>);
    regmp_ar_kernel<GSL, 16><<<blocks, threads, 0, handle->stream>>>(ar_arguments, count);
  }
  else {
    dim3 blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);

    configureActiveBlocks(handle, blocks, threads, digitmp_ar_kernel<GSL, DIGIT>);
    digitmp_ar_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(ar_arguments, count);
  }

  // package up the arguments
  powm_arguments.out_data=out->slimbs;
  powm_arguments.out_len=DIV_ROUND_UP(precision, 32);
  powm_arguments.out_stride=out->stride;
  powm_arguments.exp_data=exp->slimbs;
  powm_arguments.exp_stride=exp->stride;
  powm_arguments.exp_count=exp->count;
  powm_arguments.mod_count=0;
  powm_arguments.window_data=(xmpLimb_t *)handle->scratch;
  powm_arguments.digits=DIV_ROUND_UP(precision, DIGIT*32);
  powm_arguments.bits=exp->precision;
  powm_arguments.window_bits=windowBits;

  if(precision<=128) {
    dim3 blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);

    configureActiveBlocks(handle, blocks, threads, regmp_powm_kernel<GSL, 4, 0, 0>);
    regmp_powm_kernel<GSL, 4, 0, 0><<<blocks, threads, 0, handle->stream>>>(powm_arguments, count);
  }
  else if(precision<=256) {
    dim3 blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);

    configureActiveBlocks(handle, blocks, threads, regmp_powm_kernel<GSL, 8, 0, 0>);
    regmp_powm_kernel<GSL, 8, 0, 0><<<blocks, threads, 0, handle->stream>>>(powm_arguments, count);
  }
  else if(precision<=384) {
    dim3 blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);

    configureActiveBlocks(handle, blocks, threads, regmp_powm_kernel<GSL, 12, 0, 0>);
    regmp_powm_kernel<GSL, 12, 0, 0><<<blocks, threads, 0, handle->stream>>>(powm_arguments, count);
  }
  else if(precision<=512) {
    dim3 blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);

    configureActiveBlocks(handle, blocks, threads, regmp_powm_kernel<GSL, 16, 0, 0>);
    regmp_powm_kernel<GSL, 16, 0, 0><<<blocks, threads, 0, handle->stream>>>(powm_arguments, count);
  }
  else if(ROUND_UP(mod->precision, DIGIT*32)/8*mod->count<2048) {
    dim3    blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
    int32_t shared_mem;

    shared_mem=ROUND_UP(mod->precision, DIGIT*32)/8*mod->count;
    powm_arguments.mod_count=mod->count;  // turn on the shared memory caching

    configureActiveBlocks(handle, blocks, threads, digitmp_powm_kernel<true, GSL, DIGIT>);
    digitmp_powm_kernel<true, GSL, DIGIT><<<blocks, threads, shared_mem, handle->stream>>>(powm_arguments, count);
  }
  else {
    dim3    blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);

    configureActiveBlocks(handle, blocks, threads, digitmp_powm_kernel<false, GSL, DIGIT>);
    digitmp_powm_kernel<false, GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(powm_arguments, count);
  }

  out->setFormat(xmpFormatStrided);
  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

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

  XMP_CHECK_NE(shift,NULL);

  if(c->device!=device || a->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(c->count<count)
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

  if(c->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(c->count<count)
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

  configureActiveBlocks(handle, blocks, threads, strided_ior_kernel<GSL>);
  strided_ior_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(ior_arguments, count);

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

  if(c->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(c->count<count)
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

  configureActiveBlocks(handle, blocks, threads, strided_and_kernel<GSL>);
  strided_and_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(and_arguments, count);

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

  if(c->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(c->count<count)
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

  configureActiveBlocks(handle, blocks, threads, strided_xor_kernel<GSL>);
  strided_xor_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(xor_arguments, count);

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

  if(c->device!=device || a->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(c->count<count)
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

  configureActiveBlocks(handle, blocks, threads, strided_not_kernel<GSL>);
  strided_not_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(not_arguments, count);

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

  configureActiveBlocks(handle, blocks, threads, strided_popc_kernel<GSL>);
  strided_popc_kernel<GSL><<<blocks, threads, 0, handle->stream>>>(popc_arguments, count);

  if(attrib.memoryType==cudaMemoryTypeHost) {
    cudaMemcpyAsync(c,dst,sizeof(uint32_t)*count,cudaMemcpyDefault,handle->stream);
  }
  return xmpErrorSuccess;
}

