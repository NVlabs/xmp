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
  xmpExecutionPolicy_t policy=handle->policy;

  if(s->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(s->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(s->precision<MAX(a->precision,b->precision))
    return xmpErrorInvalidPrecision;

  XMP_CHECK_CUDA();

  a->requireFormat(handle, xmpFormatStrided);
  b->requireFormat(handle, xmpFormatStrided);

  xmpAlgorithm_t alg = policy->algorithm;
  if(alg==xmpAlgorithmDefault) {
    if(a->precision<=512 && b->precision<=512)
      alg = xmpAlgorithmRegMP;
    else
      alg = xmpAlgorithmDigitMP;
  }

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
  add_arguments.out_indices=policy->indices[0];
  add_arguments.a_indices=policy->indices[1];
  add_arguments.b_indices=policy->indices[2];
  add_arguments.a_indices_count=policy->indices_count[1];
  add_arguments.b_indices_count=policy->indices_count[2];

  //if s is in-place we need to work in scratch memory
  bool inplace = (s==a || s==b);
  size_t out_size=s->stride*s->nlimbs*sizeof(xmpLimb_t);

  if(inplace) {
    xmpError_t error;
    error=xmpSetNecessaryScratchSize(handle, out_size);
    if(error!=xmpErrorSuccess)
      return error;
    add_arguments.out_data=(xmpLimb_t*)handle->scratch;
  }

  if(alg==xmpAlgorithmRegMP)  {
    if(a->precision<=128 && b->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_add_kernel<GSL, 4>);
      regmp_add_kernel<GSL, 4><<<blocks, threads, 0, handle->stream>>>(add_arguments, count);
      goto done;
    }
    else if(a->precision<=256 && b->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_add_kernel<GSL, 8>);
      regmp_add_kernel<GSL, 8><<<blocks, threads, 0, handle->stream>>>(add_arguments, count);
      goto done;
    }
    else if(a->precision<=384 && b->precision<=384) {
      configureActiveBlocks(handle, blocks, threads, regmp_add_kernel<GSL, 12>);
      regmp_add_kernel<GSL, 12><<<blocks, threads, 0, handle->stream>>>(add_arguments, count);
      goto done;
    }
    else if(a->precision<=512 && b->precision<=512) {
      configureActiveBlocks(handle, blocks, threads, regmp_add_kernel<GSL, 16>);
      regmp_add_kernel<GSL, 16><<<blocks, threads, 0, handle->stream>>>(add_arguments, count);
      goto done;
    }
  }
  else if(alg==xmpAlgorithmDigitMP) {
    configureActiveBlocks(handle, blocks, threads, digitmp_add_kernel<GSL, DIGIT>);
    digitmp_add_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(add_arguments, count);
    goto done;
  }

  //this is only reached if they requested an unsupported algorithm
  return xmpErrorUnsupported;

done:

  if(inplace) {
    cudaMemcpyAsync(s->slimbs,add_arguments.out_data,out_size,cudaMemcpyDeviceToDevice,handle->stream);
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
  xmpExecutionPolicy_t policy=handle->policy;

  if(d->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(d->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(d->precision<MAX(a->precision,b->precision))
    return xmpErrorInvalidPrecision;

  a->requireFormat(handle, xmpFormatStrided);
  b->requireFormat(handle, xmpFormatStrided);

  xmpAlgorithm_t alg = policy->algorithm;
  if(alg==xmpAlgorithmDefault) {
    if(a->precision<=512 && b->precision<512)
      alg = xmpAlgorithmRegMP;
    else
      alg = xmpAlgorithmDigitMP;
  }

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
  sub_arguments.out_indices=policy->indices[0];
  sub_arguments.a_indices=policy->indices[1];
  sub_arguments.b_indices=policy->indices[2];
  sub_arguments.a_indices_count=policy->indices_count[1];
  sub_arguments.b_indices_count=policy->indices_count[2];

  //if d is indexed and in-place we need to work in scratch memory
  bool inplace=(d==a || d==b);
  size_t out_size=d->stride*d->nlimbs*sizeof(xmpLimb_t);

  if(inplace) {
    xmpError_t error;
    error=xmpSetNecessaryScratchSize(handle, out_size);
    if(error!=xmpErrorSuccess)
      return error;
    sub_arguments.out_data=(xmpLimb_t*)handle->scratch;
  }

  if(alg==xmpAlgorithmRegMP) {
    if(a->precision<=128 && b->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_sub_kernel<GSL, 4>);
      regmp_sub_kernel<GSL, 4><<<blocks, threads, 0, handle->stream>>>(sub_arguments, count);
      goto done;
    }
    else if(a->precision<=256 && b->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_sub_kernel<GSL, 8>);
      regmp_sub_kernel<GSL, 8><<<blocks, threads, 0, handle->stream>>>(sub_arguments, count);
      goto done;
    }
    else if(a->precision<=384 && b->precision<=384) {
      configureActiveBlocks(handle, blocks, threads, regmp_sub_kernel<GSL, 12>);
      regmp_sub_kernel<GSL, 12><<<blocks, threads, 0, handle->stream>>>(sub_arguments, count);
      goto done;
    }
    else if(a->precision<=512 && b->precision<=512) {
      configureActiveBlocks(handle, blocks, threads, regmp_sub_kernel<GSL, 16>);
      regmp_sub_kernel<GSL, 16><<<blocks, threads, 0, handle->stream>>>(sub_arguments, count);
      goto done;
    }
  }
  else if(alg==xmpAlgorithmDigitMP){
    configureActiveBlocks(handle, blocks, threads, digitmp_sub_kernel<GSL, DIGIT>);
    digitmp_sub_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(sub_arguments, count);
    goto done;
  }
  //this is only reached if they requested an unsupported algorithm
  return xmpErrorUnsupported;
done:
  if(inplace) {
    cudaMemcpyAsync(d->slimbs,sub_arguments.out_data,out_size,cudaMemcpyDeviceToDevice,handle->stream);
  }

  d->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
