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

//computes p=a*a -- private but called from xmpIntegersMulAsync
xmpError_t XMPAPI xmpIntegersSqrAsync(xmpHandle_t handle, xmpIntegers_t p, const xmpIntegers_t a, uint32_t count) {
  dim3            blocks(DIV_ROUND_UP(count, GEOMETRY)), threads(GEOMETRY);
  int             device=handle->device;
  sqr_arguments_t sqr_arguments;
  xmpExecutionPolicy_t policy=handle->policy;

  if(p->device!=device || a->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(p->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
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

  xmpAlgorithm_t alg = policy->algorithm;
  if(alg==xmpAlgorithmDefault) {
    if(a->precision<=512)
      alg = xmpAlgorithmRegMP;
    else
      alg = xmpAlgorithmDigitMP;
  }

  // package up the arguments
  sqr_arguments.out_data=dst;
  sqr_arguments.out_len=DIV_ROUND_UP(p->precision, 32);
  sqr_arguments.out_stride=p->stride;
  sqr_arguments.a_data=a->slimbs;
  sqr_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  sqr_arguments.a_stride=a->stride;
  sqr_arguments.a_count=a->count;
  sqr_arguments.out_indices=policy->indices[0];
  sqr_arguments.a_indices=policy->indices[1];
  sqr_arguments.a_indices_count=policy->indices_count[1];

  if(alg==xmpAlgorithmRegMP) {
    if(a->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_sqr_kernel<GSL, 2>);
      regmp_sqr_kernel<GSL, 2><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
      goto done;
    }
    else if(a->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_sqr_kernel<GSL, 4>);
      regmp_sqr_kernel<GSL, 4><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
      goto done;
    }
    else if(a->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_sqr_kernel<GSL, 8>);
      regmp_sqr_kernel<GSL, 8><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
      goto done;
    }
    else if(a->precision<=384) {
      configureActiveBlocks(handle, blocks, threads, regmp_sqr_kernel<GSL, 12>);
      regmp_sqr_kernel<GSL, 12><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
      goto done;
    }
    else if(a->precision<=512) {
      configureActiveBlocks(handle, blocks, threads, regmp_sqr_kernel<GSL, 16>);
      regmp_sqr_kernel<GSL, 16><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
      goto done;
    }
  }
  else if(alg==xmpAlgorithmDigitMP) {
    configureActiveBlocks(handle, blocks, threads, digitmp_sqr_kernel<GSL, DIGIT>);
    digitmp_sqr_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(sqr_arguments, count);
    goto done;
  }

  //this is only reached if they requested an unsupported algorithm
  return xmpErrorUnsupported;
done:

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
  xmpExecutionPolicy_t policy=handle->policy;

  if(a==b && policy->indices[1]==NULL && policy->indices[2]==NULL) {
    return xmpIntegersSqrAsync(handle, p, a, count);
  }

  if(p->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(p->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
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

  xmpAlgorithm_t alg = policy->algorithm;
  if(alg==xmpAlgorithmDefault) {
    if(l->precision<=512 && s->precision<512)
      alg = xmpAlgorithmRegMP;
    else
      alg = xmpAlgorithmDigitMP;
  }


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
  mul_arguments.out_indices=policy->indices[0];
  mul_arguments.a_indices=policy->indices[1];
  mul_arguments.b_indices=policy->indices[2];
  mul_arguments.a_indices_count=policy->indices_count[1];
  mul_arguments.b_indices_count=policy->indices_count[2];

  if(alg==xmpAlgorithmRegMP) {
    // multiply is a very common operator, so we have many sizes
    if(l->precision<=64 && s->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 2, 2>);
      regmp_mul_kernel<GSL, 2, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=128 && s->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 4, 2>);
      regmp_mul_kernel<GSL, 4, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=192 && s->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 6, 2>);
      regmp_mul_kernel<GSL, 6, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=256 && s->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 8, 2>);
      regmp_mul_kernel<GSL, 8, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=384 && s->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 12, 2>);
      regmp_mul_kernel<GSL, 12, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=512 && s->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 16, 2>);
      regmp_mul_kernel<GSL, 16, 2><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=128 && s->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 4, 4>);
      regmp_mul_kernel<GSL, 4, 4><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=256 && s->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 8, 4>);
      regmp_mul_kernel<GSL, 8, 4><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=256 && s->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 8, 8>);
      regmp_mul_kernel<GSL, 8, 8><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=384 && s->precision<=192) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 12, 6>);
      regmp_mul_kernel<GSL, 12, 6><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=384 && s->precision<=384) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 12, 12>);
      regmp_mul_kernel<GSL, 12, 12><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=512 && s->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 16, 8>);
      regmp_mul_kernel<GSL, 16, 8><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
    else if(l->precision<=512 && s->precision<=512) {
      configureActiveBlocks(handle, blocks, threads, regmp_mul_kernel<GSL, 16, 16>);
      regmp_mul_kernel<GSL, 16, 16><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
      goto done;
    }
  }
  else if(alg==xmpAlgorithmDigitMP) {
    configureActiveBlocks(handle, blocks, threads, digitmp_mul_kernel<GSL, DIGIT>);
    digitmp_mul_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(mul_arguments, count);
    goto done;
  }

  //this is only reached if they requested an unsupported algorithm
  return xmpErrorUnsupported;
done:

  if(inplace)
    cudaMemcpyAsync(p->slimbs,dst,out_size,cudaMemcpyDeviceToDevice,handle->stream);

  p->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}
