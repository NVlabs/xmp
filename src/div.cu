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
  xmpExecutionPolicy_t policy=handle->policy;

  if(q->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(q->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(q->precision<a->precision)
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
  div_arguments.out_indices=policy->indices[0];
  div_arguments.a_indices=policy->indices[1];
  div_arguments.b_indices=policy->indices[2];
  div_arguments.a_indices_count=policy->indices_count[1];
  div_arguments.b_indices_count=policy->indices_count[2];

  //if q is in-place we need to work in scratch memory
  bool inplace= (q==a || q==b);
  size_t out_size=q->stride*q->nlimbs*sizeof(xmpLimb_t);

  if(inplace) {
    xmpError_t error;
    error=xmpSetNecessaryScratchSize(handle, out_size);
    if(error!=xmpErrorSuccess)
      return error;
    div_arguments.out_data=(xmpLimb_t*)handle->scratch;
  }

  if(alg==xmpAlgorithmRegMP) {
    if(a->precision<=64 && b->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 2, 2>);
      regmp_div_kernel<GSL, 2, 2><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
      goto done;
    }
    else if(a->precision<=128 && b->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 4, 2>);
      regmp_div_kernel<GSL, 4, 2><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
      goto done;
    }
    else if(a->precision<=128 && b->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 4, 4>);
      regmp_div_kernel<GSL, 4, 4><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
      goto done;
    }
    else if(a->precision<=256 && b->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 8, 4>);
      regmp_div_kernel<GSL, 8, 4><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
      goto done;
    }
    else if(a->precision<=256 && b->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 8, 8>);
      regmp_div_kernel<GSL, 8, 8><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
      goto done;
    }
    else if(a->precision<=384 && b->precision<=192) {
      configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 12, 6>);
      regmp_div_kernel<GSL, 12, 6><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
      goto done;
    }
    else if(a->precision<=384 && b->precision<=384) {
      configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 12, 12>);
      regmp_div_kernel<GSL, 12, 12><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
      goto done;
    }
    else if(a->precision<=512 && b->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 16, 8>);
      regmp_div_kernel<GSL, 16, 8><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
      goto done;
    }
    else if(a->precision<=512 && b->precision<=512) {
      configureActiveBlocks(handle, blocks, threads, regmp_div_kernel<GSL, 16, 16>);
      regmp_div_kernel<GSL, 16, 16><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
      goto done;
    }
  }
  else if(alg==xmpAlgorithmDigitMP) {
    int32_t digits=DIV_ROUND_UP(a->precision, DIGIT*32) + DIV_ROUND_UP(b->precision, DIGIT*32) + 2;
    size_t  bytes=digits*DIGIT*sizeof(xmpLimb_t);

    // FIX FIX FIX - need to set up scratch in all cases, not just digitized

    bytes=bytes*ROUND_UP(count, GEOMETRY);

    if(inplace) bytes+=out_size;

    error=xmpSetNecessaryScratchSize(handle, bytes);
    if(error!=xmpErrorSuccess)
      return error;

    if(inplace)
      div_arguments.scratch=(xmpLimb_t *)(reinterpret_cast<char*>(handle->scratch)+out_size);
    else
      div_arguments.scratch=(xmpLimb_t *)handle->scratch;

    configureActiveBlocks(handle, blocks, threads, digitmp_div_kernel<GSL, DIGIT>);
    digitmp_div_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(div_arguments, count);
    goto done;
  }

  //this is only reached if they requested an unsupported algorithm
  return xmpErrorUnsupported;
done:
  q->setFormat(xmpFormatStrided);

  if(inplace) {
    cudaMemcpyAsync(q->slimbs,div_arguments.out_data,out_size,cudaMemcpyDeviceToDevice,handle->stream);
  }

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
  xmpExecutionPolicy_t policy=handle->policy;

  if(m->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(m->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(m->precision<b->precision)
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
  mod_arguments.out_indices=policy->indices[0];
  mod_arguments.a_indices=policy->indices[1];
  mod_arguments.b_indices=policy->indices[2];
  mod_arguments.a_indices_count=policy->indices_count[1];
  mod_arguments.b_indices_count=policy->indices_count[2];

  //if m is in-place we need to work in scratch memory
  bool inplace=(m==a || m==b);
  size_t out_size=m->stride*m->nlimbs*sizeof(xmpLimb_t);

  if(inplace) {
    xmpError_t error;
    error=xmpSetNecessaryScratchSize(handle, out_size);
    if(error!=xmpErrorSuccess)
      return error;
    mod_arguments.out_data=(xmpLimb_t*)handle->scratch;
  }

  if(alg==xmpAlgorithmRegMP) {
    if(a->precision<=64 && b->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 2, 2>);
      regmp_mod_kernel<GSL, 2, 2><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
      goto done;
    }
    else if(a->precision<=128 && b->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 4, 2>);
      regmp_mod_kernel<GSL, 4, 2><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
      goto done;
    }
    else if(a->precision<=128 && b->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 4, 4>);
      regmp_mod_kernel<GSL, 4, 4><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
      goto done;
    }
    else if(a->precision<=256 && b->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 8, 4>);
      regmp_mod_kernel<GSL, 8, 4><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
      goto done;
    }
    else if(a->precision<=256 && b->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 8, 8>);
      regmp_mod_kernel<GSL, 8, 8><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
      goto done;
    }
    else if(a->precision<=384 && b->precision<=192) {
      configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 12, 6>);
      regmp_mod_kernel<GSL, 12, 6><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
      goto done;
    }
    else if(a->precision<=384 && b->precision<=384) {
      configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 12, 12>);
      regmp_mod_kernel<GSL, 12, 12><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
      goto done;
    }
    else if(a->precision<=512 && b->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 16, 8>);
      regmp_mod_kernel<GSL, 16, 8><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
      goto done;
    }
    else if(a->precision<=512 && b->precision<=512) {
      configureActiveBlocks(handle, blocks, threads, regmp_mod_kernel<GSL, 16, 16>);
      regmp_mod_kernel<GSL, 16, 16><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
      goto done;
    }
  }
  else if(alg==xmpAlgorithmDigitMP) {
    int32_t digits=DIV_ROUND_UP(a->precision, DIGIT*32) + DIV_ROUND_UP(b->precision, DIGIT*32) + 2;
    size_t  bytes=digits*DIGIT*sizeof(xmpLimb_t);

    bytes=bytes*ROUND_UP(count, GEOMETRY);

    if(inplace) bytes+=out_size;

    error=xmpSetNecessaryScratchSize(handle, bytes);
    if(error!=xmpErrorSuccess)
      return error;

    if(inplace)
      mod_arguments.scratch=(xmpLimb_t *)(reinterpret_cast<char*>(handle->scratch)+out_size);
    else
      mod_arguments.scratch=(xmpLimb_t *)handle->scratch;

    configureActiveBlocks(handle, blocks, threads, digitmp_mod_kernel<GSL, DIGIT>);
    digitmp_mod_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(mod_arguments, count);
    goto done;
  }

  //this is only reached if they requested an unsupported algorithm
  return xmpErrorUnsupported;
done:

  if(inplace) {
    cudaMemcpyAsync(m->slimbs,mod_arguments.out_data,out_size,cudaMemcpyDeviceToDevice,handle->stream);
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
  xmpExecutionPolicy_t policy=handle->policy;

  if(q->device!=device || m->device!=device || a->device!=device || b->device!=device)
    return xmpErrorInvalidDevice;

  if(count==0)
    return xmpErrorSuccess;

  XMP_SET_DEVICE(handle);

  if(m->count<count || q->count<count)
    return xmpErrorInvalidCount;

  if(policy->indices[0] && policy->indices_count[0]<count)
    return xmpErrorInvalidCount;

  if(policy->indices[1] && policy->indices_count[1]<count)
    return xmpErrorInvalidCount;

  if(q->precision<a->precision || m->precision<b->precision)
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
  divmod_arguments.q_indices=policy->indices[0];
  divmod_arguments.r_indices=policy->indices[1];
  divmod_arguments.a_indices=policy->indices[2];
  divmod_arguments.b_indices=policy->indices[3];
  divmod_arguments.a_indices_count=policy->indices_count[2];
  divmod_arguments.b_indices_count=policy->indices_count[3];

  //if q or m is in-place we need to work in scratch memory
  bool qinplace=(q==a || q==b);
  bool minplace=(m==a || m==b);
  size_t qout_size=q->stride*q->nlimbs*sizeof(xmpLimb_t);
  size_t mout_size=m->stride*m->nlimbs*sizeof(xmpLimb_t);

  int num_inplace = qinplace+minplace;

  if(num_inplace>0) {
    xmpError_t error;
    error=xmpSetNecessaryScratchSize(handle, qout_size+mout_size);
    if(error!=xmpErrorSuccess)
      return error;
  }
  size_t soffset=0;
  if(qinplace) {
    divmod_arguments.q_indices=(xmpLimb_t*)handle->scratch;
    soffset+=qout_size;
  }
  if(minplace) {
    divmod_arguments.r_indices=(xmpLimb_t*)(reinterpret_cast<char*>(handle->scratch)+soffset);
  }

  if(alg==xmpAlgorithmRegMP) {
    if(a->precision<=64 && b->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 2, 2>);
      regmp_divmod_kernel<GSL, 2, 2><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
      goto done;
    }
    else if(a->precision<=128 && b->precision<=64) {
      configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 4, 2>);
      regmp_divmod_kernel<GSL, 4, 2><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
      goto done;
    }
    else if(a->precision<=128 && b->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 4, 4>);
      regmp_divmod_kernel<GSL, 4, 4><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
      goto done;
    }
    else if(a->precision<=256 && b->precision<=128) {
      configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 8, 4>);
      regmp_divmod_kernel<GSL, 8, 4><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
      goto done;
    }
    else if(a->precision<=256 && b->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 8, 8>);
      regmp_divmod_kernel<GSL, 8, 8><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
      goto done;
    }
    else if(a->precision<=384 && b->precision<=192) {
      configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 12, 6>);
      regmp_divmod_kernel<GSL, 12, 6><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
      goto done;
    }
    else if(a->precision<=384 && b->precision<=384) {
      configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 12, 12>);
      regmp_divmod_kernel<GSL, 12, 12><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
      goto done;
    }
    else if(a->precision<=512 && b->precision<=256) {
      configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 16, 8>);
      regmp_divmod_kernel<GSL, 16, 8><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
      goto done;
    }
    else if(a->precision<=512 && b->precision<=512) {
      configureActiveBlocks(handle, blocks, threads, regmp_divmod_kernel<GSL, 16, 16>);
      regmp_divmod_kernel<GSL, 16, 16><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
      goto done;
    }
  }
  else if(alg==xmpAlgorithmDigitMP) {
    int32_t digits=DIV_ROUND_UP(a->precision, DIGIT*32) + DIV_ROUND_UP(b->precision, DIGIT*32) + 2;
    size_t  bytes=digits*DIGIT*sizeof(xmpLimb_t);

    bytes=bytes*ROUND_UP(count, GEOMETRY);

    error=xmpSetNecessaryScratchSize(handle, bytes);
    if(error!=xmpErrorSuccess)
      return error;

    divmod_arguments.scratch=(xmpLimb_t *)handle->scratch;

    configureActiveBlocks(handle, blocks, threads, digitmp_divmod_kernel<GSL, DIGIT>);
    digitmp_divmod_kernel<GSL, DIGIT><<<blocks, threads, 0, handle->stream>>>(divmod_arguments, count);
    goto done;
  }

  //this is only reached if they requested an unsupported algorithm
  return xmpErrorUnsupported;
done:

  if(qinplace) {
    cudaMemcpyAsync(q->slimbs,divmod_arguments.q_indices,qout_size,cudaMemcpyDeviceToDevice,handle->stream);
  }
  if(minplace) {
    cudaMemcpyAsync(m->slimbs,divmod_arguments.r_indices,mout_size,cudaMemcpyDeviceToDevice,handle->stream);
  }

  q->setFormat(xmpFormatStrided);
  m->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();
  return xmpErrorSuccess;
}

