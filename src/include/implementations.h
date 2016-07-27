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
#include <stdio.h>
#include <stdint.h>
#include "ptx/ptx.h"
#include "regmp/regmp.h"
#include "digitmp/digitmp.h"
#include "warpmp/warpmp.h"
#include "powm/powm.h"

#define GEOMETRY 128

using namespace xmp;

#include "powm_implementations.h"

template<bool gsl, int size>
__global__ void regmp_add_kernel(add_arguments_t add_arguments, int32_t count) {
  xmpLimb_t  *s_data=add_arguments.out_data;
  int32_t     s_len=add_arguments.out_len;
  int32_t     s_stride=add_arguments.out_stride;
  xmpLimb_t  *a_data=add_arguments.a_data;
  int32_t     a_len=add_arguments.a_len;
  int32_t     a_stride=add_arguments.a_stride;
  int32_t     a_count=add_arguments.a_count;
  xmpLimb_t  *b_data=add_arguments.b_data;
  int32_t     b_len=add_arguments.b_len;
  int32_t     b_stride=add_arguments.b_stride;
  int32_t     b_count=add_arguments.b_count;
  uint32_t   *a_indices=add_arguments.a_indices;
  uint32_t   *b_indices=add_arguments.b_indices;
  uint32_t   *out_indices=add_arguments.out_indices;
  uint32_t    a_indices_count=add_arguments.a_indices_count;
  uint32_t    b_indices_count=add_arguments.b_indices_count;

  PTXInliner  inliner;
  xmpLimb_t   registers[2*size];
  RegMP       A(registers, 0, 0, size), B(registers, 0, size, size);
  xmpLimb_t  *data;
  xmpLimb_t   carry, zero=0;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];

    data=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<size;index++) {
      if(index<a_len)
        A[index]=data[index*a_stride];
      else
        A[index]=0;
    }

    data=b_data + bindex%b_count;
    #pragma unroll
    for(int index=0;index<size;index++) {
      if(index<b_len)
        B[index]=data[index*b_stride];
      else
        B[index]=0;
    }

    _add(A, A, B, false, true);       // addition with carry out set
    inliner.ADDC(carry, zero, zero);  // grab the carry flag

    data=s_data + outindex;
    #pragma unroll
    for(int index=0;index<size;index++)
      if(index<s_len)
        data[index*s_stride]=A[index];

    if(s_len>size)
      data[size*s_stride]=carry;

    #pragma nounroll
    for(int index=size+1;index<s_len;index++)
      data[index*s_stride]=0;

    if(!gsl)
      break;
  }
}

template<bool gsl, int size>
__global__ void digitmp_add_kernel(add_arguments_t add_arguments, int32_t count) {
  xmpLimb_t *s_data=add_arguments.out_data;
  int32_t    s_len=add_arguments.out_len;
  int32_t    s_stride=add_arguments.out_stride;
  xmpLimb_t *a_data=add_arguments.a_data;
  int32_t    a_len=add_arguments.a_len;
  int32_t    a_stride=add_arguments.a_stride;
  int32_t    a_count=add_arguments.a_count;
  xmpLimb_t *b_data=add_arguments.b_data;
  int32_t    b_len=add_arguments.b_len;
  int32_t    b_stride=add_arguments.b_stride;
  int32_t    b_count=add_arguments.b_count;
  uint32_t   *a_indices=add_arguments.a_indices;
  uint32_t   *b_indices=add_arguments.b_indices;
  uint32_t   *out_indices=add_arguments.out_indices;
  uint32_t    a_indices_count=add_arguments.a_indices_count;
  uint32_t    b_indices_count=add_arguments.b_indices_count;

  xmpLimb_t  registers[2*size];

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    uint32_t aindex=thread, bindex=thread, sindex=thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[bindex%b_indices_count];
    if(NULL!=out_indices) sindex=out_indices[sindex];

    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex), B(false, false, b_data, b_len, b_stride, b_count, bindex);
    DigitMP<size> S(false, false, s_data, s_len, s_stride, count, sindex);

    add<size>(registers, S, A, B);

    if(!gsl)
      break;
  }
}

template<bool gsl, int size>
__global__ void regmp_sub_kernel(sub_arguments_t sub_arguments, int32_t count) {
  xmpLimb_t  *d_data=sub_arguments.out_data;
  int32_t     d_len=sub_arguments.out_len;
  int32_t     d_stride=sub_arguments.out_stride;
  xmpLimb_t  *a_data=sub_arguments.a_data;
  int32_t     a_len=sub_arguments.a_len;
  int32_t     a_stride=sub_arguments.a_stride;
  int32_t     a_count=sub_arguments.a_count;
  xmpLimb_t  *b_data=sub_arguments.b_data;
  int32_t     b_len=sub_arguments.b_len;
  int32_t     b_stride=sub_arguments.b_stride;
  int32_t     b_count=sub_arguments.b_count;
  uint32_t   *a_indices=sub_arguments.a_indices;
  uint32_t   *b_indices=sub_arguments.b_indices;
  uint32_t   *out_indices=sub_arguments.out_indices;
  uint32_t    a_indices_count=sub_arguments.a_indices_count;
  uint32_t    b_indices_count=sub_arguments.b_indices_count;

  PTXInliner  inliner;
  xmpLimb_t   registers[2*size];
  RegMP       A(registers, 0, 0, size), B(registers, 0, size, size);
  xmpLimb_t  *data;
  xmpLimb_t   carry, zero=0;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];

    data=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<size;index++) {
      if(index<a_len)
        A[index]=data[index*a_stride];
      else
        A[index]=0;
    }

    data=b_data + bindex%b_count;
    #pragma unroll
    for(int index=0;index<size;index++) {
      if(index<b_len)
        B[index]=data[index*b_stride];
      else
        B[index]=0;
    }

    _sub(A, A, B, false, true);       // subtract with carry out set
    inliner.SUBC(carry, zero, zero);  // grab the carry flag

    data=d_data + outindex;
    #pragma unroll
    for(int index=0;index<size;index++)
      if(index<d_len)
        data[index*d_stride]=A[index];

    #pragma nounroll
    for(int index=size;index<d_len;index++)
      data[index*d_stride]=carry;

    if(!gsl)
      break;
  }
}

template<bool gsl, int size>
__global__ void digitmp_sub_kernel(sub_arguments_t sub_arguments, int32_t count) {
  xmpLimb_t *d_data=sub_arguments.out_data;
  int32_t    d_len=sub_arguments.out_len;
  int32_t    d_stride=sub_arguments.out_stride;
  xmpLimb_t *a_data=sub_arguments.a_data;
  int32_t    a_len=sub_arguments.a_len;
  int32_t    a_stride=sub_arguments.a_stride;
  int32_t    a_count=sub_arguments.a_count;
  xmpLimb_t *b_data=sub_arguments.b_data;
  int32_t    b_len=sub_arguments.b_len;
  int32_t    b_stride=sub_arguments.b_stride;
  int32_t    b_count=sub_arguments.b_count;
  uint32_t   *a_indices=sub_arguments.a_indices;
  uint32_t   *b_indices=sub_arguments.b_indices;
  uint32_t   *out_indices=sub_arguments.out_indices;
  uint32_t    a_indices_count=sub_arguments.a_indices_count;
  uint32_t    b_indices_count=sub_arguments.b_indices_count;

  xmpLimb_t  registers[2*size];

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];
    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex), B(false, false, b_data, b_len, b_stride, b_count, bindex);
    DigitMP<size> D(false, false, d_data, d_len, d_stride, count, outindex);

    sub<size>(registers, D, A, B);

    if(!gsl)
      break;
  }
}

template<bool gsl, int size>
__global__ void regmp_sqr_kernel(sqr_arguments_t sqr_arguments, int32_t count) {
  xmpLimb_t *p_data=sqr_arguments.out_data;
  int32_t    p_len=sqr_arguments.out_len;
  int32_t    p_stride=sqr_arguments.out_stride;
  xmpLimb_t *a_data=sqr_arguments.a_data;
  int32_t    a_len=sqr_arguments.a_len;
  int32_t    a_stride=sqr_arguments.a_stride;
  int32_t    a_count=sqr_arguments.a_count;
  uint32_t   *a_indices=sqr_arguments.a_indices;
  uint32_t   *out_indices=sqr_arguments.out_indices;
  uint32_t    a_indices_count=sqr_arguments.a_indices_count;

  xmpLimb_t  registers[3*size];
  RegMP      A(registers, 0, 0, size), P(registers, 0, size, 2*size);
  xmpLimb_t *data;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];

    data=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<size;index++) {
      if(index<a_len)
        A[index]=data[index*a_stride];
      else
        A[index]=0;
    }

    sqr(P, A);

    data=p_data + outindex;
    #pragma unroll
    for(int index=0;index<2*size;index++)
      if(index<p_len)
        data[index*p_stride]=P[index];

    for(int index=2*size;index<p_len;index++)
      data[index*p_stride]=0;

    if(!gsl)
      break;
  }
}

template<bool gsl, int size>
__global__ void digitmp_sqr_kernel(sqr_arguments_t sqr_arguments, int32_t count) {
  xmpLimb_t *p_data=sqr_arguments.out_data;
  int32_t    p_len=sqr_arguments.out_len;
  int32_t    p_stride=sqr_arguments.out_stride;
  xmpLimb_t *a_data=sqr_arguments.a_data;
  int32_t    a_len=sqr_arguments.a_len;
  int32_t    a_stride=sqr_arguments.a_stride;
  int32_t    a_count=sqr_arguments.a_count;
  uint32_t   *a_indices=sqr_arguments.a_indices;
  uint32_t   *out_indices=sqr_arguments.out_indices;
  uint32_t    a_indices_count=sqr_arguments.a_indices_count;

  xmpLimb_t  registers[4*size+1];

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];
    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex);
    DigitMP<size> P(false, false, p_data, p_len, p_stride, count, outindex);

    sqr<size>(registers, P, A);

    if(!gsl)
      break;
  }
}

template<bool gsl, int a_size, int b_size>
__global__ void regmp_mul_kernel(mul_arguments_t mul_arguments, int32_t count) {
  xmpLimb_t *p_data=mul_arguments.out_data;
  int32_t    p_len=mul_arguments.out_len;
  int32_t    p_stride=mul_arguments.out_stride;
  xmpLimb_t *a_data=mul_arguments.a_data;
  int32_t    a_len=mul_arguments.a_len;
  int32_t    a_stride=mul_arguments.a_stride;
  int32_t    a_count=mul_arguments.a_count;
  xmpLimb_t *b_data=mul_arguments.b_data;
  int32_t    b_len=mul_arguments.b_len;
  int32_t    b_stride=mul_arguments.b_stride;
  int32_t    b_count=mul_arguments.b_count;
  uint32_t   *a_indices=mul_arguments.a_indices;
  uint32_t   *b_indices=mul_arguments.b_indices;
  uint32_t   *out_indices=mul_arguments.out_indices;
  uint32_t    a_indices_count=mul_arguments.a_indices_count;
  uint32_t    b_indices_count=mul_arguments.b_indices_count;

  xmpLimb_t *data;

#ifdef IMAD
  xmpLimb_t  registers[2*a_size+b_size];
  RegMP      A(registers, 0, 0, a_size), B(registers, 0, 2*a_size, b_size), P(registers, 0, a_size, a_size+b_size);
#endif
#ifdef XMAD
  xmpLimb_t  registers[2*a_size+2*b_size];
  RegMP      A(registers, 0, 0, a_size), B(registers, 0, a_size, b_size), P(registers, 0, a_size+b_size, a_size+b_size);
#endif

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];

    data=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<a_size;index++) {
      if(index<a_len)
        A[index]=data[index*a_stride];
      else
        A[index]=0;
    }

    data=b_data + bindex%b_count;
    #pragma unroll
    for(int index=0;index<b_size;index++) {
      if(index<b_len)
        B[index]=data[index*b_stride];
      else
        B[index]=0;
    }

    mul(P, A, B);

    data=p_data + outindex;
    #pragma unroll
    for(int index=0;index<a_size+b_size;index++)
      if(index<p_len)
        data[index*p_stride]=P[index];

    for(int index=a_size+b_size;index<p_len;index++)
      data[index*p_stride]=0;

    if(!gsl)
      break;
  }
}

template<bool gsl, int size>
__global__ void digitmp_mul_kernel(mul_arguments_t mul_arguments, int32_t count) {
  xmpLimb_t *p_data=mul_arguments.out_data;
  int32_t    p_len=mul_arguments.out_len;
  int32_t    p_stride=mul_arguments.out_stride;
  xmpLimb_t *a_data=mul_arguments.a_data;
  int32_t    a_len=mul_arguments.a_len;
  int32_t    a_stride=mul_arguments.a_stride;
  int32_t    a_count=mul_arguments.a_count;
  xmpLimb_t *b_data=mul_arguments.b_data;
  int32_t    b_len=mul_arguments.b_len;
  int32_t    b_stride=mul_arguments.b_stride;
  int32_t    b_count=mul_arguments.b_count;
  uint32_t   *a_indices=mul_arguments.a_indices;
  uint32_t   *b_indices=mul_arguments.b_indices;
  uint32_t   *out_indices=mul_arguments.out_indices;
  uint32_t    a_indices_count=mul_arguments.a_indices_count;
  uint32_t    b_indices_count=mul_arguments.b_indices_count;

  xmpLimb_t  registers[4*size+1];

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];
    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex), B(false, false, b_data, b_len, b_stride, b_count, bindex);
    DigitMP<size> P(false, false, p_data, p_len, p_stride, count, outindex);

    mul<size>(registers, P, A, B);

    if(!gsl)
      break;
  }
}

template<bool gsl, int a_size, int b_size>
__global__ void regmp_div_kernel(div_arguments_t div_arguments, int32_t count) {
  xmpLimb_t *q_data=div_arguments.out_data;
  int32_t    q_len=div_arguments.out_len;
  int32_t    q_stride=div_arguments.out_stride;
  xmpLimb_t *a_data=div_arguments.a_data;
  int32_t    a_len=div_arguments.a_len;
  int32_t    a_stride=div_arguments.a_stride;
  int32_t    a_count=div_arguments.a_count;
  xmpLimb_t *b_data=div_arguments.b_data;
  int32_t    b_len=div_arguments.b_len;
  int32_t    b_stride=div_arguments.b_stride;
  int32_t    b_count=div_arguments.b_count;
  uint32_t   *a_indices=div_arguments.a_indices;
  uint32_t   *b_indices=div_arguments.b_indices;
  uint32_t   *out_indices=div_arguments.out_indices;
  uint32_t    a_indices_count=div_arguments.a_indices_count;
  uint32_t    b_indices_count=div_arguments.b_indices_count;

  xmpLimb_t  registers[a_size+b_size+2];
  RegMP      A(registers, 0, 0, a_size+2), B(registers, 0, a_size+2, b_size);
  xmpLimb_t *data;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];

    data=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<a_size;index++) {
      if(index<a_len)
        A[index]=data[index*a_stride];
      else
        A[index]=0;
    }
    A[a_size]=0;
    A[a_size+1]=0;

    data=b_data + bindex%b_count;
    #pragma unroll
    for(int index=0;index<b_size;index++) {
      if(index<b_len)
        B[index]=data[index*b_stride];
      else
        B[index]=0;
    }

    div(A, B);

    data=q_data + outindex;
    #pragma unroll
    for(int index=0;index<a_size;index++)
      data[index*q_stride]=A[index];

    for(int index=a_size;index<q_len;index++)
      data[index*q_stride]=0;

    if(!gsl)
      break;
  }
}

template<bool gsl, int size>
__global__ void digitmp_div_kernel(div_arguments_t div_arguments, int32_t count) {
  xmpLimb_t *q_data=div_arguments.out_data;
  int32_t    q_len=div_arguments.out_len;
  int32_t    q_stride=div_arguments.out_stride;
  xmpLimb_t *a_data=div_arguments.a_data;
  int32_t    a_len=div_arguments.a_len;
  int32_t    a_stride=div_arguments.a_stride;
  int32_t    a_count=div_arguments.a_count;
  xmpLimb_t *b_data=div_arguments.b_data;
  int32_t    b_len=div_arguments.b_len;
  int32_t    b_stride=div_arguments.b_stride;
  int32_t    b_count=div_arguments.b_count;
  xmpLimb_t *scratch=div_arguments.scratch;
  uint32_t   *a_indices=div_arguments.a_indices;
  uint32_t   *b_indices=div_arguments.b_indices;
  uint32_t   *out_indices=div_arguments.out_indices;
  uint32_t    a_indices_count=div_arguments.a_indices_count;
  uint32_t    b_indices_count=div_arguments.b_indices_count;

  xmpLimb_t  registers[4*size+4];

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];

    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex), B(false, false, b_data, b_len, b_stride, b_count, bindex);
    DigitMP<size> Q(false, false, q_data, q_len, q_stride, count, outindex);

    div<size>(registers, Q, A, B, scratch);

    if(!gsl)
      break;
  }
}

template<bool gsl, int a_size, int b_size>
__global__ void regmp_mod_kernel(mod_arguments_t mod_arguments, int32_t count) {
  xmpLimb_t *r_data=mod_arguments.out_data;
  int32_t    r_len=mod_arguments.out_len;
  int32_t    r_stride=mod_arguments.out_stride;
  xmpLimb_t *a_data=mod_arguments.a_data;
  int32_t    a_len=mod_arguments.a_len;
  int32_t    a_stride=mod_arguments.a_stride;
  int32_t    a_count=mod_arguments.a_count;
  xmpLimb_t *b_data=mod_arguments.b_data;
  int32_t    b_len=mod_arguments.b_len;
  int32_t    b_stride=mod_arguments.b_stride;
  int32_t    b_count=mod_arguments.b_count;
  uint32_t   *a_indices=mod_arguments.a_indices;
  uint32_t   *b_indices=mod_arguments.b_indices;
  uint32_t   *out_indices=mod_arguments.out_indices;
  uint32_t    a_indices_count=mod_arguments.a_indices_count;
  uint32_t    b_indices_count=mod_arguments.b_indices_count;

  xmpLimb_t  registers[a_size+b_size+2];
  RegMP      A(registers, 0, 0, a_size+2), B(registers, 0, a_size+2, b_size);
  xmpLimb_t *data;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];

    data=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<a_size;index++) {
      if(index<a_len)
        A[index]=data[index*a_stride];
      else
        A[index]=0;
    }
    A[a_size]=0;
    A[a_size+1]=0;

    data=b_data + bindex%b_count;
    #pragma unroll
    for(int index=0;index<b_size;index++) {
      if(index<b_len)
        B[index]=data[index*b_stride];
      else
        B[index]=0;
    }

    rem(A, B);

    data=r_data + outindex;
    #pragma unroll
    for(int index=0;index<b_size;index++)
      data[index*r_stride]=A[index];

    for(int index=b_size;index<r_len;index++)
      data[index*r_stride]=0;

    if(!gsl)
      break;
  }
}
template<bool gsl, int size>
__global__ void digitmp_mod_kernel(mod_arguments_t mod_arguments, int32_t count) {
  xmpLimb_t *r_data=mod_arguments.out_data;
  int32_t    r_len=mod_arguments.out_len;
  int32_t    r_stride=mod_arguments.out_stride;
  xmpLimb_t *a_data=mod_arguments.a_data;
  int32_t    a_len=mod_arguments.a_len;
  int32_t    a_stride=mod_arguments.a_stride;
  int32_t    a_count=mod_arguments.a_count;
  xmpLimb_t *b_data=mod_arguments.b_data;
  int32_t    b_len=mod_arguments.b_len;
  int32_t    b_stride=mod_arguments.b_stride;
  int32_t    b_count=mod_arguments.b_count;
  xmpLimb_t *scratch=mod_arguments.scratch;
  uint32_t   *a_indices=mod_arguments.a_indices;
  uint32_t   *b_indices=mod_arguments.b_indices;
  uint32_t   *out_indices=mod_arguments.out_indices;
  uint32_t    a_indices_count=mod_arguments.a_indices_count;
  uint32_t    b_indices_count=mod_arguments.b_indices_count;

  xmpLimb_t  registers[4*size+4];

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];

    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex), B(false, false, b_data, b_len, b_stride, b_count, bindex);
    DigitMP<size> R(false, false, r_data, r_len, r_stride, count, outindex);

    rem<size>(registers, R, A, B, scratch);

    if(!gsl)
      break;
  }
}

template<bool gsl, int a_size, int b_size>
__global__ void regmp_divmod_kernel(divmod_arguments_t divmod_arguments, int32_t count) {
  xmpLimb_t *q_data=divmod_arguments.q_data;
  int32_t    q_len=divmod_arguments.q_len;
  int32_t    q_stride=divmod_arguments.q_stride;
  xmpLimb_t *r_data=divmod_arguments.m_data;
  int32_t    r_len=divmod_arguments.m_len;
  int32_t    r_stride=divmod_arguments.m_stride;
  xmpLimb_t *a_data=divmod_arguments.a_data;
  int32_t    a_len=divmod_arguments.a_len;
  int32_t    a_stride=divmod_arguments.a_stride;
  int32_t    a_count=divmod_arguments.a_count;
  xmpLimb_t *b_data=divmod_arguments.b_data;
  int32_t    b_len=divmod_arguments.b_len;
  int32_t    b_stride=divmod_arguments.b_stride;
  int32_t    b_count=divmod_arguments.b_count;
  uint32_t   *a_indices=divmod_arguments.a_indices;
  uint32_t   *b_indices=divmod_arguments.b_indices;
  uint32_t   *q_indices=divmod_arguments.q_indices;
  uint32_t   *r_indices=divmod_arguments.r_indices;
  uint32_t    a_indices_count=divmod_arguments.a_indices_count;
  uint32_t    b_indices_count=divmod_arguments.b_indices_count;

  xmpLimb_t  registers[a_size+b_size+2];
  RegMP      A(registers, 0, 0, a_size+2), B(registers, 0, a_size+2, b_size);
  xmpLimb_t *data;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, qindex=thread, rindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=q_indices) qindex=q_indices[thread];
    if(NULL!=r_indices) rindex=r_indices[thread];

    data=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<a_size;index++) {
      if(index<a_len)
        A[index]=data[index*a_stride];
      else
        A[index]=0;
    }
    A[a_size]=0;
    A[a_size+1]=0;

    data=b_data + bindex%b_count;
    #pragma unroll
    for(int index=0;index<b_size;index++) {
      if(index<b_len)
        B[index]=data[index*b_stride];
      else
        B[index]=0;
    }

    div_rem(A, B);

    data=q_data + qindex;
    #pragma unroll
    for(int index=0;index<a_size;index++)
      data[index*q_stride]=A[index];

    for(int index=a_size;index<q_len;index++)
      data[index*q_stride]=0;

    data=r_data + rindex;
    #pragma unroll
    for(int index=0;index<b_size;index++)
      data[index*r_stride]=B[index];

    for(int index=b_size;index<r_len;index++)
      data[index*r_stride]=0;

    if(!gsl)
      break;
  }
}

template<bool gsl, int size>
__global__ void digitmp_divmod_kernel(divmod_arguments_t divmod_arguments, int32_t count) {
  xmpLimb_t *q_data=divmod_arguments.q_data;
  int32_t    q_len=divmod_arguments.q_len;
  int32_t    q_stride=divmod_arguments.q_stride;
  xmpLimb_t *r_data=divmod_arguments.m_data;
  int32_t    r_len=divmod_arguments.m_len;
  int32_t    r_stride=divmod_arguments.m_stride;
  xmpLimb_t *a_data=divmod_arguments.a_data;
  int32_t    a_len=divmod_arguments.a_len;
  int32_t    a_stride=divmod_arguments.a_stride;
  int32_t    a_count=divmod_arguments.a_count;
  xmpLimb_t *b_data=divmod_arguments.b_data;
  int32_t    b_len=divmod_arguments.b_len;
  int32_t    b_stride=divmod_arguments.b_stride;
  int32_t    b_count=divmod_arguments.b_count;
  xmpLimb_t *scratch=divmod_arguments.scratch;
  uint32_t   *a_indices=divmod_arguments.a_indices;
  uint32_t   *b_indices=divmod_arguments.b_indices;
  uint32_t   *q_indices=divmod_arguments.q_indices;
  uint32_t   *r_indices=divmod_arguments.r_indices;
  uint32_t    a_indices_count=divmod_arguments.a_indices_count;
  uint32_t    b_indices_count=divmod_arguments.b_indices_count;

  xmpLimb_t  registers[4*size+4];

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, qindex=thread, rindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=q_indices) qindex=q_indices[thread];
    if(NULL!=r_indices) rindex=r_indices[thread];

    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex), B(false, false, b_data, b_len, b_stride, b_count, bindex);
    DigitMP<size> Q(false, false, q_data, q_len, q_stride, count, qindex), R(false, false, r_data, r_len, r_stride, count, rindex);

    div_rem<size>(registers, Q, R, A, B, scratch);

    if(!gsl)
      break;
  }
}

template<bool gsl>
__global__ void strided_compare_kernel(cmp_arguments_t cmp_arguments, int32_t count) {
  int32_t   *c_data=cmp_arguments.out_data;
  xmpLimb_t *l_data=cmp_arguments.a_data;
  int32_t    l_len=cmp_arguments.a_len;
  int32_t    l_stride=cmp_arguments.a_stride;
  int32_t    l_count=cmp_arguments.a_count;
  xmpLimb_t *s_data=cmp_arguments.b_data;
  int32_t    s_len=cmp_arguments.b_len;
  int32_t    s_stride=cmp_arguments.b_stride;
  int32_t    s_count=cmp_arguments.b_count;
  int32_t    negate=cmp_arguments.negate;
  uint32_t   *l_indices=cmp_arguments.a_indices;
  uint32_t   *s_indices=cmp_arguments.b_indices;
  uint32_t    l_indices_count=cmp_arguments.a_indices_count;
  uint32_t    s_indices_count=cmp_arguments.b_indices_count;

  xmpLimb_t *l, *s;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int lindex=thread, sindex=thread;
    if(NULL!=l_indices) lindex=l_indices[lindex%l_indices_count];
    if(NULL!=s_indices) sindex=s_indices[sindex%s_indices_count];
    int32_t   result=0;

    l=l_data + lindex%l_count;
    s=s_data + sindex%s_count;

    for(int index=l_len-1;index>=s_len;index--)
      if(result==0 && l_data[index*l_stride]!=0)
        result=negate;

    for(int index=s_len-1;index>=0;index--) {
      if(result==0) {
        uint32_t l_value=l[index*l_stride], s_value=s[index*s_stride];

        if(l_value>s_value) {
          result=negate;
          break;
        }
        if(l_value<s_value) {
          result=-negate;
          break;
        }
      }
    }

    c_data[thread]=result;

    if(!gsl)
      break;
  }
}

template<bool gsl>
__global__ void strided_popc_kernel(popc_arguments_t popc_arguments, uint32_t count) {
  uint32_t  *c_data=popc_arguments.out_data;
  xmpLimb_t *a_data=popc_arguments.a_data;
  int32_t    a_len=popc_arguments.a_len;
  int32_t    a_stride=popc_arguments.a_stride;
  int32_t    a_count=popc_arguments.a_count;
  uint32_t   *a_indices=popc_arguments.a_indices;
  uint32_t    a_indices_count=popc_arguments.a_indices_count;

  xmpLimb_t *data;
  uint32_t   result;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    result=0;

    data=a_data + aindex%a_count;
    for(uint32_t index=0;index<a_len;index++)
      result+=__popc(data[index*a_stride]);
    c_data[thread]=result;

    if(!gsl)
      break;
  }
}

template<bool gsl>
__global__ void strided_ior_kernel(ior_arguments_t ior_arguments, int32_t count) {
  xmpLimb_t *c_data=ior_arguments.out_data;
  int32_t    c_len=ior_arguments.out_len;
  int32_t    c_stride=ior_arguments.out_stride;
  xmpLimb_t *l_data=ior_arguments.a_data;
  int32_t    l_len=ior_arguments.a_len;
  int32_t    l_stride=ior_arguments.a_stride;
  int32_t    l_count=ior_arguments.a_count;
  xmpLimb_t *s_data=ior_arguments.b_data;
  int32_t    s_len=ior_arguments.b_len;
  int32_t    s_stride=ior_arguments.b_stride;
  int32_t    s_count=ior_arguments.b_count;
  uint32_t   *l_indices=ior_arguments.a_indices;
  uint32_t   *s_indices=ior_arguments.b_indices;
  uint32_t    l_indices_count=ior_arguments.a_indices_count;
  uint32_t    s_indices_count=ior_arguments.b_indices_count;
  uint32_t    *out_indices=ior_arguments.out_indices;

  xmpLimb_t *l, *s, *c;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int lindex=thread, sindex=thread, outindex=thread;
    if(NULL!=l_indices) lindex=l_indices[lindex%l_indices_count];
    if(NULL!=s_indices) sindex=s_indices[sindex%s_indices_count];
    if(NULL!=out_indices) outindex=out_indices[outindex];

    l=l_data + lindex%l_count;
    s=s_data + sindex%s_count;
    c=c_data + outindex;

    for(int index=0;index<s_len;index++)
      c[index*c_stride]=l[index*l_stride] | s[index*s_stride];

    for(int index=s_len;index<l_len;index++)
      c[index*c_stride]=l[index*l_stride];

    for(int index=l_len;index<c_len;index++)
      c[index*c_stride]=0;

    if(!gsl)
      break;
  }
}

template<bool gsl>
__global__ void strided_and_kernel(and_arguments_t and_arguments, int32_t count) {
  xmpLimb_t *c_data=and_arguments.out_data;
  int32_t    c_len=and_arguments.out_len;
  int32_t    c_stride=and_arguments.out_stride;
  xmpLimb_t *l_data=and_arguments.a_data;
  int32_t    l_stride=and_arguments.a_stride;
  int32_t    l_count=and_arguments.a_count;
  xmpLimb_t *s_data=and_arguments.b_data;
  int32_t    s_len=and_arguments.b_len;
  int32_t    s_stride=and_arguments.b_stride;
  int32_t    s_count=and_arguments.b_count;
  uint32_t   *l_indices=and_arguments.a_indices;
  uint32_t   *s_indices=and_arguments.b_indices;
  uint32_t    l_indices_count=and_arguments.a_indices_count;
  uint32_t    s_indices_count=and_arguments.b_indices_count;
  uint32_t    *out_indices=and_arguments.out_indices;

  xmpLimb_t *l, *s, *c;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int lindex=thread, sindex=thread, outindex=thread;
    if(NULL!=l_indices) lindex=l_indices[lindex%l_indices_count];
    if(NULL!=s_indices) sindex=s_indices[sindex%s_indices_count];
    if(NULL!=out_indices) outindex=out_indices[outindex];

    l=l_data + lindex%l_count;
    s=s_data + sindex%s_count;
    c=c_data + outindex;

    for(int index=0;index<s_len;index++)
      c[index*c_stride]=l[index*l_stride] & s[index*s_stride];

    for(int index=s_len;index<c_len;index++)
      c[index*c_stride]=0;

    if(!gsl)
      break;
  }
}

template<bool gsl>
__global__ void strided_xor_kernel(xor_arguments_t xor_arguments, int32_t count) {
  xmpLimb_t *c_data=xor_arguments.out_data;
  int32_t    c_len=xor_arguments.out_len;
  int32_t    c_stride=xor_arguments.out_stride;
  xmpLimb_t *l_data=xor_arguments.a_data;
  int32_t    l_len=xor_arguments.a_len;
  int32_t    l_stride=xor_arguments.a_stride;
  int32_t    l_count=xor_arguments.a_count;
  xmpLimb_t *s_data=xor_arguments.b_data;
  int32_t    s_len=xor_arguments.b_len;
  int32_t    s_stride=xor_arguments.b_stride;
  int32_t    s_count=xor_arguments.b_count;
  uint32_t   *l_indices=xor_arguments.a_indices;
  uint32_t   *s_indices=xor_arguments.b_indices;
  uint32_t    l_indices_count=xor_arguments.a_indices_count;
  uint32_t    s_indices_count=xor_arguments.b_indices_count;
  uint32_t    *out_indices=xor_arguments.out_indices;

  xmpLimb_t *l, *s, *c;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int lindex=thread, sindex=thread, outindex=thread;
    if(NULL!=l_indices) lindex=l_indices[lindex%l_indices_count];
    if(NULL!=s_indices) sindex=s_indices[sindex%s_indices_count];
    if(NULL!=out_indices) outindex=out_indices[outindex];

    l=l_data + lindex%l_count;
    s=s_data + sindex%s_count;
    c=c_data + outindex;

    for(int index=0;index<s_len;index++)
      c[index*c_stride]=l[index*l_stride] ^ s[index*s_stride];

    for(int index=s_len;index<l_len;index++)
      c[index*c_stride]=l[index*l_stride];

    for(int index=l_len;index<c_len;index++)
      c[index*c_stride]=0;

    if(!gsl)
      break;
  }
}

template<bool gsl>
__global__ void strided_not_kernel(not_arguments_t not_arguments, int32_t count) {
  xmpLimb_t *c_data=not_arguments.out_data;
  int32_t    c_len=not_arguments.out_len;
  int32_t    c_stride=not_arguments.out_stride;
  xmpLimb_t *a_data=not_arguments.a_data;
  int32_t    a_len=not_arguments.a_len;
  int32_t    a_stride=not_arguments.a_stride;
  int32_t    a_count=not_arguments.a_count;
  uint32_t   *a_indices=not_arguments.a_indices;
  uint32_t   *c_indices=not_arguments.out_indices;
  uint32_t    a_indices_count=not_arguments.a_indices_count;

  xmpLimb_t *a, *c;

  #pragma nounroll
  for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, cindex=thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=c_indices) cindex=c_indices[cindex];

    a=a_data + aindex%a_count;
    c=c_data + cindex;

    for(int index=0;index<a_len;index++) {
      c[index*c_stride]=~a[index*a_stride];
    }

    for(int index=a_len;index<c_len;index++)
      c[index*c_stride]=~0;

    if(!gsl)
      break;
  }
}

template<bool gsl>
__global__ void strided_shf_kernel(shf_arguments_t shf_arguments, uint32_t count) {
  xmpLimb_t                *c_data=shf_arguments.out_data;
  int32_t                   c_len=shf_arguments.out_len;
  int32_t                   c_stride=shf_arguments.out_stride;
  xmpLimb_t * __restrict__  a_data=shf_arguments.a_data;
  int32_t                   a_len=shf_arguments.a_len;
  int32_t                   a_stride=shf_arguments.a_stride;
  int32_t                   a_count=shf_arguments.a_count;
  int32_t                  *shift_data=shf_arguments.shift_data;
  int32_t                   shift_count=shf_arguments.shift_count;
  uint32_t                 *a_indices=shf_arguments.a_indices;
  uint32_t                 *out_indices=shf_arguments.out_indices;
  uint32_t                  a_indices_count=shf_arguments.a_indices_count;

  int32_t                   bits_per_limb=sizeof(xmpLimb_t)*8;

  #pragma nounroll
  for(uint32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=out_indices) outindex=out_indices[outindex];

    int32_t shift=-shift_data[thread%shift_count];

    xmpLimb_t *out=c_data+outindex;
    const xmpLimb_t * __restrict__ in=a_data+aindex%a_count;
    int32_t limb_off = shift/bits_per_limb;
    int32_t bit_off = (shift%bits_per_limb+bits_per_limb)%bits_per_limb; //compute positive modulo

    //shift limb down where we need round to -infinity instead of 0
    if(shift<=0 && bit_off!=0) limb_off--;

    //set src_off
    int32_t src_off=limb_off;

    //campute shifts
    uint32_t shr=bit_off;
    uint32_t shl=bits_per_limb-bit_off;

    xmpLimb_t low, high;
    //read in high;
    high=( src_off>=0 && src_off<a_len ) ? in[src_off*a_stride] : 0;

    for(uint32_t i=0;i<c_len;i++) {
      xmpLimb_t limb;
      //advance forward

      src_off++;
      low=high;

      //read in new high
      high=( src_off>=0 && src_off<a_len ) ? in[src_off*a_stride] : 0;
      limb= (high<<shl) |  (low>>shr);

      //write result
      out[i*c_stride]=limb;
    }

    if(!gsl)
      break;
  }
}

