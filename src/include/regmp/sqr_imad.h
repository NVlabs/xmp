/***
Copyright (c) 2014-2015, Niall Emmart.  All rights reserved.
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

namespace xmp {
  __device__ __forceinline__ void sqr(RegMP rr, RegMP a) {
    PTXInliner inliner;
    int        i, j, length=a.length();
    uint32_t   zero=0;

    if(2*a.length()!=rr.length()) RMP_ERROR("square() - length mismatch");
    if(rr.overlap(a)) RMP_ERROR("square() - overlap error");

    #pragma unroll
    for(i=1;i<length;i++)
      inliner.MULLO(rr[i], a[0], a[i]);

    PTXChain chain1(length-1);
    #pragma unroll
    for(i=1;i<length-1;i++)
      chain1.MADHI(rr[i+1], a[0], a[i], rr[i+1]);
    chain1.MADHI(rr[length], a[0], a[length-1], zero);
    chain1.end();

    #pragma unroll
    for(j=1;j<length-1;j++) {
      PTXChain chain2(length-j);
      #pragma unroll
      for(i=2;i<length;i++)
        if(i>=j+1)
          chain2.MADLO(rr[i+j], a[i], a[j], rr[i+j]);
      chain2.ADD(rr[i+j], zero, zero);
      chain2.end();

      PTXChain chain3(length-j-1);
      #pragma unroll
      for(i=2;i<length;i++)
        if(i>=j+1)
          chain3.MADHI(rr[i+j+1], a[i], a[j], rr[i+j+1]);
      chain3.end();
    }

    PTXChain chain4(length*2-1);
    #pragma unroll
    for(i=1;i<2*length-1;i++)
      chain4.ADD(rr[i], rr[i], rr[i]);
    chain4.ADD(rr[i], zero, zero);
    chain4.end();

    PTXChain chain5(a.length()*2-1);
    inliner.MULLO(rr[0], a[0], a[0]);
    chain5.MADHI(rr[1], a[0], a[0], rr[1]);
    #pragma unroll
    for(i=1;i<length;i++) {
     chain5.MADLO(rr[i*2], a[i], a[i], rr[i*2]);
     chain5.MADHI(rr[i*2+1], a[i], a[i], rr[i*2+1]);
    }
    chain5.end();
  }

  __device__ __forceinline__ void accumulate_half_sqr(RegMP acc, RegMP a) {
    RegMP      inner, outer;
    PTXInliner inliner;
    int        i, j;
    uint32_t   high, zero=0;

    if(acc.length()<2*a.length()) RMP_ERROR("accumulate_half_sqr() - length mismatch");
    if(acc.overlap(a)) RMP_ERROR("accumulate_half_sqr() - acc overlaps a or b");

    #pragma unroll
    for(i=0;i<a.length()-1;i++) {
      PTXChain chain1(a.length()-i+1);
      #pragma unroll
      for(j=0;j<a.length();j++)
        if(j>i)
          chain1.MADLO(acc[i+j], a[i], a[j], acc[i+j]);
      chain1.ADD(acc[i+j], acc[i+j], (i==0) ? zero : high);
      chain1.ADD(high, zero, zero);
      chain1.end();

      PTXChain chain2(a.length()-i-1, false, true);
      #pragma unroll
      for(j=0;j<a.length();j++)
        if(j>i)
          chain2.MADHI(acc[i+j+1], a[i], a[j], acc[i+j+1]);
      chain2.end();

      if(i<a.length()-2)
        inliner.ADDC(high, high, zero);
    }

    inliner.ADDC_CC(acc[2*a.length()-1], acc[2*a.length()-1], high);
    #pragma unroll
    for(int word=2*a.length();word<acc.length();word++)
      inliner.ADDC_CC(acc[word], acc[word], zero);
  }
}

/*
  // slightly faster squaring, but the compiler breaks down for a>11 words

  __device__ __forceinline__ void sqr(RegMP rr, RegMP a) {
    int      i, j, adjusted;
    uint32_t zero=0;

    if(2*a.length()!=rr.length()) RMP_ERROR("square() - length mismatch");
    if(rr.overlap(a)) RMP_ERROR("square() - overlap error");

    #pragma unroll
    for(i=0;i<a.length()-1;i++) {
      PTXChain chain1(a.length());
      #pragma unroll
      for(j=0;j<2*a.length();j++) {
        adjusted=i;
        if(j>=a.length()-1)
          adjusted=i+j+1-a.length();
        if(j>i && (j-(adjusted+1)/2)<a.length()) {
          if(adjusted%2==0)
            chain1.MADLO(rr[j], a[j-adjusted/2], a[adjusted/2], (i==0) ? zero : rr[j]);
          else
            chain1.MADHI(rr[j], a[j-(adjusted+1)/2], a[adjusted/2], (j-(adjusted+1)/2==a.length()-1) ? zero : rr[j]);
        }
      }
      chain.end();
    }

    PTXChain chain2(a.length()*2-1);
    #pragma unroll
    for(i=1;i<2*a.length()-1;i++)
      chain2.ADD(rr[i], rr[i], rr[i]);
    chain2.ADD(rr[i], zero, zero);
    chain2.end();

    PTXChain chain3(a.length()*2-1);
    inliner.MULLO(rr[0], a[0], a[0]);
    chain3.MADHI(rr[1], a[0], a[0], rr[1]);
    #pragma unroll
    for(i=1;i<a.length();i++) {
     chain3.MADLO(rr[i*2], a[i], a[i], rr[i*2]);
     chain3.MADHI(rr[i*2+1], a[i], a[i], rr[i*2+1]);
    }
    chain.end();
  }

  // an experiment -- thought it might accelerate one level of Karatsuba squaring,
  // unfortunately, it doesn't help with performance, because of a dangling carry

  __device__ __forceinline__ void _sqr_2b(RegMP rr, RegMP a, RegMP b, uint32_t &carry) {
    int      i, j, length=a.length();
    uint32_t temp, zero=0;

    // compute a*a+2*b*(1<<len(a))

    if(2*a.length()!=rr.length()) RMP_ERROR("square() - length mismatch");
    if(rr.overlap(a)) RMP_ERROR("square() - overlap error");

    #pragma unroll
    for(i=1;i<length;i++)
      inliner.MULLO(rr[i], a[0], a[i]);

    PTXChain chain1(length-1);
    #pragma unroll
    for(i=1;i<length-1;i++)
      chain1.MADHI(rr[i+1], a[0], a[i], rr[i+1]);
    chain1.MADHI(rr[length], a[0], a[length-1], zero);
    chain1.end();

    #pragma unroll
    for(j=1;j<length-1;j++) {
      PTXChain chain2(length-j);
      #pragma unroll
      for(i=2;i<length;i++)
        if(i>=j+1)
          chain2.MADLO(rr[i+j], a[i], a[j], rr[i+j]);
      chain2.ADD(rr[i+j], zero, zero);
      chain2.end();

      PTXChain chain3(length-j-1);
      #pragma unroll
      for(i=2;i<length;i++)
        if(i>=j+1)
          chain3.MADHI(rr[i+j+1], a[i], a[j], rr[i+j+1]);
      chain3.end();
    }

    PTXChain chain4(length+1);
    #pragma unroll
    for(i=length;i<2*length-1;i++)
      chain4.ADD(rr[i], rr[i], b[i-length]);
    chain4.ADD(rr[i], zero, b[i-length]);
    chain4.ADD(temp, zero, zero);
    chain4.end();

    PTXChain chain5(length*2);
    #pragma unroll
    for(i=1;i<2*length-1;i++)
      chain5.ADD(rr[i], rr[i], rr[i]);
    chain5.ADD(rr[i], rr[i], rr[i]);
    chain5.ADD(temp, temp, temp);
    chain5.end();

    PTXChain chain6(length*2);
    inliner.MULLO(rr[0], a[0], a[0]);
    chain6.MADHI(rr[1], a[0], a[0], rr[1]);
    #pragma unroll
    for(i=1;i<length;i++) {
     chain6.MADLO(rr[i*2], a[i], a[i], rr[i*2]);
     chain6.MADHI(rr[i*2+1], a[i], a[i], rr[i*2+1]);
    }
    chain6.ADD(carry, carry, temp);
    chain6.end();
  }
*/
