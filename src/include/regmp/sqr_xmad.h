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

namespace xmp {
  __device__ __forceinline__ void sqr(RegMP rr, RegMP a) {
    PTXInliner inliner;
    uint32_t   carry, zero=0, permuteShift16=0x5432;
    int        i, j;

    if(rr.length()!=2*a.length()) RMP_ERROR("sqr() - length mismatch");
    if(rr.overlap(a)) RMP_ERROR("sqr() - rr overlaps a");

    #pragma unroll
    for(i=0;i<rr.length();i++)
      rr[i]=0;

    #pragma unroll
    for(i=0;i<a.length();i++) {
      PTXChain chain1(a.length()-i);
      #pragma unroll
      for(j=0;j<a.length();j++)
        if(j>=i)
          chain1.XMADHL(rr[i+j], a[j], a[i], rr[i+j]);
      chain1.end();

      if(i<a.length()-1) {
        PTXChain chain2(a.length()-i);
        #pragma unroll
        for(j=0;j<a.length();j++)
          if(j>i)
            chain2.XMADLH(rr[i+j], a[j], a[i], rr[i+j]);
        chain2.ADD(rr[i+j], zero, zero);
        chain2.end();
      }
    }

    // shift RR left by 16 bits
    #pragma unroll
    for(i=rr.length()-1;i>0;i--)
      inliner.PERMUTE(rr[i], rr[i-1], rr[i], permuteShift16);
    inliner.PERMUTE(rr[0], zero, rr[0], permuteShift16);

    #pragma unroll
    for(i=0;i<a.length()-1;i++) {
      PTXChain chain3(a.length()-i+1);
      #pragma unroll
      for(j=0;j<a.length();j++)
        if(j>i)
          chain3.XMADLL(rr[i+j], a[j], a[i], rr[i+j]);
      chain3.ADD(rr[i+j], rr[i+j], (i==0) ? zero : carry);
      chain3.ADD(carry, zero, zero);
      chain3.end();

      PTXChain chain4(a.length()-i-1, false, true);
      #pragma unroll
      for(j=0;j<a.length();j++)
        if(j>i)
          chain4.XMADHH(rr[i+j+1], a[j], a[i], rr[i+j+1]);
      chain4.end();

      if(i<a.length()-2)  // can't carry out of the last term
       inliner.ADDC(carry, carry, zero);
      else
       inliner.ADDC(rr[i+j+1], rr[i+j+1], carry);
    }

    PTXChain chain5(rr.length());
    #pragma unroll
    for(i=0;i<rr.length();i++)
      chain5.ADD(rr[i], rr[i], rr[i]);
    chain5.end();

    PTXChain chain6(rr.length());
    #pragma unroll
    for(i=0;i<a.length();i++) {
      chain6.XMADLL(rr[2*i], a[i], a[i], rr[2*i]);
      chain6.XMADHH(rr[2*i+1], a[i], a[i], rr[2*i+1]);
    }
    chain6.end();
  }

  // FIX FIX FIX - this is the IMAD algorithm
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
