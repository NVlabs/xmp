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
  __device__ __forceinline__ void mul(RegMP rr, RegMP a, RegMP b) {
    RegMP      inner, outer;
    PTXInliner inliner;
    uint32_t   carry, zero=0, permuteShift16=0x5432;
    int        i, j;

    if(rr.length()!=a.length()+b.length()) RMP_ERROR("mul() - length mismatch");
    if(rr.overlap(a) || rr.overlap(b)) RMP_ERROR("mul() - rr overlaps a or b");

    if(a.length()>=b.length()) {
      inner=a;
      outer=b;
    }
    else {
      inner=b;
      outer=a;
    }

    #pragma unroll
    for(i=0;i<rr.length();i++)
      rr[i]=0;

    #pragma unroll
    for(i=0;i<outer.length();i++) {
      PTXChain chain1(inner.length()+1);
      #pragma unroll
      for(j=0;j<inner.length();j++)
        chain1.XMADLH(rr[i+j], inner[j], outer[i], rr[i+j]);
      chain1.ADD(rr[i+j], zero, zero);
      chain1.end();

      PTXChain chain2(inner.length()+1);
      #pragma unroll
      for(j=0;j<inner.length();j++)
        chain2.XMADHL(rr[i+j], inner[j], outer[i], rr[i+j]);
      chain2.ADD(rr[i+j], rr[i+j], zero);
      chain2.end();
    }

    // shift RR left by 16 bits
    #pragma unroll
    for(i=rr.length()-1;i>0;i--)
      inliner.PERMUTE(rr[i], rr[i-1], rr[i], permuteShift16);
    inliner.PERMUTE(rr[0], zero, rr[0], permuteShift16);

    #pragma unroll
    for(i=0;i<outer.length();i++) {
      PTXChain chain3(inner.length()+2);
      #pragma unroll
      for(j=0;j<inner.length();j++)
        chain3.XMADLL(rr[i+j], inner[j], outer[i], rr[i+j]);
      chain3.ADD(rr[i+j], rr[i+j], (i==0) ? zero : carry);
      chain3.ADD(carry, zero, zero);
      chain3.end();

      PTXChain chain4(inner.length(), false, true);
      #pragma unroll
      for(j=0;j<inner.length();j++)
        chain4.XMADHH(rr[i+j+1], inner[j], outer[i], rr[i+j+1]);
      chain4.end();

      if(i<outer.length()-1)
       inliner.ADDC(carry, carry, zero);
    }
  }

  __device__ __forceinline__ void accumulate_mul(RegMP acc, RegMP a, RegMP b) {
    RegMP      inner, outer;
    PTXInliner inliner;
    int        i, j;
    uint32_t   carry, zero=0;

    __syncthreads();
    if(acc.length()<a.length()+b.length()) RMP_ERROR("accumulate_mul() - length mismatch");
    if(acc.overlap(a) || acc.overlap(b)) RMP_ERROR("accumulate_mul() - acc overlaps a or b");

    if(a.length()>=b.length()) {
      inner=a;
      outer=b;
    }
    else {
      inner=b;
      outer=a;
    }

    _rock(acc);

    #pragma unroll
    for(i=0;i<outer.length();i++) {
      PTXChain chain1(inner.length()+1);
      #pragma unroll
      for(j=0;j<inner.length();j++)
        chain1.XMADLH(acc[i+j+1], outer[i], inner[j], acc[i+j+1]);
      chain1.ADD(carry, zero, (i==0) ? zero : carry);
      chain1.end();

      PTXChain chain2(inner.length()+1, false, true);
      #pragma unroll
      for(j=0;j<inner.length();j++)
        chain2.XMADHL(acc[i+j+1], outer[i], inner[j], acc[i+j+1]);
      chain2.ADD(acc[i+j+1], acc[i+j+1], carry);
      chain2.end();

      if(i<outer.length()-1)
        inliner.ADDC(carry, zero, zero);
    }

    #pragma unroll
    for(i=outer.length()+inner.length()+1;i<acc.length();i++)
      inliner.ADDC_CC(acc[i], acc[i], zero);

    _roll(acc);

    #pragma unroll
    for(i=0;i<outer.length();i++) {
      PTXChain chain3(inner.length()+2);
      #pragma unroll
      for(j=0;j<inner.length();j++)
        chain3.XMADLL(acc[i+j], outer[i], inner[j], acc[i+j]);
      chain3.ADD(acc[i+j], acc[i+j], (i==0) ? zero : carry);
      chain3.ADD(carry, zero, zero);
      chain3.end();

      PTXChain chain4(inner.length(), false, true);
      #pragma unroll
      for(j=0;j<inner.length();j++)
        chain4.XMADHH(acc[i+j+1], outer[i], inner[j], acc[i+j+1]);
      chain4.end();

      if(i<outer.length()-1)
       inliner.ADDC(carry, carry, zero);
    }

    inliner.ADDC_CC(acc[outer.length()+inner.length()], acc[outer.length()+inner.length()], carry);
    #pragma unroll
    for(i=outer.length()+inner.length()+1;i<acc.length();i++)
      inliner.ADDC_CC(acc[i], acc[i], zero);
  }

  // FIX FIX FIX
  // This is using the IMAD algorithm.  Unfortunately, we can't switch to XMAD because a overlaps rr in Karatsuba.
  // Once Karatsuba is changed, we can change this algorithm.
  __device__ __forceinline__ void _mad(RegMP rr, RegMP a, RegMP b, RegMP c, bool carryIn) {
    int      i, j;
    uint32_t high, zero=0;

    if(rr.length()!=a.length()+b.length()) RMP_ERROR("_mad() - length mismatch");
    if(rr.overlap(b)) RMP_ERROR("_mad() - rr must not overlap b");
    if(rr.overlap(a) && !rr.upperAligned(a)) RMP_ERROR("_mad() - a must be upper aligned with rr");
    if(c.length()!=b.length()) RMP_ERROR("_mad() - length of b and c must match");

    // should use an inner/outer system

    PTXChain chain1(b.length()+1, carryIn, false);
    #pragma unroll
    for(int j=0;j<b.length();j++)
      chain1.MADLO(rr[j], a[0], b[j], c[j]);
    chain1.ADD(high, zero, zero);
    chain1.end();

    #pragma unroll
    for(i=0;i<a.length()-1;i++) {
      PTXChain chain2(b.length());
      #pragma unroll
      for(j=0;j<b.length()-1;j++)
        chain2.MADHI(rr[i+j+1], a[i], b[j], rr[i+j+1]);
      chain2.MADHI(rr[i+j+1], a[i], b[j], high);
      chain2.end();

      PTXChain chain3(b.length()+1);
      #pragma unroll
      for(int j=0;j<b.length();j++)
        chain3.MADLO(rr[i+j+1], a[i+1], b[j], rr[i+j+1]);
      chain3.ADD(high, zero, zero);
      chain3.end();
    }

    PTXChain chain4(b.length());
    #pragma unroll
    for(j=0;j<b.length()-1;j++)
      chain4.MADHI(rr[i+j+1], a[i], b[j], rr[i+j+1]);
    chain4.MADHI(rr[i+j+1], a[i], b[j], high);
    chain4.end();
  }

}
