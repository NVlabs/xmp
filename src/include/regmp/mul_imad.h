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
    int        i, j;
    uint32_t   high, zero=0;

    if(rr.length()!=a.length()+b.length()) RMP_ERROR("mul() - length mismatch");
    if(rr.overlap(a) && rr.overlap(b)) RMP_ERROR("mul() - r overlaps both a and b");

    // lay the data out correctly.  A or B can overlap R, but not both.  If the overlap
    // is at the high everything works just great.  If the overlap is at the low end of R,
    // it must be copied to the high end of R.  It's not possible to avoid the copy.

    if(!rr.overlap(a) && !rr.overlap(b)) {
      if(a.length()>=b.length()) {
        inner=a;
        outer=b;
      }
      else {
        inner=b;
        outer=a;
      }
    }
    else if(rr.overlap(a)) {
      if(rr.lowerAligned(a)) {
        #pragma unroll
        for(int index=a.length()-1;index>=0;index--)
          rr[index+b.length()]=a[index];
        inner=b;
        outer=rr.upper(a.length());
      }
      else {
        inner=b;
        outer=a;
      }
    }
    else if(rr.overlap(b)) {
      if(rr.lowerAligned(b)) {
        #pragma unroll
        for(int index=b.length()-1;index>=0;index--)
          rr[index+a.length()]=b[index];
        inner=a;
        outer=rr.upper(b.length());
      }
      else {
        inner=a;
        outer=b;
      }
    }

    #pragma unroll
    for(j=0;j<inner.length();j++)
      inliner.MULLO(rr[j], outer[0], inner[j]);

    #pragma unroll
    for(i=0;i<outer.length()-1;i++) {
      PTXChain chain1(inner.length());
      #pragma unroll
      for(j=0;j<inner.length()-1;j++)
        chain1.MADHI(rr[i+j+1], outer[i], inner[j], rr[i+j+1]);
      chain1.MADHI(rr[i+j+1], outer[i], inner[j], (i==0) ? zero : high);
      chain1.end();

      PTXChain chain2(inner.length()+1);
      #pragma unroll
      for(int j=0;j<inner.length();j++)
        chain2.MADLO(rr[i+j+1], outer[i+1], inner[j], rr[i+j+1]);
      chain2.ADD(high, zero, zero);
      chain2.end();
    }

    PTXChain chain3(inner.length());
    #pragma unroll
    for(j=0;j<inner.length()-1;j++)
      chain3.MADHI(rr[i+j+1], outer[i], inner[j], rr[i+j+1]);
    chain3.MADHI(rr[i+j+1], outer[i], inner[j], (outer.length()==1) ? zero : high);
    chain3.end();
  }

  __device__ __forceinline__ void accumulate_mul(RegMP acc, RegMP a, RegMP b) {
    PTXInliner inliner;
    RegMP      inner, outer;
    int        i, j;
    uint32_t   high, zero=0;

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

    #pragma unroll
    for(i=0;i<outer.length();i++) {
      PTXChain chain1(inner.length()+2);
      #pragma unroll
      for(j=0;j<inner.length();j++)
        chain1.MADLO(acc[i+j], outer[i], inner[j], acc[i+j]);
      chain1.ADD(acc[i+j], acc[i+j], (i==0) ? zero : high);
      chain1.ADD(high, zero, zero);
      chain1.end();

      PTXChain chain2(inner.length(), false, true);
      #pragma unroll
      for(j=0;j<inner.length();j++)
        chain2.MADHI(acc[i+j+1], outer[i], inner[j], acc[i+j+1]);
      chain2.end();
      if(i<outer.length()-1)
        inliner.ADDC(high, high, zero);
    }

    #pragma unroll
    for(j=outer.length()+inner.length();j<acc.length();j++)
      inliner.ADDC_CC(acc[j], acc[j], (j==outer.length()+inner.length()) ? high : zero);
  }

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
