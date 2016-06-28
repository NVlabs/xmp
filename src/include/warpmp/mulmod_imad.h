/***
Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

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
  template<int32_t _words>
  __device__ __forceinline__ void WarpMP::mulmod(RegMP r, RegMP a, RegMP b, RegMP n, uint32_t np0) {
    PTXInliner inliner, c1Inliner(1, true), c2Inliner(2, true);
    int32_t    groupThread=threadIdx.x & _width-1;
    uint32_t   r1=0, r2=0, q, x, source, p, g;
    uint32_t   zero=0, one=1, ones=0xFFFFFFFF;

    if(a.length()!=_words || b.length()!=_words || n.length()!=_words || r.length()!=_words)
      RMP_ERROR("mulmod() - length mismatch");

    source=groupThread+1;

    #pragma unroll
    for(int index=0;index<_words;index++)
      r[index]=0;
    r1=0;
    #pragma nounroll
    for(int32_t thread=0;thread<_width;thread++) {
      #pragma unroll
      for(int word=0;word<_words;word++) {
        warp_transmit(x, b[word], thread, _width);

        PTXChain chain1(_words+1);
        #pragma unroll
        for(int index=0;index<_words;index++)
          chain1.MADLO(r[index], a[index], x, r[index]);
        chain1.ADD(r1, r1, zero);
        chain1.end();

        PTXChain chain2(_words+1);
        #pragma unroll
        for(int index=0;index<_words-1;index++)
          chain2.MADHI(r[index+1], a[index], x, r[index+1]);
        chain2.MADHI(r1, a[_words-1], x, r1);
        chain2.ADD(r2, zero, zero);
        chain2.end();

        warp_transmit(x, r[0], 0, _width);
        q=x*np0;
        PTXChain chain3(_words+2);
        #pragma unroll
        for(int index=0;index<_words;index++)
          chain3.MADLO(r[index], n[index], q, r[index]);
        warp_transmit(x, r[0], source, _width);
        chain3.ADD(r1, r1, x);
        chain3.ADD(r2, r2, zero);
        chain3.end();

        PTXChain chain4(_words+1);
        #pragma unroll
        for(int index=0;index<_words-1;index++)
          chain4.MADHI(r[index], n[index], q, r[index+1]);
        chain4.MADHI(r[_words-1], n[_words-1], q, r1);
        chain4.ADD(r1, r2, zero);
        chain4.end();
      }
    }

    // Very important: carry resolution must run in constant time.
    // So we have to use a ballot approach and not a carry ripple approach

    // r2:r1:r <= 0x00000002 0xFFFFFFFD
    asm volatile ("shfl.up.b32 %0|%%c1, %1, 1, %2;" : "=r"(x) : "r"(r1), "r"((256-_width)*256));

    // all but most significant thread(s) clear r1
    if(groupThread!=_width-1)
      r1=0;

    // this would be slightly faster if predication worked
    //   c1Inliner.ADD_CC(r, r, x);
    //   c1Inliner.ADDC(r1, r1, zero);
    c1Inliner.SELP(x, x, zero);
    PTXChain chain5(_words+1);
    #pragma unroll
    for(int index=0;index<_words;index++)
      chain5.ADD(r[index], r[index], index==0 ? x : zero);
    chain5.ADD(r1, r1, zero);
    chain5.end();

    x=r[0];
    #pragma unroll
    for(int index=1;index<_words;index++)
      x=x & r[index];

    // p=__ballot(r==0xFFFFFFFF);
    // g=__ballot(r1==1);
    // ensure that p for low word is always clear
    asm volatile ("setp.eq.and.u32 %%c2,%1,0xFFFFFFFF,%%c1;\n\t"
                  "vote.ballot.b32 %0,%%c2;" : "=r"(p) : "r"(x));
    asm volatile ("setp.eq.u32 %%c2,%1,1;\n\t"
                  "vote.ballot.b32 %0,%%c2;" : "=r"(g) : "r"(r1));

    // if(_forceBFE || _width<32) {
    //   inliner.ADD_CC(x, g, g);
    //   inliner.ADDC_CC(x, x, p);
    //   inliner.ADDC(x, x, zero);
    //   inliner.BFE_S(g, x, _carryPosition, one);
    // }
    // else {
    //   inliner.ADD_CC(x, g, g);
    //   inliner.ADDC(g, zero, zero);
    //   inliner.ADD_CC(x, x, p);
    //   inliner.ADDC_CC(g, g, ones);   // sets the carry bit if we need to subtract
    // }
    inliner.ADD_CC(x, g, g);
    inliner.ADDC_CC(x, x, p);
    inliner.ADDC(x, x, zero);
    inliner.BFE_S(g, x, _carryPosition, one);

    p=(x^p) & _lane;
    PTXChain chain6(_words+1);
    chain6.ADD(p, p, ones);
    #pragma unroll
    for(int index=0;index<_words;index++)
      chain6.ADD(r[index], r[index], zero);
    chain6.end();

    // if(_forceBFE || _width<32)
    //   x=_negN0 & g;
    // else
    //   x=_negN0 & ~g;

    PTXChain chain7(_words+1);
    x=n[0] - _group0;
    x=~x & g;
    chain7.ADD(r[0], r[0], x);
    #pragma unroll
    for(int index=1;index<_words;index++) {
      x=~n[index] & g;
      chain7.ADD(r[index], r[index], x);
    }
    chain7.ADD(r1, zero, zero);
    chain7.end();

    x=r[0];
    #pragma unroll
    for(int index=1;index<_words;index++)
      x=x & r[index];

    // p=__ballot(r==0xFFFFFFFF);
    // g=__ballot(r1==1);
    // ensure that p for low word is always clear
    asm volatile ("setp.eq.and.u32 %%c2,%1,0xFFFFFFFF,%%c1;\n\t"
                  "vote.ballot.b32 %0,%%c2;" : "=r"(p) : "r"(x));
    asm volatile ("setp.eq.u32 %%c2,%1,1;\n\t"
                  "vote.ballot.b32 %0,%%c2;" : "=r"(g) : "r"(r1));

    x=p+g+g;
    x=(x^p) & _lane;

    // all but the least significant thread(s)
    // c1Inliner.ADD_CC(x, x, ones);
    // c1Inliner.ADDC(r, r, zero);
    PTXChain chain8(_words+1);
    chain8.ADD(x, x, ones);
    #pragma unroll
    for(int index=0;index<_words;index++)
      chain8.ADD(r[index], r[index], zero);
    chain8.end();
  }
}