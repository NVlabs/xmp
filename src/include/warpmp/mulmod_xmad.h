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
  __device__ __forceinline__ void WarpMP::mulmod(RegMP ra, RegMP a, RegMP b, RegMP n, uint32_t np0) {
    PTXInliner inliner, c1Inliner(1, true), c2Inliner(2, true);
    int32_t    groupThread=threadIdx.x & _width-1;
    uint32_t   ru1, ru[_words], r1=0, r2=0, carry, x, q, source, p, g;
    uint32_t   zero=0, one=1, ones=0xFFFFFFFF, permuteShift16=0x5432;

    if(a.length()!=_words || b.length()!=_words || n.length()!=_words || ra.length()!=_words)
      RMP_ERROR("mulmod() - length mismatch");

    source=groupThread+1;

    #pragma unroll
    for(int index=0;index<_words;index++) {
      ra[index]=0;
      ru[index]=0;
    }
    r1=0;
    carry=0;
    #pragma nounroll
    for(int32_t thread=0;thread<_width;thread++) {
      #pragma unroll
      for(int word=0;word<_words;word++) {
        warp_transmit(x, b[word], thread, _width);

        PTXChain chain1(_words+1);
        #pragma unroll
        for(int index=0;index<_words;index++)
          chain1.XMADLH(ru[index], a[index], x, ru[index]);
        chain1.ADD(ru1, zero, zero);
        chain1.end();

        PTXChain chain2(_words+1);
        #pragma unroll
        for(int index=0;index<_words;index++)
          chain2.XMADHL(ru[index], a[index], x, ru[index]);
        chain2.ADD(ru1, ru1, zero);
        chain2.end();

        PTXChain chain4(_words+1);
        #pragma unroll
        for(int index=0;index<_words;index++)
          chain4.XMADLL(ra[index], a[index], x, ra[index]);
        chain4.ADD(r1, r1, zero);
        chain4.end();

        PTXChain chain5(_words+1);
        #pragma unroll
        for(int index=0;index<_words-1;index++)
          chain5.XMADHH(ra[index+1], a[index], x, ra[index+1]);
        chain5.XMADHH(r1, a[_words-1], x, r1);
        chain5.ADD(r2, zero, zero);
        chain5.end();

        // split ru[0] and add it into ra
        x=ru[0]<<16;
        inliner.ADD_CC(ra[0], ra[0], x);
        x=ru[0]>>16;
        inliner.ADDC(carry, carry, x);

        warp_transmit(x, ra[0], 0, _width);
        q=x*np0;

        // skip ru[0]
        PTXChain chain6(_words);
        #pragma unroll
        for(int index=1;index<_words;index++)
          chain6.XMADLH(ru[index], n[index], q, ru[index]);
        chain6.ADD(ru1, ru1, zero);
        chain6.end();

        // skip ru[0], shift
        PTXChain chain7(_words);
        #pragma unroll
        for(int index=1;index<_words;index++)
          chain7.XMADHL(ru[index-1], n[index], q, ru[index]);
        chain7.ADD(ru[_words-1], ru1, zero);
        chain7.end();

        // push the carry along
        inliner.ADD_CC(ra[1], ra[1], carry);
        inliner.ADDC(carry, zero, zero);

        // handles four XMADs for the q * n0 terms
        inliner.MADLO_CC(ra[0], n[0], q, ra[0]);
        inliner.MADHIC_CC(ra[1], n[0], q, ra[1]);
        inliner.ADDC(carry, carry, zero);

        warp_transmit(x, ra[0], source, _width);
        PTXChain chain8(_words+1);
        #pragma unroll
        for(int index=1;index<_words;index++)
          chain8.XMADLL(ra[index], n[index], q, ra[index]);
        chain8.ADD(r1, r1, x);
        chain8.ADD(r2, r2, zero);
        chain8.end();

        ra[0]=ra[1];
        PTXChain chain9(_words);
        #pragma unroll
        for(int index=1;index<_words-1;index++)
          chain9.XMADHH(ra[index], n[index], q, ra[index+1]);
        chain9.XMADHH(ra[_words-1], n[_words-1], q, r1);
        chain9.ADD(r1, r2, zero);
        chain9.end();
      }
    }

    // need to merge in carry
    PTXChain chain10(_words);
    #pragma unroll
    for(int index=1;index<_words;index++)
      chain10.ADD(ra[index], ra[index], (index==1) ? carry : zero);
    chain10.ADD(r1, r1, zero);
    chain10.end();

    #pragma unroll
    for(int index=_words-1;index>0;index--)
      inliner.PERMUTE(ru[index], ru[index-1], ru[index], permuteShift16);
    inliner.PERMUTE(ru[0], zero, ru[0], permuteShift16);

    PTXChain chain11(_words+1);
    for(int index=0;index<_words;index++)
      chain11.ADD(ra[index], ra[index], ru[index]);
    chain11.ADD(r1, r1, zero);
    chain11.end();

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
    PTXChain chain12(_words+1);
    #pragma unroll
    for(int index=0;index<_words;index++)
      chain12.ADD(ra[index], ra[index], index==0 ? x : zero);
    chain12.ADD(r1, r1, zero);
    chain12.end();

    x=ra[0];
    #pragma unroll
    for(int index=1;index<_words;index++)
      x=x & ra[index];

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
    PTXChain chain13(_words+1);
    chain13.ADD(p, p, ones);
    #pragma unroll
    for(int index=0;index<_words;index++)
      chain13.ADD(ra[index], ra[index], zero);
    chain13.end();

    // if(_forceBFE || _width<32)
    //   x=_negN0 & g;
    // else
    //   x=_negN0 & ~g;

    PTXChain chain14(_words+1);
    x=n[0] - _group0;
    x=~x & g;
    chain14.ADD(ra[0], ra[0], x);
    #pragma unroll
    for(int index=1;index<_words;index++) {
      x=~n[index] & g;
      chain14.ADD(ra[index], ra[index], x);
    }
    chain14.ADD(r1, zero, zero);
    chain14.end();

    x=ra[0];
    #pragma unroll
    for(int index=1;index<_words;index++)
      x=x & ra[index];

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
    PTXChain chain15(_words+1);
    chain15.ADD(x, x, ones);
    #pragma unroll
    for(int index=0;index<_words;index++)
      chain15.ADD(ra[index], ra[index], zero);
    chain15.end();
  }
}