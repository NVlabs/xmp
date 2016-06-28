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
  __device__ __forceinline__ void WarpMP::mulmod(uint32_t& r, uint32_t& a, uint32_t& b, uint32_t& n, uint32_t np0) {
    PTXInliner inliner, c1Inliner(1, true), c2Inliner(2, true);
    int32_t    groupThread=threadIdx.x & _width-1;
    uint32_t   r1=0, r2=0, q, x, source, p, g;
    uint32_t   zero=0, one=1, ones=0xFFFFFFFF;

    source=groupThread+1;

    r=0;
    #pragma nounroll
    for(int32_t word=0;word<_width;word++) {
      // broadcast b[i]
      warp_transmit(x, b, word, _width);

      inliner.MADLO_CC(r, a, x, r);
      inliner.MADHIC_CC(r1, a, x, r1);
      inliner.ADDC(r2, zero, zero);

      // broadcast r[0]
      warp_transmit(x, r, 0, _width);
      q=x*np0;

      inliner.MADLO_CC(r, n, q, r);
      inliner.MADHIC_CC(r1, n, q, r1);
      inliner.ADDC(r2, r2, zero);

      // shift right by 32 bits
      warp_transmit(x, r, source, _width);
      inliner.ADD_CC(r, r1, x);
      inliner.ADDC(r1, r2, zero);
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
    inliner.ADD_CC(r, r, x);
    inliner.ADDC(r1, r1, zero);

    // p=__ballot(r==0xFFFFFFFF);
    // g=__ballot(r1==1);
    // ensure that p for low word is always clear
    asm volatile ("setp.eq.and.u32 %%c2,%1,0xFFFFFFFF,%%c1;\n\t"
                  "vote.ballot.b32 %0,%%c2;" : "=r"(p) : "r"(r));
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
    inliner.ADD_CC(p, p, ones);
    inliner.ADDC(r, r, zero);

    // if(_forceBFE || _width<32)
    //   x=_negN0 & g;
    // else
    //   x=_negN0 & ~g;

    x=n - _group0;   // n must be odd, so there is no chance for a carry ripple
    x=~x & g;

    inliner.ADD_CC(r, r, x);
    inliner.ADDC(r1, zero, zero);

    // p=__ballot(r==0xFFFFFFFF);
    // g=__ballot(r1==1);
    // ensure that p for low word is always clear
    asm volatile ("setp.eq.and.u32 %%c2,%1,0xFFFFFFFF,%%c1;\n\t"
                  "vote.ballot.b32 %0,%%c2;" : "=r"(p) : "r"(r));
    asm volatile ("setp.eq.u32 %%c2,%1,1;\n\t"
                  "vote.ballot.b32 %0,%%c2;" : "=r"(g) : "r"(r1));

    x=p+g+g;
    x=(x^p) & _lane;

    // all but the least significant thread(s)
    // c1Inliner.ADD_CC(x, x, ones);
    // c1Inliner.ADDC(r, r, zero);
    inliner.ADD_CC(x, x, ones);
    inliner.ADDC(r, r, zero);
  }
}


