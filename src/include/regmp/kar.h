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
   template <int level>
   __device__ __forceinline__ void kar_mul(RegMP rr, RegMP a, RegMP b, RegMP t) {
     PTXInliner inliner;
     int        length=a.length(), half=length/2;
     RegMP      al=a.lower(half), ah=a.upper(half);
     RegMP      bl=b.lower(half), bh=b.upper(half);
     RegMP      rl=rr.lower(length), rh=rr.upper(length);
     RegMP      rll=rl.lower(half), rlh=rl.upper(half);
     RegMP      rhl=rh.lower(half), rhh=rh.upper(half);
     RegMP      tl=t.lower(half), th=t.upper(half);
     RegMP      pl=rll.concatenate(rhl), ph=rlh.concatenate(rhh), pskar=ah.concatenate(th);
     uint32_t   carryA, carryB, zero=0;

     carryA=add_cc(tl, al, ah);
     carryB=add_cc(th, bl, bh);

     kar_mul<level-1>(ph, ah, bh, rll);
     kar_mul<level-1>(pl, al, bl, ah);

     bitwise_and(al, tl, -carryB);
     bitwise_and(ah, th, -carryA);

     carryA=carryA & carryB;

     _add(al, al, ah, false, true);
     inliner.ADDC(carryA, carryA, zero);

     _sub(rlh, rhl, rlh, false, true);
     inliner.SUBC(carryB, zero, zero);

     _sub(rhl, al, rlh, true, true);
     inliner.SUBC(carryA, carryA, carryB);

     _sub(rlh, rlh, rll, false, true);
     _sub(rhl, rhl, rhh, true, true);
     inliner.SUBC(carryA, carryA, zero);

     kar_mul<level-1>(pskar, tl, th, al);
     _add(rlh, rlh, ah, false, true);
     _add(rhl, rhl, th, true, true);
     _add_si(rhh, rhh, carryA, true, false);
   }

   template <>
   __device__ __forceinline__ void kar_mul<0>(RegMP rr, RegMP a, RegMP b, RegMP t) {
     mul(rr, a, b);
   }

   template <>
   __device__ __forceinline__ void kar_mul<1>(RegMP rr, RegMP a, RegMP b, RegMP t) {
     PTXInliner inliner;
     int        length=a.length(), half=length/2;
     RegMP      al=a.lower(half), ah=a.upper(half);
     RegMP      bl=b.lower(half), bh=b.upper(half);
     RegMP      rl=rr.lower(length), rh=rr.upper(length);
     RegMP      rll=rl.lower(half), rlh=rl.upper(half);
     RegMP      rhl=rh.lower(half), rhh=rh.upper(half);
     RegMP      tl=t.lower(half);
     RegMP      pl=rll.concatenate(rhl), ph=rlh.concatenate(rhh), psgs=rlh.concatenate(al);
     uint32_t   carryA, carryB, carryC, zero=0, minusOne=0xFFFFFFFF;

     carryB=add_cc(tl, bl, bh);
     mul(ph, ah, bh);

     carryA=add_cc(ah, al, ah);
     mul(pl, al, bl);

     _sub(rlh, rhl, rlh, false, true);
     inliner.ADDC(carryC, zero, zero);

     bitwise_and(rhl, tl, -carryA);
     bitwise_and(al, ah, -carryB);

     carryA=carryA & carryB;
     _add(rhl, rhl, al, false, true);
     inliner.ADDC(carryA, carryA, zero);

     inliner.ADD_CC(carryC, carryC, minusOne);
     _sub(rhl, rhl, rlh, true, true);
     inliner.SUBC(carryA, carryA, carryC);

     _sub(rlh, rlh, rll, false, true);
     _sub(rhl, rhl, rhh, true, true);
     inliner.SUBC(carryA, carryA, zero);

     _mad(psgs, ah, tl, rlh, false);
     _add(rhl, rhl, al, false, true);
     _add_si(rhh, rhh, carryA, true, false);
   }


/*
   // this is an old version that uses more registers than the version above

   template <>
   __device__ __forceinline__ void kar_mul<1>(RegMP rr, RegMP a, RegMP b, RegMP t) {
     PTXInliner inliner;
     int        length=a.length(), half=length/2;
     RegMP      al=a.lower(half), ah=a.upper(half);
     RegMP      bl=b.lower(half), bh=b.upper(half);
     RegMP      rl=rr.lower(length), rh=rr.upper(length);
     RegMP      rll=rl.lower(half), rlh=rl.upper(half);
     RegMP      rhl=rh.lower(half), rhh=rh.upper(half);
     RegMP      tl=t.lower(half), th=t.upper(half);
     RegMP      pl=rll.concatenate(rhl), ph=rlh.concatenate(rhh), psgs=rlh.concatenate(th);
     uint32_t   carryA, carryB, zero=0;

     carryA=add_cc(tl, al, ah);
     carryB=add_cc(th, bl, bh);

     mul(ph, ah, bh);
     mul(pl, al, bl);

     bitwise_and(al, tl, -carryB);
     bitwise_and(ah, th, -carryA);

     carryA=carryA & carryB;

     _add(al, al, ah, false, true);
     inliner.ADDC(carryA, carryA, zero);

     _sub(rlh, rhl, rlh, false, true);
     inliner.SUBC(carryB, zero, zero);

     _sub(rhl, al, rlh, true, true);
     inliner.SUBC(carryA, carryA, carryB);

     _sub(rlh, rlh, rll, false, true);
     _sub(rhl, rhl, rhh, true, true);
     inliner.SUBC(carryA, carryA, zero);

     _mad(psgs, th, tl, rlh, false);
     _add(rhl, rhl, th, false, true);
     _add_si(rhh, rhh, carryA, true, false);
   }
*/


   template <int level>
   __device__ __forceinline__ void kar_sqr(RegMP rr, RegMP a, RegMP t) {
     PTXInliner inliner;
     int        length=a.length(), half=length/2;
     RegMP      al=a.lower(half), ah=a.upper(half);
     RegMP      rl=rr.lower(length), rh=rr.upper(length);
     RegMP      rll=rl.lower(half), rlh=rl.upper(half);
     RegMP      rhl=rh.lower(half), rhh=rh.upper(half);
     RegMP      tl=t.lower(half);
     uint32_t   carryA, carryB, zero=0;

     _sub(tl, al, ah, false, true);
     inliner.SUBC(carryB, zero, zero);

     PTXChain chain1(half);
     #pragma unroll
     for(int index=0;index<half;index++) {
       tl[index]=tl[index] ^ carryB;
       chain1.SUB(tl[index], tl[index], carryB);
     }
     chain1.end();

     kar_sqr<level-1>(rl, al, rhl);
     kar_sqr<level-1>(rh, ah, al);

     _add(rhl, rhl, rlh, false, true);
     inliner.ADDC(carryB, zero, zero);

     kar_sqr<level-1>(a, tl, rlh);

     _sub(rlh, rhl, al, false, true);
     _sub(rhl, rhl, ah, true, true);
     inliner.SUBC(carryA, carryB, zero);

     _add_ui(rhl, rhl, carryB, false, true);       // handle the dangling carry - no way around this
     inliner.ADDC(carryA, carryA, zero);

     _add(rlh, rlh, rll, false, true);
     _add(rhl, rhl, rhh, true, true);
     _add_si(rhh, rhh, carryA, true, false);
   }

   template <>
   __device__ __forceinline__ void kar_sqr<0>(RegMP rr, RegMP a, RegMP t) {
     sqr(rr, a);
   }
}

