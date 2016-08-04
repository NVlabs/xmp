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
#pragma once

#ifdef NDEBUG
#define PTX_ERROR(message) asm volatile("PTXInliner_ERRROR " message)
#else
#define PTX_ERROR(message)
#endif

namespace xmp {
  class PTXInliner {
    public:
      uint32_t _predicate;
      bool     _set;

      __device__ PTXInliner();
      __device__ PTXInliner(uint32_t predicate, bool set);

      __device__ PTXInliner pnot();
      __device__ void       nestStart();
      __device__ void       nestEnd();
      __device__ void       declarePredicate();

      // These are just covers over asm inlines.  The main reason to use them is that they make
      // the math code more concise and easier to read.
      //
      // Usage example:
      //    uint32_t   a[3], b[3], r[3];
      //    PTXInliner inliner;
      //
      //    inliner.ADD_CC(r[0], a[0], b[0]);
      //    inliner.ADDC_CC(r[1], a[1], b[1]);
      //    inliner.ADDC(r[2], a[2], b[2]);
      //
      // Will generate:
      //    add.cc.u32    r0, a0, b0;
      //    addc.cc.u32   r1, a1, b1;
      //    addc.u32      r2, a2, b2;


      // math ops and carry ops
      __device__ void ADD(uint32_t& r, uint32_t& a, uint32_t& b);
      __device__ void ADD_CC(uint32_t& r, uint32_t& a, uint32_t& b);
      __device__ void ADDC(uint32_t& r, uint32_t& a, uint32_t& b);
      __device__ void ADDC_CC(uint32_t& r, uint32_t& a, uint32_t& b);

      __device__ void SUB(uint32_t& r, uint32_t& a, uint32_t& b);
      __device__ void SUB_CC(uint32_t& r, uint32_t& a, uint32_t& b);
      __device__ void SUBC(uint32_t& r, uint32_t& a, uint32_t& b);
      __device__ void SUBC_CC(uint32_t& r, uint32_t& a, uint32_t& b);

      __device__ void MULLO(uint32_t& r, uint32_t& a, uint32_t& b);
      __device__ void MULHI(uint32_t& r, uint32_t& a, uint32_t& b);

      __device__ void MADWIDE(uint64_t& r, uint32_t& a, uint32_t& b, uint64_t& c);

      __device__ void MADLO(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void MADLO_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void MADLOC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void MADLOC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);

      __device__ void MADHI(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void MADHI_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void MADHIC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void MADHIC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);

      __device__ void XMADLL(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADLL_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADLLC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADLLC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);

      __device__ void XMADLH(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADLH_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADLHC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADLHC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);

      __device__ void XMADHL(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADHL_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADHLC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADHLC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);

      __device__ void XMADHH(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADHH_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADHHC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADHHC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);

      // setp operations
      __device__ void SETP_EQ(uint32_t& a, uint32_t& b);
      __device__ void SETP_NE(uint32_t& a, uint32_t& b);
      __device__ void SELP(uint32_t& r, uint32_t& a, uint32_t& b);

      // misc inlines
      __device__ void PERMUTE(uint32_t& r, uint32_t& a, uint32_t& b, uint32_t& c);
      __device__ void BFE(uint32_t& r, uint32_t& a, uint32_t& start, uint32_t& len);
      __device__ void BFE_S(uint32_t& r, uint32_t& a, uint32_t& start, uint32_t& len);
      __device__ void SHF_L_WRAP(uint32_t& r, uint32_t& a, uint32_t& b, uint32_t& c);
      __device__ void SHF_R_WRAP(uint32_t& r, uint32_t& a, uint32_t& b, uint32_t& c);
  };

  class PTXChain {
    public:
      uint32_t _predicate;
      bool     _set;
      int32_t  _size;
      int32_t  _position;
      bool     _carryIn;
      bool     _carryOut;

      __device__ PTXChain();
      __device__ PTXChain(int32_t size);
      __device__ PTXChain(int32_t size, bool carryIn, bool carryOut);

      __device__ PTXChain(PTXInliner inliner);
      __device__ PTXChain(PTXInliner inliner, int32_t size);
      __device__ PTXChain(PTXInliner inliner, int32_t size, bool carryIn, bool carryOut);

      __device__ void start(int32_t size);
      __device__ void start(int32_t size, bool carryIn, bool carryOut);
      __device__ void end();

      // These routines just call the right PTXInliner routine depending on the place in the carry chain.
      //
      // usage:
      //    uint32_t a[3], b[3], r[3];
      //    PTXChain chain;
      //
      //    chain.start(3);
      //    chain.ADD(r[0], a[0], b[0]);
      //    chain.ADD(r[1], a[1], b[1]);
      //    chain.ADD(r[2], a[2], b[2]);
      //    chain.end();
      //
      // Will generate:
      //    add.cc.u32    r0, a0, b0;
      //    addc.cc.u32   r1, a1, b1;
      //    addc.u32      r2, a2, b2;
      //
      // These routines are usually used with #pragma unroll loops and lead to cleaner looking code.

      __device__ void ADD(uint32_t& r, uint32_t& a, uint32_t& b);
      __device__ void SUB(uint32_t& r, uint32_t& a, uint32_t& b);
      __device__ void MADLO(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void MADHI(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADLL(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADLH(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADHL(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
      __device__ void XMADHH(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c);
  };

  // Compile with -DHACKSAW to insert __syncthreads() around the various XMAD operators.  The
  // syncthreads make it possible for the SASS hacking tool to detect and replace the PTX XMAD
  // equivalent with XMAD instructions

  __device__ __forceinline__ void HACKSAW_SYNC() {
  #ifdef HACKSAW
    __syncthreads();
  #endif
  }
}

#include "PTXInliner_impl.h"
#include "PTXChain_impl.h"


