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

#if defined(__CUDACC__)
#ifdef DEVELOPMENT
#define RMP_ERROR(text) asm volatile ("REGMP_ERROR " text);
#else
#define RMP_ERROR(text)
#endif
#else
#include <stdlib.h>
#define RMP_ERROR(text) { fprintf(stderr, "REGMP_ERROR - %s\n", text); exit(1); }
#endif

#define B0T0 (blockIdx.x==0 && threadIdx.x==0)

#define RMP_NO_OVERLAP 0
#define RMP_LOWER_ALIGNED 1
#define RMP_UPPER_ALIGNED 2
#define RMP_UNALIGNED 3

#define RMP_LD_GLOBAL       100
#define RMP_LD_GLOBAL_V2    101
#define RMP_LD_GLOBAL_V4    102
#define RMP_ST_GLOBAL       103
#define RMP_ST_GLOBAL_V2    104
#define RMP_ST_GLOBAL_V4    105

#if defined(IMAD) && defined(XMAD)
  #pragma error Both IMAD and XMAD are defined
#endif

#if !defined(IMAD) && !defined(XMAD)
   #if __CUDA_ARCH__<500
      #define IMAD
   #else
      #define XMAD
   #endif
#endif

namespace xmp {
  typedef struct {
    uint32_t *registers;
    uint32_t tag;
    uint32_t start, length;
  } Registers;

  class RegMP {
    public:
      Registers _lower;
      Registers _upper;

      // constructor/initializers
      __device__           RegMP();
      __device__           RegMP(uint32_t *registers, const uint32_t tag, const uint32_t count);
      __device__           RegMP(uint32_t *registers, const uint32_t tag, const uint32_t start, const uint32_t length);
      __device__           RegMP(RegMP mp, const int32_t count);    // length>0 for lower(), length<0 for upper()
      __device__ RegMP     concatenate(RegMP upper);                // concatenate
      __device__ RegMP     lower(const uint32_t length);
      __device__ RegMP     upper(const uint32_t length);

      // overlap and alignment
      __device__ bool      overlap(RegMP check);
      __device__ bool      lowerAligned(RegMP check);
      __device__ bool      upperAligned(RegMP check);
      __device__ int       alignment(RegMP check);

      // access
      __device__ uint32_t  length();
      __device__ uint32_t& operator[] (const int index);
      __device__ uint32_t* registers();

      // debugging
      __device__ void      print();
      __device__ void      print(const char *text);
  };

  // ASM instruction generation
  __device__ void      instruction_store(int instruction, uint32_t *address, uint32_t& a);
  __device__ void      instruction_store(int instruction, uint32_t *address, uint32_t& a, uint32_t& b);
  __device__ void      instruction_store(int instruction, uint32_t *address, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d);

  // RegMP operators
  __device__ void      set_ui(RegMP r, uint32_t ui);
  __device__ void      set_si(RegMP r, int32_t si);

  __device__ void      bitwise_not(RegMP r, RegMP a);
  __device__ void      bitwise_and(RegMP r, RegMP a, RegMP b);
  __device__ void      bitwise_and(RegMP r, RegMP a, uint32_t b);
  __device__ void      bitwise_or(RegMP r, RegMP a, RegMP b);
  __device__ void      bitwise_or(RegMP r, RegMP a, uint32_t b);
  __device__ void      bitwise_xor(RegMP r, RegMP a, RegMP b);
  __device__ void      bitwise_xor(RegMP r, RegMP a, uint32_t b);

  __device__ void      add(RegMP r, RegMP a, RegMP b);
  __device__ void      addc(RegMP r, RegMP a, RegMP b, bool carry);
  __device__ bool      add_cc(RegMP r, RegMP a, RegMP b);
  __device__ bool      addc_cc(RegMP r, RegMP a, RegMP b, bool carry);

  __device__ void      sub(RegMP r, RegMP a, RegMP b);
  __device__ void      subc(RegMP r, RegMP a, RegMP b, bool carry);
  __device__ bool      sub_cc(RegMP r, RegMP a, RegMP b);
  __device__ bool      subc_cc(RegMP r, RegMP a, RegMP b, bool carry);

  __device__ void      mul(RegMP rr, RegMP a, RegMP b);
  __device__ void      sqr(RegMP rr, RegMP a);
#ifdef IMAD
  __device__ void      reduce(RegMP r, RegMP xx, RegMP n, uint32_t np0);
#endif
#ifdef XMAD
  __device__ void      reduce(RegMP r, RegMP xx, RegMP n, RegMP temp, uint32_t np0);
#endif

  // Storage APIs
  __device__ void      load_contiguous_direct(RegMP x, void *base);
  __device__ void      store_contiguous_direct(RegMP x, void *base);

  __device__ void      load_strided(RegMP x, void *base);
  __device__ void      store_strided(RegMP x, void *base);

  // Misc math
  __device__ uint32_t  computeNP0(uint32_t n0);

  // Debugging
  __device__ void      print(const char *text, RegMP x);
}

#include "basics.h"
#include "assignment.h"
#include "logic.h"
#include "add.h"
#include "sub.h"
#include "shift.h"

#include "clz.h"
#include "zeros_ones.h"


#ifdef IMAD
  #include "mul_imad.h"
  #include "sqr_imad.h"
  #include "reduce_imad.h"
#endif

#ifdef XMAD
  #include "rockroll.h"
  #include "mul_xmad.h"
  #include "sqr_xmad.h"
  #include "reduce_xmad.h"
#endif

#include "div.h"
#include "kar.h"
#include "np0.h"
