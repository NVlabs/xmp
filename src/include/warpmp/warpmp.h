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
  __device__ __forceinline__ void warp_transmit(uint32_t& res, uint32_t value, int32_t source, int32_t width) {
    asm volatile ("shfl.idx.b32 %0, %1, %2, %3;" : "=r"(res) : "r"(value), "r"(source), "r"((256-width)*256+31));
  }

  __device__ __forceinline__ void warp_shift_down(uint32_t& res, uint32_t value, int32_t width) {
    asm volatile ("shfl.down.b32 %0, %1, 1, %2;" : "=r"(res) : "r"(value), "r"((256-width)*256+31));
  }

  __device__ __forceinline__ void warp_shift_up(uint32_t& res, uint32_t value, int32_t width) {
    asm volatile ("shfl.up.b32 %0, %1, 1, %2;" : "=r"(res) : "r"(value), "r"((256-width)*256));
  }

  class WarpMP {
    public:
      int32_t  _width;
      uint32_t _lane;
      uint32_t _group0;
      uint32_t _carryPosition;

    __device__ __forceinline__ WarpMP(int width) {
      int32_t groupThread=threadIdx.x & width-1;

      _width=width;
      _lane=groupThread==0 ? 0 : (1<<(threadIdx.x & 0x1F));
      _group0=(groupThread==0) ? 1 : 0;
      _carryPosition=(threadIdx.x & ~(width-1)) + width & 0x1F;
    }

    template<int32_t _words>
    __device__ void mulmod(uint32_t &r, uint32_t& a, uint32_t& b, uint32_t& n, uint32_t np0);
    template<int32_t _words>
    __device__ void mulmod(RegMP r, RegMP a, RegMP b, RegMP n, uint32_t np0);
  };
}

#include "mulmod_one.h"

#ifdef IMAD
  #include "mulmod_imad.h"
#endif
#ifdef XMAD
  #include "mulmod_xmad.h"
#endif
