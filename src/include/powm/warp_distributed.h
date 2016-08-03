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
  template<int _words>
  class Warp_Distributed {
    public:
      int32_t  _width;
      int32_t  _logWidth;
      uint32_t _windowBits;
      uint32_t _np0;
      uint32_t _registers[_words*4];
      WarpMP   _context;

      __device__ __forceinline__ Warp_Distributed(int32_t words, int width, int32_t bits, int32_t window_bits) : _context(width) {
        _width=width;
        _logWidth=__popc(width-1);
        _windowBits=window_bits;
      }

      __device__ __forceinline__ void initialize(uint32_t *window_data, int32_t mod_count) {
        PTXInliner c1Inliner(1, true), c2Inliner(2, true);
        uint32_t   x;

        c1Inliner.declarePredicate();
        c2Inliner.declarePredicate();

        // load n
        if(_words==1) {
          _registers[2]=window_data[(1<<_windowBits)*32];
          warp_transmit(x, _registers[2], 0, _width);
          _np0=computeNP0(x);
        }
        else {
          RegMP N(_registers, 0, 2*_words, _words);

          window_data=window_data + (1<<_windowBits)*32*_words;
          #pragma unroll
          for(int word=0;word<_words;word++)
            N[word]=window_data[word*32];

          warp_transmit(x, N[0], 0, _width);
          _np0=computeNP0(x);
        }
      }

      __device__ __forceinline__ void computeRModuloN(uint32_t *window_data) {
        int32_t groupThread=threadIdx.x & _width-1;

        if(_words==1)
          _registers[0]=~_registers[2] + (groupThread==0 ? 1 : 0);
        else {
          RegMP A(_registers, 0, 0, _words), N(_registers, 0, 2*_words, _words);

          A[0]=~N[0] + (groupThread==0 ? 1 : 0);
          #pragma unroll
          for(int word=1;word<_words;word++)
            A[word]=~N[word];
        }
      }

      __device__ __forceinline__ void storeResult(uint32_t *window, uint32_t *out_data, int32_t out_len, int32_t out_stride) {
        int32_t groupThread=threadIdx.x & _width-1;

        if(_words==1) {
          if(groupThread<out_len)
            out_data[groupThread * out_stride]=_registers[0];
        }
        else {
          RegMP A(_registers, 0, 0, _words);

          #pragma unroll
          for(int word=0;word<_words;word++) {
            if(groupThread*_words+word<out_len)
              out_data[groupThread*_words*out_stride + word*out_stride]=A[word];
          }
        }
      }

      __device__ __forceinline__ void printCurrent(uint32_t *context, const char *text) {
        RegMP    A(_registers, 0, 0, _words);
        int32_t  groupThread=threadIdx.x & _width-1;
        uint32_t x;

        if(groupThread==0) printf("%s=", text);
        for(int thread=_width-1;thread>=0;thread--) {
          #pragma unroll
          for(int word=_words-1;word>=0;word--) {
            warp_transmit(x, A[word], thread, _width);
            if(groupThread==0) printf("%08X", x);
          }
        }
        if(groupThread==0) printf("\n");
      }

      __device__ __forceinline__ void loadWindow(uint32_t *window_data, int32_t index) {
        if(_words==1)
          _registers[0]=window_data[index*32];
        else {
          RegMP A(_registers, 0, 0, _words);

          window_data=window_data + index*32*_words;
          #pragma unroll
          for(int word=0;word<_words;word++)
            A[word]=window_data[word*32];
        }
      }

      __device__ __forceinline__ void storeWindow(uint32_t *window_data, int index) {
        // if(blockIdx.x==0 && threadIdx.x<_width) {
        //   if(threadIdx.x==0) printf("W[%d]", index);
        //   printCurrent(NULL, "");
        // }

        if(_words==1)
          window_data[index*32]=_registers[0];
        else {
          RegMP A(_registers, 0, 0, _words);

          window_data=window_data + index*32*_words;
          #pragma unroll
          for(int word=0;word<_words;word++)
            window_data[word*32]=A[word];
        }
      }

      __device__ __forceinline__ uint32_t getBits(uint32_t *window_data, int32_t bitOffset, int32_t bitLength) {
        int32_t  wordOffset;
        uint32_t result;

        wordOffset=bitOffset>>5;
        bitOffset=bitOffset & 0x1F;

        window_data=window_data + (1<<_windowBits)*_words*32 + _words*32 - (threadIdx.x & _width-1);

        result=window_data[(wordOffset & _width-1) + (wordOffset>>_logWidth)*32]>>bitOffset;
        if(32-bitOffset<bitLength) {
          wordOffset++;
          result=result | (window_data[(wordOffset & _width-1) + (wordOffset>>_logWidth)*32]<<32-bitOffset);
        }

        return result & ((1<<bitLength)-1);
      }

      __device__ __forceinline__ void mulmod() {
        if(_words==1) {
          _context.mulmod<_words>(_registers[3], _registers[0], _registers[1], _registers[2], _np0);
          _registers[0]=_registers[3];
        }
        else {
          RegMP A(_registers, 0, 0, _words), B(_registers, 0, _words, _words), N(_registers, 0, 2*_words, _words), R(_registers, 0, 3*_words, _words);

          _context.mulmod<_words>(R, A, B, N, _np0);
          #pragma unroll
          for(int index=0;index<_words;index++)
            A[index]=R[index];
        }
      }

      __device__ __forceinline__ void reduceCurrent(uint32_t *window_data, int32_t mod_count) {
      }

      __device__ __forceinline__ void squareCurrent(uint32_t *window) {
        if(_words==1)
          _registers[1]=_registers[0];
        else {
          RegMP A(_registers, 0, 0, _words), B(_registers, 0, _words, _words);

          #pragma unroll
          for(int index=0;index<_words;index++)
            B[index]=A[index];
        }
        mulmod();
      }

      __device__ __forceinline__ void multiplyCurrentByWindow(uint32_t *window_data, int index) {
        if(_words==1)
          _registers[1]=window_data[index*32];
        else {
          RegMP B(_registers, 0, _words, _words);

          window_data=window_data + index*32*_words;
          #pragma unroll
          for(int word=0;word<_words;word++)
            B[word]=window_data[word*32];
        }
        mulmod();
      }

      __device__ __forceinline__ void multiplyCurrentByConstant(uint32_t *window_data, int index) {
      }

      __device__ __forceinline__ void multiplyCurrentByOne(uint32_t *window) {
        int groupThread=threadIdx.x & _width-1;

        if(_words==1)
          _registers[1]=groupThread==0 ? 1 : 0;
        else {
          RegMP B(_registers, 0, _words, _words);

          set_ui(B, (groupThread==0) ? 1 : 0);
        }
        mulmod();
      }
  };
}
