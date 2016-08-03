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
  template <int _words, int _ks, int _km>
  class ThreeN_n {
    public:
      uint32_t _np0;
      uint32_t _registers[_words*4];
      int32_t  _window_size;
      int32_t  _exp_offset;
      int32_t  _exp_length;

      __device__ __forceinline__ ThreeN_n(int32_t words, int32_t width, int32_t bits, int32_t window_bits) {
        _window_size=1<<window_bits;
//        _exp_offset=(_window_size*_words + _words)*32;
        _exp_length=(bits+31)>>5;
      }

      __device__ __forceinline__ void initialize(uint32_t *window_data, int32_t mod_count) {
        _np0=computeNP0(window_data[0]);
      }

      __device__ __forceinline__ void computeRModuloN(uint32_t *window_data) {
        RegMP    A(_registers, 0, 0, _words);
        uint32_t zero=0;

        #pragma unroll
        for(int word=0;word<_words;word++)
          A[word]=window_data[32*word];

        PTXChain chain(A.length());
        #pragma unroll
        for(int word=0;word<_words;word++)
          chain.SUB(A[word], zero, A[word]);
        chain.end();
      }

/*
      __device__ __forceinline__ void loadArgument(uint32_t *a_data, int32_t a_len, int32_t a_stride, int32_t a_count) {
        RegMP   A(_registers, 0, 0, _words);
        int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;

        a_data=a_data+thread%a_count;
        #pragma unroll
        for(int word=0;word<_words;word++)
          if(word<a_len)
            A[word]=a_data[word*a_stride];
          else
            A[word]=0;
      }
*/

      __device__ __forceinline__ void storeResult(uint32_t *window, uint32_t *out_data, int32_t out_len, int32_t out_stride) {
        RegMP   A(_registers, 0, 0, _words);

        #pragma unroll
        for(int word=0;word<_words;word++)
          if(word<out_len)
            out_data[word*out_stride]=A[word];
      }

      __device__ __forceinline__ void printCurrent(uint32_t *context, const char *text) {
        RegMP A(_registers, 0, 0, _words);

        A.print(text);
      }

      __device__ __forceinline__ void loadWindow(uint32_t *window_data, int32_t index) {
        RegMP A(_registers, 0, 0, _words);

        #pragma unroll
        for(int word=0;word<_words;word++)
          A[word]=window_data[index*_words*32 + word*32 + _words*32];
      }

      __device__ __forceinline__ void storeWindow(uint32_t *window_data, int index) {
        RegMP A(_registers, 0, 0, _words);

        #pragma unroll
        for(int word=0;word<_words;word++)
          window_data[index*_words*32 + word*32 + _words*32]=A[word];
      }

      __device__ __forceinline__ uint32_t getBits(uint32_t *window_data, int32_t bitOffset, int32_t bitLength) {
        int32_t  wordOffset;
        uint32_t result;

        wordOffset=bitOffset>>5;
        bitOffset=bitOffset & 0x1F;

        if(wordOffset<_exp_length)
          result=window_data[(_window_size*_words + _words + wordOffset)*32]>>bitOffset;
        else
          return 0;

        if(32-bitOffset<bitLength)
          if(wordOffset+1<_exp_length)
            result=result | (window_data[(_window_size*_words + _words + wordOffset + 1)*32]<<32-bitOffset);

        return result & ((1<<bitLength)-1);
      }

      __device__ __forceinline__ void reduceCurrent(uint32_t *window_data, int32_t mod_count) {
        RegMP    A(_registers, 0, 0, _words), RR(_registers, 0, _words, _words*2), N(_registers, 0, 0, _words), T(_registers, 0, _words*3, _words);

        #pragma unroll
        for(int word=0;word<_words;word++)
          N[word]=window_data[32*word];

#ifdef XMAD
        reduce(A, RR, N, T, _np0);
#else
        reduce(A, RR, N, _np0);
#endif
      }

      __device__ __forceinline__ void squareCurrent(uint32_t *window) {
        RegMP    A(_registers, 0, 0, _words), RR(_registers, 0, _words, _words*2), T(_registers, 0, _words*3, _words/2);

#ifdef XMAD
        sqr(RR, A);
#else
        kar_sqr<_ks>(RR, A, T);
#endif
      }

      __device__ __forceinline__ void multiplyCurrentByWindow(uint32_t *window_data, int index) {
        RegMP    A(_registers, 0, 0, _words), C(_registers, 0, _words*2, _words), RR(_registers, 0, _words, _words*2), T(_registers, 0, _words*3, _words);

#ifdef XMAD
        #pragma unroll
        for(int word=0;word<_words;word++)
          T[word]=window_data[index*_words*32 + word*32 + _words*32];
        mul(RR, A, T);
#else
        #pragma unroll
        for(int word=0;word<_words;word++)
          C[word]=window_data[index*_words*32 + word*32 + _words*32];
        kar_mul<_km>(RR, A, C, T);
#endif
      }

      __device__ __forceinline__ void multiplyCurrentByConstant(uint32_t *window_data, int index) {
/*
        RegMP    A(_registers, 0, 0, _words), C(_registers, 0, _words*2, _words), RR(_registers, 0, _words, _words*2), T(_registers, 0, _words*3, _words);

#ifdef XMAD
        #pragma unroll
        for(int word=0;word<_words;word++)
          T[word]=window_data[index*_words*32 + word*32];
        mul(RR, A, T);
#else
        #pragma unroll
        for(int word=0;word<_words;word++)
          C[word]=window_data[index*_words*32 + word*32];
        mul(RR, A, C);
#endif
*/
      }

      __device__ __forceinline__ void multiplyCurrentByOne(uint32_t *window) {
        RegMP A(_registers, 0, 0, _words), RR(_registers, 0, _words, _words*2);

        #pragma unroll
        for(int word=0;word<_words;word++)
          RR[word]=A[word];
        #pragma unroll
        for(int word=_words;word<2*_words;word++)
          RR[word]=0;
      }
  };
}

