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
namespace xmp {
  template <bool use_sm_cache, int size>
  class Digitized {
    public:
      uint32_t _np0;
      uint32_t _digits;
      uint32_t _registers[size*4+1];
      uint32_t _window_bits;

      __device__ __forceinline__ Digitized(int32_t digits, int32_t width, int32_t bits, int32_t window_bits) {
        _digits=digits;
        _window_bits=window_bits;
      }

      __device__ __forceinline__ void initialize(uint32_t *window_data, int32_t mod_count) {
        DigitMP<size> WINDOW(false, false, window_data), N(WINDOW, 0, _digits);

        _np0=computeNP0(N.load_digit_word(0, 0));

        if(use_sm_cache) {
          DigitMP<size> SM(_digits, mod_count);

          set<size>(_registers, SM, N);
        }
      }

      __device__ __forceinline__ void computeRModuloN(uint32_t *window_data) {
        PTXInliner    inliner;
        RegMP         A(_registers, 0, 0, size), B(_registers, 0, size, size);
        DigitMP<size> WINDOW(false, false, window_data), N(WINDOW, 0, _digits), CURRENT(WINDOW, _digits, _digits);
        uint32_t      borrow, zero=0;

        set_ui(B, 0);

        borrow=0;
        #pragma unroll
        for(int digit=0;digit<_digits;digit++) {
          N.load_digit(A, digit);
          inliner.SUB_CC(borrow, zero, borrow);
          _sub(A, B, A, true, true);
          inliner.SUBC(borrow, zero, zero);
          CURRENT.store_digit(A, digit);
        }
      }

/*
      __device__ __forceinline__ void loadArgument(uint32_t *a_data, int32_t a_len, int32_t a_stride, int32_t a_count) {
        DigitMP<size> ARG(a_data, a_len, a_stride), CURRENT(window_data, );

        a_data=a_data + thread%a_count;
        #pragma unroll
        for(int word=0;word<_words;word++)
          if(word<a_len)
            A[word]=a_data[word*a_stride];
          else
            A[word]=0;
      }
*/

      __device__ __forceinline__ void storeResult(uint32_t *window_data, uint32_t *out_data, int32_t out_len, int32_t out_stride) {
        DigitMP<size> WINDOW(false, false, window_data);
        DigitMP<size> CURRENT(WINDOW, _digits, _digits), OUT(false, false, out_data, out_len, out_stride, 0x7FFFFFFF, 0);

        set<size>(_registers, OUT, CURRENT);
      }


      __device__ __forceinline__ void printCurrent(uint32_t *window_data, const char *text) {
        DigitMP<size> WINDOW(false, false, window_data);
        DigitMP<size> CURRENT(WINDOW, _digits, _digits);

        printf("%s=", text);
        ppp(CURRENT);
      }

      __device__ __forceinline__ void loadWindow(uint32_t *window_data, int32_t index) {
        DigitMP<size> WINDOW(false, false, window_data), CURRENT(WINDOW, _digits, _digits), W(WINDOW, (index+4)*_digits, _digits);

        set<size>(_registers, CURRENT, W);
      }

      __device__ __forceinline__ void storeWindow(uint32_t *window_data, int index) {
        DigitMP<size> WINDOW(false, false, window_data), CURRENT(WINDOW, _digits, _digits), W(WINDOW, (index+4)*_digits, _digits);

        set<size>(_registers, W, CURRENT);
      }

      __device__ __forceinline__ uint32_t getBits(uint32_t *window_data, int32_t bitOffset, int32_t bitLength) {
        DigitMP<size> WINDOW(false, false, window_data), EXP(WINDOW, ((1<<_window_bits) + 4)*_digits, 0xFFFF);

        return EXP.get_bits(bitOffset, bitLength);
      }

      __device__ __forceinline__ void reduceCurrent(uint32_t *window_data, int32_t mod_count) {
        DigitMP<size> WINDOW(false, false, window_data), CURRENT(WINDOW, _digits, _digits), ACC(WINDOW, 2*_digits, 2*_digits);

        if(use_sm_cache) {
          DigitMP<size> SM(_digits, mod_count);

          _reduce<size>(_registers, CURRENT, ACC, SM, _np0);
        }
        else {
          DigitMP<size> N(WINDOW, 0, _digits);

          _reduce<size>(_registers, CURRENT, ACC, N, _np0);
        }
      }

      __device__ __forceinline__ void squareCurrent(uint32_t *window_data) {
        DigitMP<size> WINDOW(false, false, window_data), CURRENT(WINDOW, _digits, _digits), ACC(WINDOW, 2*_digits, 2*_digits);

        _sqr<size>(_registers, ACC, CURRENT);
      }

      __device__ __forceinline__ void multiplyCurrentByWindow(uint32_t *window_data, int index) {
        DigitMP<size> WINDOW(false, false, window_data);
        DigitMP<size> CURRENT(WINDOW, _digits, _digits), ACC(WINDOW, 2*_digits, 2*_digits), W(WINDOW, (index+4)*_digits, _digits);

        _mul<size>(_registers, ACC, CURRENT, W);
      }

      __device__ __forceinline__ void multiplyCurrentByConstant(uint32_t *window_data, int index) {
      }

      __device__ __forceinline__ void multiplyCurrentByOne(uint32_t *window_data) {
        DigitMP<size> WINDOW(false, false, window_data);
        DigitMP<size> CURRENT(WINDOW, _digits, _digits), ACC_LOW(WINDOW, 2*_digits, _digits), ACC_HIGH(WINDOW, 3*_digits, _digits);

        set<size>(_registers, ACC_LOW, CURRENT);
        set_ui<size>(_registers, ACC_HIGH, 0);
      }
  };
}

