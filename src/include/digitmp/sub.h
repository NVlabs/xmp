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
  template<uint32_t size>
  __device__ __forceinline__ void sub(uint32_t *registers, DigitMP<size> r, DigitMP<size> a, DigitMP<size> b) {
    PTXInliner inliner;
    RegMP      A(registers, 0, 0, size), B(registers, 0, size, size);
    uint32_t   borrow, zero=0;

    borrow=0;
    for(int32_t index=0;index<r.digits();index++) {
      a.load_digit(A, index);
      b.load_digit(B, index);

      inliner.SUB_CC(borrow, zero, borrow);
      _sub(A, A, B, true, true);
      inliner.SUBC(borrow, zero, zero);

      r.store_digit(A, index);
    }
  }
}

