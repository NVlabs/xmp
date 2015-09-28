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
  __device__ __forceinline__ void _reduce(uint32_t *registers, DigitMP<size> r, DigitMP<size> xx, DigitMP<size> n, uint32_t np0) {
    PTXInliner inliner;
    RegMP      ACC(registers, 0, 0, 2*size+1), ACC_LOW(registers, 0, 0, size), ACC_HIGH(registers, 0, size, size), ACC_LH(registers, 0, 0, 2*size);
    RegMP      A(registers, 0, 2*size+1, size), B(registers, 0, 3*size+1, size);
    int32_t    col, row, from, to;
    uint32_t   mask, borrow, zero=0;

    // assumes length of r and length of n match
    // assumes xx is twice the length of n

    set_ui(ACC_HIGH, 0);
    ACC[2*size]=0;

    #pragma nounroll
    for(col=0;col<xx.digits()-1;col++) {
      xx.load_digit(B, col);
      _add(ACC_LOW, ACC_HIGH, B, false, true);
      inliner.ADDC(ACC[size], ACC[2*size], zero);

      #pragma unroll
      for(int word=size+1;word<=2*size;word++)
        ACC[word]=0;

      from=(col<n.digits()) ? 0 : col-n.digits()+1;
      to=(col<n.digits()) ? col : n.digits();

      #pragma nounroll
      for(row=from;row<to;row++) {
        r.load_digit(A, row);
        n.load_digit(B, col-row);
        accumulate_mul(ACC, A, B);
      }

      if(col<n.digits()) {
        n.load_digit(B, 0);
        ACC[2*size]+=_reduce(ACC_LH, A, B, np0);
        r.store_digit(A, col);
      }
      else
        r.store_digit(ACC_LOW, col-n.digits());
    }

    xx.load_digit(B, xx.digits()-1);
    _add(ACC_HIGH, ACC_HIGH, B, false, true);
    inliner.ADDC(ACC[2*size], ACC[2*size], zero);

    r.store_digit(ACC_HIGH, xx.digits()-n.digits()-1);

    mask=-ACC[2*size];

    borrow=0;
    #pragma nounroll
    for(int col=0;col<r.digits();col++) {
      r.load_digit(A, col);
      n.load_digit(B, col);

      bitwise_and(B, B, mask);
      inliner.SUB_CC(borrow, zero, borrow);
      _sub(A, A, B, true, true);
      inliner.SUBC(borrow, zero, zero);

      r.store_digit(A, col);
    }
  }
}
