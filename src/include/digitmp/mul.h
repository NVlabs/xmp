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
  __device__ __forceinline__ void mul(uint32_t *registers, DigitMP<size> r, DigitMP<size> a, DigitMP<size> b) {
    DigitMP<size> inner, outer;
    RegMP         ACC(registers, 0, 0, 2*size+1), ACC_LOW(registers, 0, 0, size), ACC_HIGH(registers, 0, size, size);
    RegMP         A(registers, 0, 2*size+1, size), B(registers, 0, 3*size+1, size);
    int32_t       col, row, from, to;

    if(a.digits()<=b.digits()) {
      outer.set(a);
      inner.set(b);
    }
    else {
      outer.set(b);
      inner.set(a);
    }

    set_ui(ACC_HIGH, 0);
    ACC[2*size]=0;
    #pragma nounroll
    for(col=0;col<r.digits();col++) {
      shift_right_const(ACC, ACC, size);
      from=(col<outer.digits()) ? 0 : col-outer.digits()+1;
      to=(col<outer.digits()) ? col+1 : min(col+1, inner.digits());
      #pragma nounroll
      for(row=from;row<to;row++) {
        outer.load_digit(A, col-row);
        inner.load_digit(B, row);
        accumulate_mul(ACC, A, B);
      }
      r.store_digit(ACC_LOW, col);
    }
  }

  // here we assume a.digits()==b.digits()
  template<int32_t size>
  __device__ __forceinline__ void _mul(uint32_t *registers, DigitMP<size> r, DigitMP<size> a, DigitMP<size> b) {
    RegMP   ACC(registers, 0, 0, 2*size+1), ACC_LOW(registers, 0, 0, size), ACC_HIGH(registers, 0, size, size);
    RegMP   A(registers, 0, 2*size+1, size), B(registers, 0, 3*size+1, size);
    int32_t col, row, from, to;

    set_ui(ACC_HIGH, 0);
    ACC[2*size]=0;
    #pragma nounroll
    for(col=0;col<2*a.digits()-1;col++) {
      shift_right_const(ACC, ACC, size);
      from=(col<a.digits()) ? 0 : col-a.digits()+1;
      to=(col<a.digits()) ? col+1 : a.digits();
      #pragma nounroll
      for(row=from;row<to;row++) {
        a.load_digit(A, row);
        b.load_digit(B, col-row);
        accumulate_mul(ACC, A, B);
      }
      r.store_digit(ACC_LOW, col);
    }
    r.store_digit(ACC_HIGH, 2*a.digits()-1);
  }
}


