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
  __device__ __forceinline__ void sqr(uint32_t *registers, DigitMP<size> r, DigitMP<size> a) {
    PTXInliner inliner;
    RegMP      ACC(registers, 0, 0, 2*size+1), ACC_LOW(registers, 0, 0, size), ACC_HIGH(registers, 0, size, size);
    RegMP      A(registers, 0, 2*size+1, size), B(registers, 0, 3*size+1, size);
    int32_t    col, row;
    uint32_t   carryDouble=0, carryDiagonal=0, zero=0, ff=0xFFFFFFFF;

    set_ui(ACC_HIGH, 0);
    ACC[2*size]=0;
    #pragma nounroll
    for(col=0;col<r.digits();col++) {
      shift_right_const(ACC, ACC, size);

      #pragma nounroll
      for(row=(col<a.digits()) ? 0 : col-a.digits()+1; row<(col+1)/2; row++) {
        a.load_digit(A, col-row);
        a.load_digit(B, row);
        accumulate_mul(ACC, A, B);
      }
      if(col-row==row) {
        a.load_digit(B, row);
        accumulate_half_sqr(ACC, B);
        inliner.ADD_CC(carryDouble, carryDouble, ff);
        _add(ACC_LOW, ACC_LOW, ACC_LOW, true, true);
        inliner.ADDC(carryDouble, zero, zero);
        inliner.ADD_CC(carryDiagonal, carryDiagonal, ff);
        PTXChain chain1(size, true, true);
        #pragma unroll
        for(int word=0;word<size;word++) {
          if(word%2==0)
            chain1.MADLO(ACC[word], B[word/2], B[word/2], ACC[word]);
          else
            chain1.MADHI(ACC[word], B[word/2], B[word/2], ACC[word]);
        }
        chain1.end();
        inliner.ADDC(carryDiagonal, zero, zero);
      }
      else if(col-row+1==row) {
        inliner.ADD_CC(carryDouble, carryDouble, ff);
        _add(ACC_LOW, ACC_LOW, ACC_LOW, true, true);
        inliner.ADDC(carryDouble, zero, zero);
        inliner.ADD_CC(carryDiagonal, carryDiagonal, ff);
        PTXChain chain2(size, true, true);
        #pragma unroll
        for(int word=0;word<size;word++) {
          if((word+size)%2==0)
            chain2.MADLO(ACC[word], B[(word+size)/2], B[(word+size)/2], ACC[word]);
          else
            chain2.MADHI(ACC[word], B[(word+size)/2], B[(word+size)/2], ACC[word]);
        }
        chain2.end();
        inliner.ADDC(carryDiagonal, zero, zero);
      }
      r.store_digit(ACC_LOW, col);
    }
  }

  template<uint32_t size>
  __device__ __forceinline__ void _sqr(uint32_t *registers, DigitMP<size> r, DigitMP<size> a) {
    PTXInliner inliner;
    RegMP      ACC(registers, 0, 0, 2*size+1), ACC_LOW(registers, 0, 0, size), ACC_HIGH(registers, 0, size, size);
    RegMP      A(registers, 0, 2*size+1, size), B(registers, 0, 3*size+1, size);
    int32_t    col, row;
    uint32_t   carryDouble=0, carryDiagonal=0, zero=0, ff=0xFFFFFFFF;

    set_ui(ACC_HIGH, 0);
    ACC[2*size]=0;
    #pragma nounroll
    for(col=0;col<a.digits()*2-1;col++) {
      shift_right_const(ACC, ACC, size);

      #pragma nounroll
      for(row=(col<a.digits()) ? 0 : col-a.digits()+1; row<(col+1)/2; row++) {
        a.load_digit(A, col-row);
        a.load_digit(B, row);
        accumulate_mul(ACC, A, B);
      }
      if(col-row==row) {
        a.load_digit(B, row);
        accumulate_half_sqr(ACC, B);
        inliner.ADD_CC(carryDouble, carryDouble, ff);
        _add(ACC_LOW, ACC_LOW, ACC_LOW, true, true);
        inliner.ADDC(carryDouble, zero, zero);
        inliner.ADD_CC(carryDiagonal, carryDiagonal, ff);
        PTXChain chain1(size, true, true);
        #pragma unroll
        for(int word=0;word<size;word++) {
          if(word%2==0)
            chain1.MADLO(ACC[word], B[word/2], B[word/2], ACC[word]);
          else
            chain1.MADHI(ACC[word], B[word/2], B[word/2], ACC[word]);
        }
        chain1.end();
        inliner.ADDC(carryDiagonal, zero, zero);
      }
      else {
        inliner.ADD_CC(carryDouble, carryDouble, ff);
        _add(ACC_LOW, ACC_LOW, ACC_LOW, true, true);
        inliner.ADDC(carryDouble, zero, zero);
        inliner.ADD_CC(carryDiagonal, carryDiagonal, ff);
        PTXChain chain2(size, true, true);
        #pragma unroll
        for(int word=0;word<size;word++) {
          if((word+size)%2==0)
            chain2.MADLO(ACC[word], B[(word+size)/2], B[(word+size)/2], ACC[word]);
          else
            chain2.MADHI(ACC[word], B[(word+size)/2], B[(word+size)/2], ACC[word]);
        }
        chain2.end();
        inliner.ADDC(carryDiagonal, zero, zero);
      }
      r.store_digit(ACC_LOW, col);
    }

    // one final round of doubling to go
    inliner.ADD_CC(carryDouble, carryDouble, ff);
    _add(ACC_HIGH, ACC_HIGH, ACC_HIGH, true, false);
    // can't carry out
    inliner.ADD_CC(carryDiagonal, carryDiagonal, ff);
    PTXChain chain3(size, true, false);
    #pragma unroll
    for(int word=size;word<2*size;word++) {
      if(word%2==0)
        chain3.MADLO(ACC[word], B[word/2], B[word/2], ACC[word]);
      else
        chain3.MADHI(ACC[word], B[word/2], B[word/2], ACC[word]);
    }
    chain3.end();
    // can't carry out

    r.store_digit(ACC_HIGH, a.digits()*2-1);
  }
}

