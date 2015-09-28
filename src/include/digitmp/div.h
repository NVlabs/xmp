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

#define INST 1

  template<uint32_t size>
  __device__ __forceinline__ int32_t _div_add(uint32_t *scratch, DigitMP<size> xx, DigitMP<size> d, int32_t shift, uint32_t mask) {
    PTXInliner inliner;
    RegMP      A(scratch, 0, 0, size), B(scratch, 0, size, size);
    uint32_t   carry, zero=0, ff=0xFFFFFFFF;

    carry=0;

    #pragma nounroll
    for(int index=(shift>=0) ? 0 : -shift;index<d.digits();index++) {
      xx.load_digit(A, index+shift);
      d.load_digit(B, index);

      bitwise_and(B, B, mask);

      inliner.ADD_CC(carry, carry, ff);
      _add(A, A, B, true, true);
      inliner.ADDC(carry, zero, zero);

      xx.store_digit(A, index+shift);
    }

    return carry;
  }

  template<uint32_t size>
  __device__ __forceinline__ int32_t _div_sub(uint32_t *scratch, DigitMP<size> xx, DigitMP<size> d, int32_t shift) {
    PTXInliner inliner;
    RegMP      A(scratch, 0, 0, size), B(scratch, 0, size, size);
    uint32_t   borrow, zero=0;

    borrow=0;

    #pragma nounroll
    for(int index=(shift>=0) ? 0 : -shift;index<d.digits();index++) {
      xx.load_digit(A, index+shift);
      d.load_digit(B, index);

      inliner.SUB_CC(borrow, zero, borrow);
      _sub(A, A, B, true, true);
      inliner.SUBC(borrow, zero, zero);

      xx.store_digit(A, index+shift);
    }
    return borrow;
  }

  template<uint32_t size>
  __device__ __forceinline__ int32_t _div_mul_sub(uint32_t *scratch, DigitMP<size> xx, DigitMP<size> d, RegMP EST, int32_t shift) {
    PTXInliner inliner;
    RegMP      ACC_LOW(scratch, 0, 0, size), ACC(scratch, 0, 0, size*2+1), A(scratch, 0, size*2+1, size);
    uint32_t   borrow=0, zero=0;

    set_ui(ACC, 0);

    #pragma nounroll
    for(int index=(shift>=0) ? 0 : -shift;index<d.digits();index++) {
      d.load_digit(A, index);

      // could be a _mad if we had a maxwell version
      accumulate_mul(ACC, A, EST);

      xx.load_digit(A, index+shift);

      inliner.SUB_CC(borrow, zero, borrow);
      _sub(A, A, ACC_LOW, true, true);
      inliner.SUBC(borrow, zero, zero);

      xx.store_digit(A, index+shift);

      shift_right_const(ACC, ACC, size);
    }

    A[0]=xx.load_digit_word(d.digits()+shift, 0);


    inliner.SUB_CC(borrow, zero, borrow);
    inliner.SUBC_CC(A[0], A[0], ACC[0]);

    return A[0];
  }

  template<uint32_t size>
  __device__ __forceinline__ bool div(uint32_t *registers, DigitMP<size> q, DigitMP<size> numerator, DigitMP<size> denominator, uint32_t *space) {
    PTXInliner    inliner;
    DigitMP<size> scratch(false, false, space, numerator.digits()+denominator.digits()+2);
    DigitMP<size> inverse(scratch, 0, 1), xx(scratch, 1, numerator.digits()+1), d(scratch, numerator.digits()+2, denominator.digits());
    RegMP         ACC_LOW(registers, 0, 0, size), ACC_HIGH(registers, 0, size, size), ACC(registers, 0, 0, 2*size+2);
    RegMP         A_HIGH(registers, 0, 2*size+3, size), A(registers, 0, 2*size+2, size+1), B_LOW(registers, 0, 3*size+3, size), B(registers, 0, 3*size+3, size+1);
    uint32_t      shifted, borrow, load, mask1, mask2, update, zero=0;

    // xx must have space for an extra digit and be at least 3 digits in length
    // d must be at least 2 digits in length

    shifted=count_leading_zeros(registers, denominator);
    if(shifted==denominator.digits()*size*32)
      return false;
    shift_left<size>(registers, d, denominator, shifted);
    shift_left<size>(registers, xx, numerator, remainder<size*32>(shifted));

    #pragma unroll
    for(int word=0;word<2*size;word++)
      ACC[word]=0xFFFFFFFF;
    ACC[2*size]=0;
    ACC[2*size+1]=0;

    d.load_digit(B_LOW, d.digits()-1);
    div(ACC, B_LOW);

    inverse.store_digit(ACC_LOW, 0);

// this can't happen!
//    mask1=_div_sub(registers, xx, d, xx.digits()-d.digits());
//    _div_add(registers, xx, d, xx.digits()-d.digits(), mask1);
//    highBit=mask1+1;

    // digit points to the current high digit of the numerator
    #pragma nounroll
    for(int32_t digit=xx.digits()-1;digit>=d.digits()-divide<size*32>(shifted);digit--) {
      B[size]=0;
      inverse.load_digit(B_LOW, 0);
      xx.load_digit(A_HIGH, digit);

      // compute estimate in B
      mul(ACC.lower(2*size), A_HIGH, B_LOW);
      add(B_LOW, A_HIGH, ACC_HIGH);
      add_ui(B, B, 3);

      // load top size+1 words from d
      A[0]=d.load_digit_word(d.digits()-2, size-1);
      d.load_digit(A_HIGH, d.digits()-1);

      // compute full A*B, but we're only going to use lower size+2 words, compiler should optimize this
      mul(ACC, A, B);

      // load corresponding words from xx, and subtract
      borrow=0xFFFFFFF;   // compute value-1, which is the same as rounding xx up
      load=xx.load_digit_word(digit-2, size-1);
      inliner.SUB_CC(borrow, zero, borrow);
      inliner.SUBC_CC(ACC[0], ACC[0], load);
      inliner.SUBC(borrow, zero, zero);

      #pragma unroll
      for(int word=0;word<size;word++) {
        load=xx.load_digit_word(digit-1, word);
        inliner.SUB_CC(borrow, zero, borrow);
        inliner.SUBC_CC(ACC[word+1], ACC[word+1], load);
        inliner.SUBC(borrow, zero, zero);
      }

      load=xx.load_digit_word(digit, 0);
      inliner.SUB_CC(borrow, zero, borrow);
      inliner.SUBC(ACC[size+1], ACC[size+1], load);

      // refine the estimate
      update=0;
      #pragma nounroll
      for(int count=0;count<5;count++) {
        mask1=((int32_t)~ACC[size+1])>>31;
        update=update-mask1;
        bitwise_and(A, A, mask1);
        _sub(ACC.lower(size+1), ACC.lower(size+1), A, false, true);
        inliner.SUBC(ACC[size+1], ACC[size+1], zero);
      }
      sub_ui(B, B, update);

      // Clamp the estimate: if B<0 then B=0, if B>Digit.MAX_VALUE, B=Digit.MAX_VALUE
      mask1=(((int32_t)B[size])>0) ? 0xFFFFFFFF : 0;     // or mask
      mask2=(((int32_t)B[size])>=0) ? 0xFFFFFFFF : 0;    // and mask
      #pragma unroll
      for(int word=0;word<size;word++)
        B_LOW[word]=(B_LOW[word] | mask1) & mask2;

      // Subtract est * d
      mask1=_div_mul_sub(registers, xx, d, B_LOW, digit-d.digits());
      _div_add(registers, xx, d, digit-d.digits(), mask1);
      sub_ui(B_LOW, B_LOW, -mask1);

      xx.store_digit(B_LOW, digit);
    }

    shift_right<size>(registers, q, xx, (d.digits()-divide<size*32>(shifted))*size*32);

    return true;
  }

  template<uint32_t size>
  __device__ __forceinline__ bool div_rem(uint32_t *registers, DigitMP<size> q, DigitMP<size> r, DigitMP<size> numerator, DigitMP<size> denominator, uint32_t *space) {
    PTXInliner    inliner;
    DigitMP<size> scratch(false, false, space, numerator.digits()+denominator.digits()+2);
    DigitMP<size> inverse(scratch, 0, 1), xx(scratch, 1, numerator.digits()+1), d(scratch, numerator.digits()+2, denominator.digits());
    RegMP         ACC_LOW(registers, 0, 0, size), ACC_HIGH(registers, 0, size, size), ACC(registers, 0, 0, 2*size+2);
    RegMP         A_HIGH(registers, 0, 2*size+3, size), A(registers, 0, 2*size+2, size+1), B_LOW(registers, 0, 3*size+3, size), B(registers, 0, 3*size+3, size+1);
    uint32_t      shifted, borrow, load, mask1, mask2, update, zero=0;

    // xx must have space for an extra digit and be at least 3 digits in length
    // d must be at least 2 digits in length

    shifted=count_leading_zeros(registers, denominator);
    if(shifted==denominator.digits()*size*32)
      return false;
    shift_left<size>(registers, d, denominator, shifted);
    shift_left<size>(registers, xx, numerator, remainder<size*32>(shifted));

    #pragma unroll
    for(int word=0;word<2*size;word++)
      ACC[word]=0xFFFFFFFF;
    ACC[2*size]=0;
    ACC[2*size+1]=0;

    d.load_digit(B_LOW, d.digits()-1);
    div(ACC, B_LOW);

    inverse.store_digit(ACC_LOW, 0);

// this can't happen!
//    mask1=_div_sub(registers, xx, d, xx.digits()-d.digits());
//    _div_add(registers, xx, d, xx.digits()-d.digits(), mask1);
//    highBit=mask1+1;

    // digit points to the current high digit of the numerator
    #pragma nounroll
    for(int32_t digit=xx.digits()-1;digit>=d.digits()-divide<size*32>(shifted);digit--) {
      B[size]=0;
      inverse.load_digit(B_LOW, 0);
      xx.load_digit(A_HIGH, digit);

      // compute estimate in B
      mul(ACC.lower(2*size), A_HIGH, B_LOW);
      add(B_LOW, A_HIGH, ACC_HIGH);
      add_ui(B, B, 3);

      // load top size+1 words from d
      A[0]=d.load_digit_word(d.digits()-2, size-1);
      d.load_digit(A_HIGH, d.digits()-1);

      // compute full A*B, but we're only going to use lower size+2 words, compiler should optimize this
      mul(ACC, A, B);

      // load corresponding words from xx, and subtract
      borrow=0xFFFFFFF;   // compute value-1, which is the same as rounding xx up
      load=xx.load_digit_word(digit-2, size-1);
      inliner.SUB_CC(borrow, zero, borrow);
      inliner.SUBC_CC(ACC[0], ACC[0], load);
      inliner.SUBC(borrow, zero, zero);

      #pragma unroll
      for(int word=0;word<size;word++) {
        load=xx.load_digit_word(digit-1, word);
        inliner.SUB_CC(borrow, zero, borrow);
        inliner.SUBC_CC(ACC[word+1], ACC[word+1], load);
        inliner.SUBC(borrow, zero, zero);
      }

      load=xx.load_digit_word(digit, 0);
      inliner.SUB_CC(borrow, zero, borrow);
      inliner.SUBC(ACC[size+1], ACC[size+1], load);

      // refine the estimate
      update=0;
      #pragma nounroll
      for(int count=0;count<5;count++) {
        mask1=((int32_t)~ACC[size+1])>>31;
        update=update-mask1;
        bitwise_and(A, A, mask1);
        _sub(ACC.lower(size+1), ACC.lower(size+1), A, false, true);
        inliner.SUBC(ACC[size+1], ACC[size+1], zero);
      }
      sub_ui(B, B, update);

      // Clamp the estimate: if B<0 then B=0, if B>Digit.MAX_VALUE, B=Digit.MAX_VALUE
      mask1=(((int32_t)B[size])>0) ? 0xFFFFFFFF : 0;     // or mask
      mask2=(((int32_t)B[size])>=0) ? 0xFFFFFFFF : 0;    // and mask
      #pragma unroll
      for(int word=0;word<size;word++)
        B_LOW[word]=(B_LOW[word] | mask1) & mask2;

      // Subtract est * d
      mask1=_div_mul_sub(registers, xx, d, B_LOW, digit-d.digits());
      _div_add(registers, xx, d, digit-d.digits(), mask1);
      sub_ui(B_LOW, B_LOW, -mask1);

      xx.store_digit(B_LOW, digit);
    }

    DigitMP<size> lowerXX(scratch, 1, d.digits()-divide<size*32>(shifted));

    shift_right<size>(registers, r, lowerXX, remainder<size*32>(shifted));
    shift_right<size>(registers, q, xx, (d.digits()-divide<size*32>(shifted))*size*32);

    return true;
  }

  template<uint32_t size>
  __device__ __forceinline__ bool rem(uint32_t *registers, DigitMP<size> r, DigitMP<size> numerator, DigitMP<size> denominator, uint32_t *space) {
    PTXInliner    inliner;
    DigitMP<size> scratch(false, false, space, numerator.digits()+denominator.digits()+2);
    DigitMP<size> inverse(scratch, 0, 1), xx(scratch, 1, numerator.digits()+1), d(scratch, numerator.digits()+2, denominator.digits());
    RegMP         ACC_LOW(registers, 0, 0, size), ACC_HIGH(registers, 0, size, size), ACC(registers, 0, 0, 2*size+2);
    RegMP         A_HIGH(registers, 0, 2*size+3, size), A(registers, 0, 2*size+2, size+1), B_LOW(registers, 0, 3*size+3, size), B(registers, 0, 3*size+3, size+1);
    uint32_t      shifted, borrow, load, mask1, mask2, update, zero=0;

    // xx must have space for an extra digit and be at least 3 digits in length
    // d must be at least 2 digits in length

    shifted=count_leading_zeros(registers, denominator);
    if(shifted==denominator.digits()*size*32)
      return false;
    shift_left<size>(registers, d, denominator, shifted);
    shift_left<size>(registers, xx, numerator, remainder<size*32>(shifted));

    #pragma unroll
    for(int word=0;word<2*size;word++)
      ACC[word]=0xFFFFFFFF;
    ACC[2*size]=0;
    ACC[2*size+1]=0;

    d.load_digit(B_LOW, d.digits()-1);
    div(ACC, B_LOW);

    inverse.store_digit(ACC_LOW, 0);

// this can't happen!
//    mask1=_div_sub(registers, xx, d, xx.digits()-d.digits());
//    _div_add(registers, xx, d, xx.digits()-d.digits(), mask1);
//    highBit=mask1+1;

    // digit points to the current high digit of the numerator
    #pragma nounroll
    for(int32_t digit=xx.digits()-1;digit>=d.digits()-divide<size*32>(shifted);digit--) {
      B[size]=0;
      inverse.load_digit(B_LOW, 0);
      xx.load_digit(A_HIGH, digit);

      // compute estimate in B
      mul(ACC.lower(2*size), A_HIGH, B_LOW);
      add(B_LOW, A_HIGH, ACC_HIGH);
      add_ui(B, B, 3);

      // load top size+1 words from d
      A[0]=d.load_digit_word(d.digits()-2, size-1);
      d.load_digit(A_HIGH, d.digits()-1);

      // compute full A*B, but we're only going to use lower size+2 words, compiler should optimize this
      mul(ACC, A, B);

      // load corresponding words from xx, and subtract
      borrow=0xFFFFFFF;   // compute value-1, which is the same as rounding xx up
      load=xx.load_digit_word(digit-2, size-1);
      inliner.SUB_CC(borrow, zero, borrow);
      inliner.SUBC_CC(ACC[0], ACC[0], load);
      inliner.SUBC(borrow, zero, zero);

      #pragma unroll
      for(int word=0;word<size;word++) {
        load=xx.load_digit_word(digit-1, word);
        inliner.SUB_CC(borrow, zero, borrow);
        inliner.SUBC_CC(ACC[word+1], ACC[word+1], load);
        inliner.SUBC(borrow, zero, zero);
      }

      load=xx.load_digit_word(digit, 0);
      inliner.SUB_CC(borrow, zero, borrow);
      inliner.SUBC(ACC[size+1], ACC[size+1], load);

      // refine the estimate
      update=0;
      #pragma nounroll
      for(int count=0;count<5;count++) {
        mask1=((int32_t)~ACC[size+1])>>31;
        update=update-mask1;
        bitwise_and(A, A, mask1);
        _sub(ACC.lower(size+1), ACC.lower(size+1), A, false, true);
        inliner.SUBC(ACC[size+1], ACC[size+1], zero);
      }
      sub_ui(B, B, update);

      // Clamp the estimate: if B<0 then B=0, if B>Digit.MAX_VALUE, B=Digit.MAX_VALUE
      mask1=(((int32_t)B[size])>0) ? 0xFFFFFFFF : 0;     // or mask
      mask2=(((int32_t)B[size])>=0) ? 0xFFFFFFFF : 0;    // and mask
      #pragma unroll
      for(int word=0;word<size;word++)
        B_LOW[word]=(B_LOW[word] | mask1) & mask2;

      // Subtract est * d
      mask1=_div_mul_sub(registers, xx, d, B_LOW, digit-d.digits());
      _div_add(registers, xx, d, digit-d.digits(), mask1);
      sub_ui(B_LOW, B_LOW, -mask1);

      xx.store_digit(B_LOW, digit);
    }

    DigitMP<size> lowerXX(scratch, 1, d.digits()-divide<size*32>(shifted));

    shift_right<size>(registers, r, lowerXX, remainder<size*32>(shifted));

    return true;
  }

  template<uint32_t size>
  __device__ __forceinline__ bool _rem(uint32_t *registers, DigitMP<size> r, DigitMP<size> xx, DigitMP<size> denominator, DigitMP<size> d, DigitMP<size> inverse) {
    PTXInliner inliner;
    RegMP      ACC_LOW(registers, 0, 0, size), ACC_HIGH(registers, 0, size, size), ACC(registers, 0, 0, 2*size+2);
    RegMP      A_HIGH(registers, 0, 2*size+3, size), A(registers, 0, 2*size+2, size+1), B_LOW(registers, 0, 3*size+3, size), B(registers, 0, 3*size+3, size+1);
    uint32_t   shifted, borrow, load, mask1, mask2, update, zero=0;

    // xx must have space for an extra digit and be at least 3 digits in length
    // d must be at least 2 digits in length

    shifted=count_leading_zeros(registers, denominator);
    if(shifted==denominator.digits()*size*32)
      return false;
    shift_left<size>(registers, d, denominator, shifted);
    shift_left<size>(registers, xx, xx, remainder<size*32>(shifted));

    #pragma unroll
    for(int word=0;word<2*size;word++)
      ACC[word]=0xFFFFFFFF;
    ACC[2*size]=0;
    ACC[2*size+1]=0;

    d.load_digit(B_LOW, d.digits()-1);
    div(ACC, B_LOW);

    inverse.store_digit(ACC_LOW, 0);

    // digit points to the current high digit of the numerator
    #pragma nounroll
    for(int32_t digit=xx.digits()-1;digit>=d.digits()-divide<size*32>(shifted);digit--) {
      B[size]=0;
      inverse.load_digit(B_LOW, 0);
      xx.load_digit(A_HIGH, digit);

      // compute estimate in B
      mul(ACC.lower(2*size), A_HIGH, B_LOW);
      add(B_LOW, A_HIGH, ACC_HIGH);
      add_ui(B, B, 3);

      // load top size+1 words from d
      A[0]=d.load_digit_word(d.digits()-2, size-1);
      d.load_digit(A_HIGH, d.digits()-1);

//      d.load_digit(A, d.digits(), -(size+1));

      // compute full A*B, but we're only going to use lower size+2 words, compiler should optimize this
      mul(ACC, A, B);

      // load corresponding words from xx, and subtract
      borrow=0xFFFFFFF;      // compute value-1, which is the same as rounding xx up
      load=xx.load_digit_word(digit-2, size-1);
      inliner.SUB_CC(borrow, zero, borrow);
      inliner.SUBC_CC(ACC[0], ACC[0], load);
      inliner.SUBC(borrow, zero, zero);

      #pragma unroll
      for(int word=0;word<size;word++) {
        load=xx.load_digit_word(digit-1, word);
        inliner.SUB_CC(borrow, zero, borrow);
        inliner.SUBC_CC(ACC[word+1], ACC[word+1], load);
        inliner.SUBC(borrow, zero, zero);
      }

      load=xx.load_digit_word(digit, 0);
      inliner.SUB_CC(borrow, zero, borrow);
      inliner.SUBC(ACC[size+1], ACC[size+1], load);

      // refine the estimate
      update=0;
      #pragma nounroll
      for(int count=0;count<5;count++) {
        mask1=((int32_t)~ACC[size+1])>>31;
        update=update-mask1;
        bitwise_and(A, A, mask1);
        _sub(ACC.lower(size+1), ACC.lower(size+1), A, false, true);
        inliner.SUBC(ACC[size+1], ACC[size+1], zero);
      }
      sub_ui(B, B, update);

      // Clamp the estimate: if B<0 then B=0, if B>Digit.MAX_VALUE, B=Digit.MAX_VALUE
      mask1=(((int32_t)B[size])>0) ? 0xFFFFFFFF : 0;     // or mask
      mask2=(((int32_t)B[size])>=0) ? 0xFFFFFFFF : 0;    // and mask
      #pragma unroll
      for(int word=0;word<size;word++)
        B_LOW[word]=(B_LOW[word] | mask1) & mask2;

      // Subtract est * d
      mask1=_div_mul_sub(registers, xx, d, B_LOW, digit-d.digits());
      _div_add(registers, xx, d, digit-d.digits(), mask1);
      sub_ui(B_LOW, B_LOW, -mask1);

      xx.store_digit(B_LOW, digit);
    }

    DigitMP<size> lowerXX(xx, 0, d.digits()-divide<size*32>(shifted));

    shift_right<size>(registers, r, lowerXX, remainder<size*32>(shifted));

    return true;
  }

}
