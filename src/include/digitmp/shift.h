namespace xmp {

  template<uint32_t _size>
  __device__ __forceinline__ void shift_right(uint32_t *scratch, DigitMP<_size> r, DigitMP<_size> x, int32_t bits) {
    RegMP    digit(scratch, 0, 0, _size), accumulator(scratch, 0, _size, 2*_size), lower(scratch, 0, _size, _size);
    uint32_t digitShift=divide<_size*32>(bits), bitShift=bits-digitShift*_size*32;

    if(digitShift<x.digits())
      x.load_digit(digit, digitShift);
    else {
      #pragma unroll
      for(int word=0;word<_size;word++)
        digit[word]=0;
    }

    for(int index=0;index<r.digits();index++) {
      #pragma unroll
      for(int word=0;word<_size;word++)
        accumulator[word]=digit[word];

      if(index+digitShift+1<x.digits())
        x.load_digit(digit, index+digitShift+1);
      else {
        #pragma unroll
        for(int word=0;word<_size;word++)
          digit[word]=0;
      }

      #pragma unroll
      for(int word=0;word<_size;word++)
        accumulator[word+_size]=digit[word];

      shift_right_words(accumulator, accumulator, bitShift>>5);
      shift_right_bits(accumulator, accumulator, bitShift & 0x1F);

      r.store_digit(lower, index);
    }
  }

  template<int32_t size>
  __device__ __forceinline__ void shift_left(uint32_t *registers, DigitMP<size> r, DigitMP<size> x, int32_t bits) {
    RegMP    digit(registers, 0, 0, size), accumulator(registers, 0, size, 2*size), upper(registers, 0, 2*size, size);
    uint32_t digitShift=divide<size*32>(bits), bitShift=bits-digitShift*size*32;

    if(r.digits()>=digitShift+1 && r.digits()-digitShift-1<x.digits())
      x.load_digit(digit, r.digits()-digitShift-1);
    else {
      #pragma unroll
      for(int word=0;word<size;word++)
        digit[word]=0;
    }

    for(int index=r.digits()-1;index>=0;index--) {
      #pragma unroll
      for(int word=0;word<size;word++)
        accumulator[word+size]=digit[word];

      if(index>=digitShift+1 && index-digitShift-1<x.digits())
        x.load_digit(digit, index-digitShift-1);
      else {
        #pragma unroll
        for(int word=0;word<size;word++)
          digit[word]=0;
      }

      #pragma unroll
      for(int word=0;word<size;word++)
        accumulator[word]=digit[word];

      shift_left_words(accumulator, accumulator, bitShift>>5);
      shift_left_bits(accumulator, accumulator, bitShift & 0x1F);

      r.store_digit(upper, index);
    }
  }
}
