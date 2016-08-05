#include <arguments.h>

template<int32_t _size>
__global__ void regmp_ar_kernel(ar_arguments_t ar_arguments, int32_t start, int32_t count) {
  uint32_t   thread;
  xmpLimb_t *window_data=ar_arguments.window_data;
  int32_t    window_bits=ar_arguments.window_bits;
  uint32_t   window_size=ar_arguments.window_size;
  xmpLimb_t *a_data=ar_arguments.a_data;
  int32_t    a_len=ar_arguments.a_len;
  int32_t    a_stride=ar_arguments.a_stride;
  int32_t    a_count=ar_arguments.a_count;
  xmpLimb_t *exp_data=ar_arguments.exp_data;
  int32_t    exp_len=ar_arguments.exp_len;
  int32_t    exp_stride=ar_arguments.exp_stride;
  int32_t    exp_count=ar_arguments.exp_count;
  xmpLimb_t *mod_data=ar_arguments.mod_data;
  int32_t    mod_len=ar_arguments.mod_len;
  int32_t    mod_stride=ar_arguments.mod_stride;
  int32_t    mod_count=ar_arguments.mod_count;
  uint32_t  *a_indices=ar_arguments.a_indices;
  uint32_t  *exp_indices=ar_arguments.exp_indices;
  uint32_t  *mod_indices=ar_arguments.mod_indices;
  uint32_t   a_indices_count=ar_arguments.a_indices_count;
  uint32_t   exp_indices_count=ar_arguments.exp_indices_count;
  uint32_t   mod_indices_count=ar_arguments.mod_indices_count;

  PTXInliner inliner;
  xmpLimb_t  registers[3*_size+2];
  RegMP      MOD(registers, 0, 0, _size), AR(registers, 0, _size, 2*_size+2);
  xmpLimb_t *source, *window;
  uint64_t   window_offset;

  // FIX FIX FIX - check for odd

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int aindex=start + thread, expindex=start + thread, modindex=start + thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=exp_indices) expindex=exp_indices[expindex%exp_indices_count];
    if(NULL!=mod_indices) modindex=mod_indices[modindex%mod_indices_count];

    window_offset=thread & 0x1F;
    thread=thread & ~0x1F;
    inliner.MADWIDE(window_offset, thread, window_size, window_offset);
    window=window_data + window_offset;

    source=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<_size;index++) {
      AR[index]=0;
      if(index<a_len)
        AR[index+_size]=source[index*a_stride];
      else
        AR[index+_size]=0;
    }
    AR[_size+_size]=0;
    AR[_size+_size+1]=0;

    source=exp_data + expindex%exp_count;
    #pragma nounroll
    for(int index=0,word=(1<<window_bits)*_size+_size;word<window_size;word++,index++)
      if(index<exp_len)
        window[word*32]=source[index*exp_stride];
      else
        window[word*32]=0;

    source=mod_data + modindex%mod_count;
    #pragma unroll
    for(int index=0;index<_size;index++) {
      if(index<mod_len)
        MOD[index]=source[index*mod_stride];
      else
        MOD[index]=0;
    }

    #pragma unroll
    for(int index=0;index<_size;index++)
      window[index*32]=MOD[index];

    rem(AR, MOD);

    #pragma unroll
    for(int index=0;index<_size;index++)
      window[index*32+2*_size*32]=AR[index];
  }
}

template<int32_t lb_threads, int32_t lb_blocks, bool use_sm_cache, int32_t _words, int32_t _ks, int32_t _km>
__launch_bounds__(lb_threads, lb_blocks)
__global__ void regmp_powm_kernel(powm_arguments_t powm_arguments, int32_t start, int32_t count) {
  uint32_t    thread;
  int32_t     mod_count=powm_arguments.mod_count;
  xmpLimb_t  *window_data=powm_arguments.window_data;
  int32_t     bits=powm_arguments.bits;
  int32_t     window_bits=powm_arguments.window_bits;
  uint32_t    window_size=powm_arguments.window_size;

  PTXInliner  inliner;
  xmpLimb_t  *w;
  uint64_t    window_offset;

  // use_sm_cache is not currently supported

  // exp_len is passed in the bits parameters, mod_len is passed in _words parameter

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    window_offset=thread & 0x1F;
    thread=thread & ~0x1F;
    inliner.MADWIDE(window_offset, thread, window_size, window_offset);
    w=window_data + window_offset;

    typedef ThreeN_n<_words, _ks, _km> ThreeNModel;

    fwe<ThreeNModel, true>(mod_count,
                           w,
                           _words, 0, bits, window_bits);
  }
}

template<int _words>
__global__ void regmp_copy_out_kernel(copy_out_arguments_t copy_out_arguments, int32_t start, int32_t count) {
  uint32_t    thread;
  xmpLimb_t  *out_data=copy_out_arguments.out_data;
  int32_t     out_len=copy_out_arguments.out_len;
  int32_t     out_stride=copy_out_arguments.out_stride;
  uint32_t   *out_indices=copy_out_arguments.out_indices;
  xmpLimb_t  *window_data=copy_out_arguments.window_data;
  uint32_t    window_size=copy_out_arguments.window_size;

  PTXInliner  inliner;
  xmpLimb_t  *o, *w;
  uint64_t    window_offset;

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int32_t outindex=start + thread;
    if(NULL!=out_indices) outindex=out_indices[outindex];

    window_offset=thread & 0x1F;
    thread=thread & ~0x1F;
    inliner.MADWIDE(window_offset, thread, window_size, window_offset);

    o=out_data + outindex;
    w=window_data + window_offset;

    #pragma nounroll
    for(int index=0;index<out_len;index++)
      o[index*out_stride]=w[_words*32 + index*32];
  }
}

template<int size>
__global__ void digitmp_ar_kernel(ar_arguments_t ar_arguments, int32_t start, int32_t count) {
  uint32_t   thread;
  xmpLimb_t *window_data=ar_arguments.window_data;
  int32_t    window_bits=ar_arguments.window_bits;
  uint32_t   window_size=ar_arguments.window_size;
  xmpLimb_t *a_data=ar_arguments.a_data;
  int32_t    a_len=ar_arguments.a_len;
  int32_t    a_stride=ar_arguments.a_stride;
  int32_t    a_count=ar_arguments.a_count;
  xmpLimb_t *exp_data=ar_arguments.exp_data;
  int32_t    exp_len=ar_arguments.exp_len;
  int32_t    exp_stride=ar_arguments.exp_stride;
  int32_t    exp_count=ar_arguments.exp_count;
  xmpLimb_t *mod_data=ar_arguments.mod_data;
  int32_t    mod_len=ar_arguments.mod_len;
  int32_t    mod_stride=ar_arguments.mod_stride;
  int32_t    mod_count=ar_arguments.mod_count;
  uint32_t  *a_indices=ar_arguments.a_indices;
  uint32_t  *exp_indices=ar_arguments.exp_indices;
  uint32_t  *mod_indices=ar_arguments.mod_indices;
  uint32_t   a_indices_count=ar_arguments.a_indices_count;
  uint32_t   exp_indices_count=ar_arguments.exp_indices_count;
  uint32_t   mod_indices_count=ar_arguments.mod_indices_count;

  xmpLimb_t  registers[4*size+4];
  RegMP      ZERO(registers, 0, 0, size);
  int32_t    digits=divide<size>(a_len+size-1);

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int aindex=start + thread, expindex=start + thread, modindex=start + thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=exp_indices) expindex=exp_indices[expindex%exp_indices_count];
    if(NULL!=mod_indices) modindex=mod_indices[modindex%mod_indices_count];

    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex);
    DigitMP<size> EXP(false, false, exp_data, exp_len, exp_stride, exp_count, expindex);
    DigitMP<size> MOD(false, false, mod_data, mod_len, mod_stride, mod_count, modindex);

    DigitMP<size> WINDOW(false, false, window_data, window_size*digits, thread);
    DigitMP<size> N(WINDOW, 0, digits), REM(WINDOW, 5*digits, digits), E(WINDOW, ((1<<window_bits)+4)*digits, (window_size-(1<<window_bits)-4)*digits);
    DigitMP<size> AR(WINDOW, digits, 2*digits+1), AR_LOW(WINDOW, digits, digits), AR_HIGH(WINDOW, 2*digits, digits);
    DigitMP<size> TEMP(WINDOW, 4*digits, digits), INVERSE(WINDOW, 6*digits, 1);

    set<size>(registers, N, MOD);
    set<size>(registers, E, EXP);

    set_ui<size>(registers, AR_LOW, 0);
    set<size>(registers, AR_HIGH, A);

    set_ui(ZERO, 0);
    AR.store_digit(ZERO, 2*digits);

    _rem(registers, REM, AR, N, TEMP, INVERSE);
  }
}

// #if __CUDA_ARCH__ >= 200 && __CUDA_ARCH__ < 350
// //Fermi and small Kepler (GK100),  already restricted to 63 registers
// #elif __CUDA_ARCH__ >= 350 && __CUDA_ARCH__ < 370
// //GK110,  allows up to 255 registers.  Restrict down to avoid occupancy issues.
// __launch_bounds__(128,8)
// #elif __CUDA_ARCH__ == 370
// //GK210,  2x registers.  Go for maximum occupancy
// __launch_bounds__(128,16)
// #elif __CUDA_ARCH__ < 600 && __CUDA_ARCH__>= 500  //Maxwell
// //Maxwell
// __launch_bounds__(128,6)
// #endif

template<int32_t lb_threads, int32_t lb_blocks, bool use_sm_cache, int size>
__launch_bounds__(lb_threads, lb_blocks)
__global__ void digitmp_powm_kernel(powm_arguments_t powm_arguments, int32_t start, int32_t count) {
  uint32_t   thread;
  int32_t    mod_count=powm_arguments.mod_count;
  xmpLimb_t *window_data=powm_arguments.window_data;
  uint32_t   window_size=powm_arguments.window_size;
  int32_t    digits=powm_arguments.digits;
  int32_t    bits=powm_arguments.bits;
  int32_t    window_bits=powm_arguments.window_bits;

  // exp_len is passed in the bits parameters, mod_len is passed in digits parameter

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    DigitMP<size> WINDOW(false, false, window_data, window_size*digits, thread);

    typedef Digitized<use_sm_cache, size> DigitizedModel;

    fwe<DigitizedModel, false>(mod_count,
                               WINDOW._base,
                               digits, 0, bits, window_bits);

    // needed for GSL because threads are reused
    //    window_data=window_data + blockDim.x*gridDim.x*((1<<window_bits)+4)*digits*size;
  }
}

template<int size>
__global__ void digitmp_copy_out_kernel(copy_out_arguments_t copy_out_arguments, int32_t start, int32_t count) {
  uint32_t   thread;
  xmpLimb_t *out_data=copy_out_arguments.out_data;
  int32_t    out_len=copy_out_arguments.out_len;
  int32_t    out_stride=copy_out_arguments.out_stride;
  uint32_t  *out_indices=copy_out_arguments.out_indices;
  xmpLimb_t *window_data=copy_out_arguments.window_data;
  uint32_t   window_size=copy_out_arguments.window_size;
  int32_t    digits=copy_out_arguments.digits;

  xmpLimb_t  registers[size];
  RegMP      LOAD(registers, 0, 0, size);
  xmpLimb_t *o;

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int32_t outindex=start + thread;
    if(NULL!=out_indices) outindex=out_indices[outindex];

    DigitMP<size> WINDOW(false, false, window_data, window_size*digits, thread), OUT(WINDOW, digits, digits);
    o=out_data + outindex;

    #pragma nounroll
    for(int digit=0;digit<digits;digit++) {
      OUT.load_digit(LOAD, digit);
      #pragma unroll
      for(int word=0;word<size;word++) {
        if(digit*size+word<out_len)
          o[(digit*size+word)*out_stride]=registers[word];
      }
    }
  }
}


template<int32_t size>
__global__ void warpmp_small_ar_kernel(ar_arguments_t ar_arguments, int32_t start, int32_t count) {
#if __CUDA_ARCH__>=300
  uint32_t    thread;
  xmpLimb_t  *window_data=ar_arguments.window_data;
  int32_t     window_bits=ar_arguments.window_bits;
  uint32_t    window_size=ar_arguments.window_size;
  xmpLimb_t  *a_data=ar_arguments.a_data;
  int32_t     a_len=ar_arguments.a_len;
  int32_t     a_stride=ar_arguments.a_stride;
  int32_t     a_count=ar_arguments.a_count;
  xmpLimb_t  *exp_data=ar_arguments.exp_data;
  int32_t     exp_len=ar_arguments.exp_len;
  int32_t     exp_stride=ar_arguments.exp_stride;
  int32_t     exp_count=ar_arguments.exp_count;
  xmpLimb_t  *mod_data=ar_arguments.mod_data;
  int32_t     mod_len=ar_arguments.mod_len;
  int32_t     mod_stride=ar_arguments.mod_stride;
  int32_t     mod_count=ar_arguments.mod_count;
  int32_t     width=ar_arguments.width;
  uint32_t   *a_indices=ar_arguments.a_indices;
  uint32_t   *exp_indices=ar_arguments.exp_indices;
  uint32_t   *mod_indices=ar_arguments.mod_indices;
  uint32_t    a_indices_count=ar_arguments.a_indices_count;
  uint32_t    exp_indices_count=ar_arguments.exp_indices_count;
  uint32_t    mod_indices_count=ar_arguments.mod_indices_count;

  PTXInliner  inliner;
  xmpLimb_t   registers[3*size+2];
  RegMP       MOD(registers, 0, 0, size), AR(registers, 0, size, 2*size+2);
  xmpLimb_t  *source, *window;
  int32_t     words=size/width;
  uint64_t    window_offset;

  // FIX FIX FIX - check for odd

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int aindex=start + thread, expindex=start + thread, modindex=start + thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=exp_indices) expindex=exp_indices[expindex%exp_indices_count];
    if(NULL!=mod_indices) modindex=mod_indices[modindex%mod_indices_count];

    source=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<size;index++) {
      AR[index]=0;
      if(index<a_len)
        AR[index+size]=source[index*a_stride];
      else
        AR[index+size]=0;
    }
    AR[size+size]=0;
    AR[size+size+1]=0;

    thread=thread*width;
    window_offset=thread & 0x1F;
    thread=thread & ~0x1F;
    window_size=window_size*words;
    inliner.MADWIDE(window_offset, thread, window_size, window_offset);
    window=window_data + window_offset;

    source=exp_data + expindex%exp_count;
    #pragma nounroll
    for(int index=0;index<exp_len;index++)
      window[(1<<window_bits)*words*32 + words*32 + index/width*32 + index%width]=source[index*exp_stride];

    source=mod_data + modindex%mod_count;
    #pragma unroll
    for(int index=0;index<size;index++) {
      if(index<mod_len)
        MOD[index]=source[index*mod_stride];
      else
        MOD[index]=0;
    }

    #pragma unroll
    for(int index=0;index<size;index++)
      window[(1<<window_bits)*words*32 + index/words + index%words*32]=MOD[index];

    rem(AR, MOD);

    #pragma unroll
    for(int index=0;index<size;index++)
      window[words*32 + index/words + index%words*32]=AR[index];
  }
#endif
}

// size here is the digit size
template<int32_t size>
__global__ void warpmp_large_ar_kernel(ar_arguments_t ar_arguments, int32_t start, int32_t count) {
#if __CUDA_ARCH__>=300
  uint32_t   thread;
  uint32_t   precision=ar_arguments.precision;
  xmpLimb_t *scratch_data=ar_arguments.scratch_data;
  xmpLimb_t *window_data=ar_arguments.window_data;
  int32_t    window_bits=ar_arguments.window_bits;
  uint32_t   window_size=ar_arguments.window_size;
  xmpLimb_t *a_data=ar_arguments.a_data;
  int32_t    a_len=ar_arguments.a_len;
  int32_t    a_stride=ar_arguments.a_stride;
  int32_t    a_count=ar_arguments.a_count;
  xmpLimb_t *exp_data=ar_arguments.exp_data;
  int32_t    exp_len=ar_arguments.exp_len;
  int32_t    exp_stride=ar_arguments.exp_stride;
  int32_t    exp_count=ar_arguments.exp_count;
  xmpLimb_t *mod_data=ar_arguments.mod_data;
  int32_t    mod_len=ar_arguments.mod_len;
  int32_t    mod_stride=ar_arguments.mod_stride;
  int32_t    mod_count=ar_arguments.mod_count;
  int32_t    width=ar_arguments.width;
  uint32_t  *a_indices=ar_arguments.a_indices;
  uint32_t  *exp_indices=ar_arguments.exp_indices;
  uint32_t  *mod_indices=ar_arguments.mod_indices;
  uint32_t   a_indices_count=ar_arguments.a_indices_count;
  uint32_t   exp_indices_count=ar_arguments.exp_indices_count;
  uint32_t   mod_indices_count=ar_arguments.mod_indices_count;

  xmpLimb_t  registers[4*size+4];
  RegMP      ZERO(registers, 0, 0, size);
  int32_t    digits=divide<size>(precision);
  int32_t    words=precision/width;

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int aindex=start + thread, expindex=start + thread, modindex=start + thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=exp_indices) expindex=exp_indices[expindex%exp_indices_count];
    if(NULL!=mod_indices) modindex=mod_indices[modindex%mod_indices_count];

    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex);
    DigitMP<size> EXP(false, false, exp_data, exp_len, exp_stride, exp_count, expindex);
    DigitMP<size> MOD(false, false, mod_data, mod_len, mod_stride, mod_count, modindex);

    DigitMP<size> SCRATCH(false, false, scratch_data, 5*digits+1, thread);
    DigitMP<size> N(SCRATCH, 0, digits), TEMP(SCRATCH, digits, digits), INVERSE(SCRATCH, 2*digits, digits);
    DigitMP<size> AR(SCRATCH, 3*digits, 2*digits+1), AR_LOW(SCRATCH, 3*digits, digits), AR_HIGH(SCRATCH, 4*digits, digits);
    DigitMP<size> WINDOW_N(window_data, width, words, 1<<window_bits, 1, window_size, false, thread);
    DigitMP<size> WINDOW_REM(window_data, width, words, 1, 1, window_size, false, thread);
    DigitMP<size> WINDOW_EXP(window_data, width, words, (1<<window_bits)+1, window_size-(1<<window_bits)-1, window_size, true, thread);

    set<size>(registers, N, MOD);
    set<size>(registers, WINDOW_N, N);
    set<size>(registers, WINDOW_EXP, EXP);

    set_ui<size>(registers, AR_LOW, 0);
    set<size>(registers, AR_HIGH, A);
    set_ui(ZERO, 0);
    AR.store_digit(ZERO, 2*digits);

    _rem(registers, WINDOW_REM, AR, N, TEMP, INVERSE);
  }
#endif
}

template<int32_t lb_threads, int32_t lb_blocks, int32_t _words>
__launch_bounds__(lb_threads, lb_blocks)
__global__ void warpmp_powm_kernel(powm_arguments_t powm_arguments, int32_t start, int32_t count) {
#if __CUDA_ARCH__>=300
  uint32_t    thread;
  int32_t     mod_count=powm_arguments.mod_count;
  xmpLimb_t  *window_data=powm_arguments.window_data;
  int32_t     width=powm_arguments.width;
  int32_t     bits=powm_arguments.bits;
  int32_t     window_bits=powm_arguments.window_bits;
  uint32_t    window_size=powm_arguments.window_size;

  PTXInliner  inliner;
  xmpLimb_t  *w;
  uint64_t    window_offset;

  // exp_len is passed in the bits parameters, mod_len is passed in _words parameter

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count*width) {
    window_offset=thread & 0x1F;
    thread=thread & ~0x1F;
    window_size=window_size*_words;
    inliner.MADWIDE(window_offset, thread, window_size, window_offset);
    w=window_data + window_offset;

    typedef Warp_Distributed<_words> WarpDistributedModel;

    fwe<WarpDistributedModel, true>(mod_count,
                                    w,
                                    _words, width, bits, window_bits);
  }
#endif
}

template<int words>
__global__ void warpmp_copy_out_kernel(copy_out_arguments_t copy_out_arguments, int32_t start, int32_t count) {
#if __CUDA_ARCH__>=300
  uint32_t    thread;
  xmpLimb_t  *out_data=copy_out_arguments.out_data;
  int32_t     out_len=copy_out_arguments.out_len;
  int32_t     out_stride=copy_out_arguments.out_stride;
  uint32_t   *out_indices=copy_out_arguments.out_indices;
  xmpLimb_t  *window_data=copy_out_arguments.window_data;
  uint32_t    window_size=copy_out_arguments.window_size;
  int32_t     width=copy_out_arguments.width;

  PTXInliner  inliner;
  xmpLimb_t  *o, *w;
  uint64_t    window_offset;

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int32_t outindex=start + thread;
    if(NULL!=out_indices) outindex=out_indices[outindex];

    o=out_data + outindex;

    thread=thread*width;
    window_offset=thread & 0x1F;
    thread=thread & ~0x1F;
    window_size=window_size*words;
    inliner.MADWIDE(window_offset, thread, window_size, window_offset);
    w=window_data + window_offset;

    #pragma nounroll
    for(int word=0;word<out_len;word++) {
      o[word*out_stride]=w[word/words + word%words*32];
    }
  }
#endif
}
