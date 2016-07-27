template<int32_t _size>
__global__ void regmp_ar_kernel(ar_arguments_t ar_arguments, int32_t start, int32_t count) {
  int32_t    thread;
  xmpLimb_t *window_data=ar_arguments.window_data;
  int32_t    window_bits=ar_arguments.window_bits;
  xmpLimb_t *a_data=ar_arguments.a_data;
  int32_t    a_len=ar_arguments.a_len;
  int32_t    a_stride=ar_arguments.a_stride;
  int32_t    a_count=ar_arguments.a_count;
  xmpLimb_t *mod_data=ar_arguments.mod_data;
  int32_t    mod_len=ar_arguments.mod_len;
  int32_t    mod_stride=ar_arguments.mod_stride;
  int32_t    mod_count=ar_arguments.mod_count;
  uint32_t  *a_indices=ar_arguments.a_indices;
  uint32_t  *mod_indices=ar_arguments.mod_indices;
  uint32_t   a_indices_count=ar_arguments.a_indices_count;
  uint32_t   mod_indices_count=ar_arguments.mod_indices_count;

  PTXInliner inliner;
  xmpLimb_t  registers[3*_size+2];
  RegMP      MOD(registers, 0, 0, _size), AR(registers, 0, _size, 2*_size+2);
  xmpLimb_t *data;

  // FIX FIX FIX - check for odd

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int aindex=start + thread, modindex=start + thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=mod_indices) modindex=mod_indices[modindex%mod_indices_count];

    data=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<_size;index++) {
      AR[index]=0;
      if(index<a_len)
        AR[index+_size]=data[index*a_stride];
      else
        AR[index+_size]=0;
    }
    AR[_size+_size]=0;
    AR[_size+_size+1]=0;

    data=mod_data + modindex%mod_count;
    #pragma unroll
    for(int index=0;index<_size;index++) {
      if(index<mod_len)
        MOD[index]=data[index*mod_stride];
      else
        MOD[index]=0;
    }

    data=window_data + (thread & ~0x1F)*((1<<window_bits)+4)*_size + (thread & 0x1F);
    #pragma unroll
    for(int index=0;index<_size;index++)
      data[index*32]=MOD[index];

    rem(AR, MOD);

    #pragma unroll
    for(int index=0;index<_size;index++)
      data[index*32+5*_size*32]=AR[index];
  }
}

template<int32_t lb_threads, int32_t lb_blocks, bool use_sm_cache, int32_t _words, int32_t _ks, int32_t _km>
__launch_bounds__(lb_threads, lb_blocks)
__global__ void regmp_powm_kernel(powm_arguments_t powm_arguments, int32_t start, int32_t count) {
  int32_t    thread;
  xmpLimb_t *out_data=powm_arguments.out_data;
  int32_t    out_len=powm_arguments.out_len;
  int32_t    out_stride=powm_arguments.out_stride;
  xmpLimb_t *exp_data=powm_arguments.exp_data;
  int32_t    exp_stride=powm_arguments.exp_stride;
  int32_t    exp_count=powm_arguments.exp_count;
  int32_t    mod_count=powm_arguments.mod_count;
  xmpLimb_t *window_data=powm_arguments.window_data;
  int32_t    bits=powm_arguments.bits;
  int32_t    window_bits=powm_arguments.window_bits;
  uint32_t   *exp_indices=powm_arguments.exp_indices;
  uint32_t   *out_indices=powm_arguments.out_indices;
  uint32_t    exp_indices_count=powm_arguments.exp_indices_count;

  xmpLimb_t *o, *e, *w;

  // use_sm_cache is not currently supported

  // exp_len is passed in the bits parameters, mod_len is passed in _words parameter

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int expindex=start + thread, outindex=start + thread;
    if(NULL!=exp_indices) expindex=exp_indices[expindex%exp_indices_count];
    if(NULL!=out_indices) outindex=out_indices[outindex];

    o=out_data + outindex;
    e=exp_data + expindex%exp_count;
    w=window_data + (thread & ~0x1F)*((1<<window_bits)+4)*_words + (thread & 0x1F);

    typedef ThreeN_n<_words, _ks, _km> ThreeNModel;

    fwe<ThreeNModel>(o, out_len, out_stride,
                     e, exp_stride,
                     mod_count,
                     w,
                     _words, 0, bits, window_bits);
  }
}

template<int size>
__global__ void digitmp_ar_kernel(ar_arguments_t ar_arguments, int32_t start, int32_t count) {
  int32_t    thread;
  xmpLimb_t *window_data=ar_arguments.window_data;
  int32_t    window_bits=ar_arguments.window_bits;
  xmpLimb_t *a_data=ar_arguments.a_data;
  int32_t    a_len=ar_arguments.a_len;
  int32_t    a_stride=ar_arguments.a_stride;
  int32_t    a_count=ar_arguments.a_count;
  xmpLimb_t *mod_data=ar_arguments.mod_data;
  int32_t    mod_len=ar_arguments.mod_len;
  int32_t    mod_stride=ar_arguments.mod_stride;
  int32_t    mod_count=ar_arguments.mod_count;
  uint32_t   *a_indices=ar_arguments.a_indices;
  uint32_t   *mod_indices=ar_arguments.mod_indices;
  uint32_t    a_indices_count=ar_arguments.a_indices_count;
  uint32_t    mod_indices_count=ar_arguments.mod_indices_count;

  xmpLimb_t  registers[4*size+4];
  RegMP      ZERO(registers, 0, 0, size);
  int32_t    digits=divide<size>(a_len+size-1);

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int aindex=start + thread, modindex=start + thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=mod_indices) modindex=mod_indices[modindex%mod_indices_count];

    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex), MOD(false, false, mod_data, mod_len, mod_stride, mod_count, modindex);
    DigitMP<size> WINDOW(false, false, window_data, (4+(1<<window_bits))*digits, thread);
    DigitMP<size> N(WINDOW, 0, digits), REM(WINDOW, 5*digits, digits);
    DigitMP<size> AR(WINDOW, digits, 2*digits+1), AR_LOW(WINDOW, digits, digits), AR_HIGH(WINDOW, 2*digits, digits);
    DigitMP<size> TEMP(WINDOW, 4*digits, digits), INVERSE(WINDOW, 6*digits, 1);

    set<size>(registers, N, MOD);

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
  int32_t    thread;
  xmpLimb_t *out_data=powm_arguments.out_data;
  int32_t    out_len=powm_arguments.out_len;
  int32_t    out_stride=powm_arguments.out_stride;
  xmpLimb_t *exp_data=powm_arguments.exp_data;
  int32_t    exp_stride=powm_arguments.exp_stride;
  int32_t    exp_count=powm_arguments.exp_count;
  int32_t    mod_count=powm_arguments.mod_count;
  xmpLimb_t *window_data=powm_arguments.window_data;
  int32_t    digits=powm_arguments.digits;
  int32_t    bits=powm_arguments.bits;
  int32_t    window_bits=powm_arguments.window_bits;
  uint32_t   *exp_indices=powm_arguments.exp_indices;
  uint32_t   *out_indices=powm_arguments.out_indices;
  uint32_t    exp_indices_count=powm_arguments.exp_indices_count;

  xmpLimb_t *e, *o;

  // exp_len is passed in the bits parameters, mod_len is passed in digits parameter

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int expindex=start + thread, outindex=start + thread;
    if(NULL!=exp_indices) expindex=exp_indices[expindex%exp_indices_count];
    if(NULL!=out_indices) outindex=out_indices[outindex];

    e=exp_data + expindex%exp_count;
    o=out_data + outindex;

    typedef Digitized<use_sm_cache, size> DigitizedModel;

    fwe<DigitizedModel>(o, out_len, out_stride,
                        e, exp_stride,
                        mod_count,
                        window_data,
                        digits, 0, bits, window_bits);

    // needed for GSL because threads are reused
    //    window_data=window_data + blockDim.x*gridDim.x*((1<<window_bits)+4)*digits*size;
  }
}

template<int32_t size>
__global__ void warpmp_small_ar_kernel(ar_arguments_t ar_arguments, int32_t start, int32_t count) {
#if __CUDA_ARCH__>=300
  int32_t    thread;
  xmpLimb_t *window_data=ar_arguments.window_data;
  int32_t    window_bits=ar_arguments.window_bits;
  xmpLimb_t *a_data=ar_arguments.a_data;
  int32_t    a_len=ar_arguments.a_len;
  int32_t    a_stride=ar_arguments.a_stride;
  int32_t    a_count=ar_arguments.a_count;
  xmpLimb_t *mod_data=ar_arguments.mod_data;
  int32_t    mod_len=ar_arguments.mod_len;
  int32_t    mod_stride=ar_arguments.mod_stride;
  int32_t    mod_count=ar_arguments.mod_count;
  int32_t    width=ar_arguments.width;
  uint32_t  *a_indices=ar_arguments.a_indices;
  uint32_t  *mod_indices=ar_arguments.mod_indices;
  uint32_t   a_indices_count=ar_arguments.a_indices_count;
  uint32_t   mod_indices_count=ar_arguments.mod_indices_count;

  PTXInliner inliner;
  xmpLimb_t  registers[3*size+2];
  RegMP      MOD(registers, 0, 0, size), AR(registers, 0, size, 2*size+2);
  xmpLimb_t *data;
  int32_t    words=size/width;

  // FIX FIX FIX - check for odd

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int aindex=start + thread, modindex=start + thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=mod_indices) modindex=mod_indices[modindex%mod_indices_count];

    data=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<size;index++) {
      AR[index]=0;
      if(index<a_len)
        AR[index+size]=data[index*a_stride];
      else
        AR[index+size]=0;
    }
    AR[size+size]=0;
    AR[size+size+1]=0;

    data=mod_data + modindex%mod_count;
    #pragma unroll
    for(int index=0;index<size;index++) {
      if(index<mod_len)
        MOD[index]=data[index*mod_stride];
      else
        MOD[index]=0;
    }

    data=window_data + (thread*width & ~0x1F)*((1<<window_bits)+1)*words + (thread*width & 0x1F);
    #pragma unroll
    for(int index=0;index<size;index++)
      data[(1<<window_bits)*words*32 + index/words + index%words*32]=MOD[index];

    rem(AR, MOD);

    #pragma unroll
    for(int index=0;index<size;index++)
      data[words*32 + index/words + index%words*32]=AR[index];
  }
#endif
}

// size here is the digit size
template<int32_t size>
__global__ void warpmp_large_ar_kernel(ar_arguments_t ar_arguments, int32_t start, int32_t count) {
  int32_t    thread;
  uint32_t   precision=ar_arguments.precision;
  xmpLimb_t *scratch_data=ar_arguments.scratch_data;
  xmpLimb_t *window_data=ar_arguments.window_data;
  int32_t    window_bits=ar_arguments.window_bits;
  xmpLimb_t *a_data=ar_arguments.a_data;
  int32_t    a_len=ar_arguments.a_len;
  int32_t    a_stride=ar_arguments.a_stride;
  int32_t    a_count=ar_arguments.a_count;
  xmpLimb_t *mod_data=ar_arguments.mod_data;
  int32_t    mod_len=ar_arguments.mod_len;
  int32_t    mod_stride=ar_arguments.mod_stride;
  int32_t    mod_count=ar_arguments.mod_count;
  int32_t    width=ar_arguments.width;
  uint32_t  *a_indices=ar_arguments.a_indices;
  uint32_t  *mod_indices=ar_arguments.mod_indices;
  uint32_t   a_indices_count=ar_arguments.a_indices_count;
  uint32_t   mod_indices_count=ar_arguments.mod_indices_count;

  xmpLimb_t  registers[4*size+4];
  RegMP      ZERO(registers, 0, 0, size);
  int32_t    digits=divide<size>(precision);
  int32_t    words=precision/width;

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count) {
    int aindex=start + thread, modindex=start + thread;
    if(NULL!=a_indices) aindex=a_indices[aindex%a_indices_count];
    if(NULL!=mod_indices) modindex=mod_indices[modindex%mod_indices_count];

    DigitMP<size> A(false, false, a_data, a_len, a_stride, a_count, aindex), MOD(false, false, mod_data, mod_len, mod_stride, mod_count, modindex);
    DigitMP<size> SCRATCH(false, false, scratch_data, 5*digits+1, thread);
    DigitMP<size> N(SCRATCH, 0, digits), TEMP(SCRATCH, digits, digits), INVERSE(SCRATCH, 2*digits, digits);
    DigitMP<size> AR(SCRATCH, 3*digits, 2*digits+1), AR_LOW(SCRATCH, 3*digits, digits), AR_HIGH(SCRATCH, 4*digits, digits);
    DigitMP<size> WINDOW_N(window_data, width, words, 1<<window_bits, (1<<window_bits)+1, thread);
    DigitMP<size> WINDOW_REM(window_data, width, words, 1, (1<<window_bits)+1, thread);

    set<size>(registers, N, MOD);
    set<size>(registers, WINDOW_N, N);

    set_ui<size>(registers, AR_LOW, 0);
    set<size>(registers, AR_HIGH, A);
    set_ui(ZERO, 0);
    AR.store_digit(ZERO, 2*digits);

    _rem(registers, WINDOW_REM, AR, N, TEMP, INVERSE);
  }
}

template<int32_t lb_threads, int32_t lb_blocks, int32_t _words>
__launch_bounds__(lb_threads, lb_blocks)
__global__ void warpmp_powm_kernel(powm_arguments_t powm_arguments, int32_t start, int32_t count) {
#if __CUDA_ARCH__>=300
  int32_t    thread;
  xmpLimb_t *out_data=powm_arguments.out_data;
  int32_t    out_len=powm_arguments.out_len;
  int32_t    out_stride=powm_arguments.out_stride;
  xmpLimb_t *exp_data=powm_arguments.exp_data;
  int32_t    exp_stride=powm_arguments.exp_stride;
  int32_t    exp_count=powm_arguments.exp_count;
  int32_t    mod_count=powm_arguments.mod_count;
  xmpLimb_t *window_data=powm_arguments.window_data;
  int32_t    width=powm_arguments.width;
  int32_t    bits=powm_arguments.bits;
  int32_t    window_bits=powm_arguments.window_bits;
  uint32_t   *exp_indices=powm_arguments.exp_indices;
  uint32_t   *out_indices=powm_arguments.out_indices;
  uint32_t    exp_indices_count=powm_arguments.exp_indices_count;

  xmpLimb_t *o, *e, *w;

  // exp_len is passed in the bits parameters, mod_len is passed in _words parameter

  thread=blockIdx.x*blockDim.x + threadIdx.x;
  if(thread<count*width) {
    int expindex=start + (thread/width), outindex=start + (thread/width);
    if(NULL!=exp_indices) expindex=exp_indices[expindex%exp_indices_count];
    if(NULL!=out_indices) outindex=out_indices[outindex];

    // o=out_data + outindex*out_len;
    o=out_data + outindex;
    e=exp_data + expindex%exp_count;
    w=window_data + (thread & ~0x1F)*((1<<window_bits)+1)*_words + (thread & 0x1F);

    typedef Warp_Distributed<_words> WarpDistributedModel;

    fwe<WarpDistributedModel>(o, out_len, out_stride,
                              e, exp_stride,
                              mod_count,
                              w,
                              _words, width, bits, window_bits);
  }
#endif
}
