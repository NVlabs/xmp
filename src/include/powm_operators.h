#pragma once

#define AR_GEOMETRY 128
#define AR_DIGIT 8
#define COPY_OUT_GEOMETRY 128
#define SMALL_AR_SIZE 512

typedef void (*powm_kernel)(powm_arguments_t powm_arguments, int32_t start, int32_t count);

template<class T>
inline void determineMaxBlocks(T *kernel, int32_t threads, int32_t *blocks_per_sm) {
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks_per_sm, kernel, threads, 0);
}

inline uint32_t windowBitsForPrecision(uint32_t precision) {
  uint32_t windowBits;

  // these are not tuned
  if(precision<400)
    windowBits=4;
  else if(precision<800)
    windowBits=5;
  else if(precision<1600)
    windowBits=6;
  else if(precision<=4096)
    windowBits=7;
  else
    windowBits=8;

  return windowBits;
}

template<int32_t geometry, int32_t min_blocks, int32_t words, int32_t kar_mult, int32_t kar_sqr>
xmpError_t internalPowmRegMP(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t start, uint32_t count, xmpLimb_t *out_buffer, uint32_t *instances_per_block, uint32_t *blocks_per_sm) {
  // geometry - # of threads per block
  // min_blocks - used for launch bounds
  // words is the size of the modulus in words
  // kar_mult and kar_sqr are the levels of Karatsuba and probably should be 0

  xmpExecutionPolicy_t policy=handle->policy;
  int32_t              bits, windowBits, windowSize;
  size_t               windowBytes;
  int32_t              precision=out->precision;
  ar_arguments_t       ar_arguments;
  powm_arguments_t     powm_arguments;
  copy_out_arguments_t copy_out_arguments;
  xmpError_t           error;

  XMP_SET_DEVICE(handle);

  a->requireFormat(handle, xmpFormatStrided);
  mod->requireFormat(handle, xmpFormatStrided);
  exp->requireFormat(handle, xmpFormatStrided);

  bits=exp->precision;
  windowBits=windowBitsForPrecision(bits);

  while(true) {
    windowSize=((1<<windowBits) + 1 + DIV_ROUND_UP(bits, words*32))*words;
    windowBytes=windowSize * 4;
    windowBytes*=ROUND_UP(count, geometry);

    error=xmpSetNecessaryScratchSize(handle, windowBytes);
    if(error==xmpErrorSuccess)
      break;
    if(error!=xmpErrorIncreaseScratchLimit || windowBits<=2)
      return error;
    // try a smaller window
    windowBits--;
  }

  if(instances_per_block!=NULL)
    *instances_per_block=geometry;

  if(blocks_per_sm!=NULL) {
    determineMaxBlocks(regmp_powm_kernel<geometry, min_blocks, false, words, kar_mult, kar_sqr>, geometry, (int32_t *)blocks_per_sm);
    XMP_CHECK_CUDA();
  }

  if(instances_per_block!=NULL || blocks_per_sm!=NULL)
    return xmpErrorSuccess;

  ar_arguments.window_data=(xmpLimb_t *)handle->scratch;
  ar_arguments.window_bits=windowBits;
  ar_arguments.window_size=windowSize;
  ar_arguments.a_data=a->slimbs;
  ar_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  ar_arguments.a_stride=a->stride;
  ar_arguments.a_count=a->count;
  ar_arguments.exp_data=exp->slimbs;
  ar_arguments.exp_len=DIV_ROUND_UP(exp->precision, 32);
  ar_arguments.exp_stride=exp->stride;
  ar_arguments.exp_count=exp->count;
  ar_arguments.mod_data=mod->slimbs;
  ar_arguments.mod_len=DIV_ROUND_UP(mod->precision, 32);
  ar_arguments.mod_stride=mod->stride;
  ar_arguments.mod_count=mod->count;
  ar_arguments.a_indices=policy->indices[1];
  ar_arguments.exp_indices=policy->indices[2];
  ar_arguments.mod_indices=policy->indices[3];
  ar_arguments.a_indices_count=policy->indices_count[1];
  ar_arguments.exp_indices_count=policy->indices_count[2];
  ar_arguments.mod_indices_count=policy->indices_count[3];

  {
    dim3 blocks(DIV_ROUND_UP(count, AR_GEOMETRY)), threads(AR_GEOMETRY);

    regmp_ar_kernel<words><<<blocks, threads, 0, handle->stream>>>(ar_arguments, start, count);
  }

  powm_arguments.mod_count=0;
  powm_arguments.window_data=(xmpLimb_t *)handle->scratch;
  powm_arguments.bits=exp->precision;
  powm_arguments.window_bits=windowBits;
  powm_arguments.window_size=windowSize;

  {
    dim3 blocks(DIV_ROUND_UP(count, geometry)), threads(geometry);

    regmp_powm_kernel<geometry, min_blocks, false, words, kar_mult, kar_sqr><<<blocks, threads, 0, handle->stream>>>(powm_arguments, start, count);
  }

  copy_out_arguments.out_data=out_buffer;
  copy_out_arguments.out_len=DIV_ROUND_UP(precision, 32);
  copy_out_arguments.out_stride=out->stride;
  copy_out_arguments.out_indices=policy->indices[0];
  copy_out_arguments.window_data=(xmpLimb_t *)handle->scratch;
  copy_out_arguments.window_bits=windowBits;
  copy_out_arguments.window_size=windowSize;

  {
    dim3 blocks(DIV_ROUND_UP(count, COPY_OUT_GEOMETRY)), threads(COPY_OUT_GEOMETRY);

    regmp_copy_out_kernel<words><<<blocks, threads, 0, handle->stream>>>(copy_out_arguments, start, count);
  }

  out->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();

  return xmpErrorSuccess;
}

template<int32_t geometry, int32_t min_blocks, int32_t size>
xmpError_t internalPowmDigitMP(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t start, uint32_t count, xmpLimb_t *out_buffer, uint32_t *instances_per_block, uint32_t *blocks_per_sm) {
  // geometry - # of threads per block
  // min_blocks - used for launch bounds
  // size is the size of the digit in words

  xmpExecutionPolicy_t policy=handle->policy;
  int32_t              bits, roundedPrecision, windowBits, windowSize;
  size_t               windowBytes;
  int32_t              precision=out->precision;
  ar_arguments_t       ar_arguments;
  powm_arguments_t     powm_arguments;
  copy_out_arguments_t copy_out_arguments;
  xmpError_t           error;

  XMP_SET_DEVICE(handle);

  a->requireFormat(handle, xmpFormatStrided);
  mod->requireFormat(handle, xmpFormatStrided);
  exp->requireFormat(handle, xmpFormatStrided);

  bits=exp->precision;
  windowBits=windowBitsForPrecision(bits);

  while(true) {
    roundedPrecision=ROUND_UP(precision, DIGIT*32);
    windowSize=(1<<windowBits) + 4 + DIV_ROUND_UP(bits, roundedPrecision);
    windowBytes=windowSize * roundedPrecision/8;
    windowBytes*=ROUND_UP(count, geometry);

    error=xmpSetNecessaryScratchSize(handle, windowBytes);
    if(error==xmpErrorSuccess)
      break;
    if(error!=xmpErrorIncreaseScratchLimit || windowBits<=2)
      return error;

    // try a smaller window size
    windowBits--;
  }

  if(instances_per_block!=NULL)
    *instances_per_block=geometry;

  if(blocks_per_sm!=NULL) {
    determineMaxBlocks(digitmp_powm_kernel<geometry, min_blocks, false, size>, geometry, (int32_t *)blocks_per_sm);
    XMP_CHECK_CUDA();
  }

  if(instances_per_block!=NULL || blocks_per_sm!=NULL)
    return xmpErrorSuccess;

  ar_arguments.window_data=(xmpLimb_t *)handle->scratch;
  ar_arguments.window_bits=windowBits;
  ar_arguments.window_size=windowSize;
  ar_arguments.a_data=a->slimbs;
  ar_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  ar_arguments.a_stride=a->stride;
  ar_arguments.a_count=a->count;
  ar_arguments.exp_data=exp->slimbs;
  ar_arguments.exp_len=DIV_ROUND_UP(exp->precision, 32);
  ar_arguments.exp_stride=exp->stride;
  ar_arguments.exp_count=exp->count;
  ar_arguments.mod_data=mod->slimbs;
  ar_arguments.mod_len=DIV_ROUND_UP(mod->precision, 32);
  ar_arguments.mod_stride=mod->stride;
  ar_arguments.mod_count=mod->count;
  ar_arguments.a_indices=policy->indices[1];
  ar_arguments.exp_indices=policy->indices[2];
  ar_arguments.mod_indices=policy->indices[3];
  ar_arguments.a_indices_count=policy->indices_count[1];
  ar_arguments.exp_indices_count=policy->indices_count[2];
  ar_arguments.mod_indices_count=policy->indices_count[3];

  {
    dim3 blocks(DIV_ROUND_UP(count, AR_GEOMETRY)), threads(AR_GEOMETRY);

    digitmp_ar_kernel<size><<<blocks, threads, 0, handle->stream>>>(ar_arguments, start, count);
  }

  powm_arguments.mod_count=0;
  powm_arguments.window_data=(xmpLimb_t *)handle->scratch;
  powm_arguments.digits=DIV_ROUND_UP(precision, DIGIT*32);
  powm_arguments.bits=exp->precision;
  powm_arguments.window_bits=windowBits;
  powm_arguments.window_size=windowSize;

  {
    dim3 blocks(DIV_ROUND_UP(count, geometry)), threads(geometry);

    digitmp_powm_kernel<geometry, min_blocks, false, size><<<blocks, threads, 0, handle->stream>>>(powm_arguments, start, count);
  }

  copy_out_arguments.out_data=out_buffer;
  copy_out_arguments.out_len=DIV_ROUND_UP(precision, 32);
  copy_out_arguments.out_stride=out->stride;
  copy_out_arguments.out_indices=policy->indices[0];
  copy_out_arguments.window_data=(xmpLimb_t *)handle->scratch;
  copy_out_arguments.window_bits=windowBits;
  copy_out_arguments.window_size=windowSize;
  copy_out_arguments.digits=DIV_ROUND_UP(precision, DIGIT*32);

  {
    dim3 blocks(DIV_ROUND_UP(count, COPY_OUT_GEOMETRY)), threads(COPY_OUT_GEOMETRY);

    digitmp_copy_out_kernel<size><<<blocks, threads, 0, handle->stream>>>(copy_out_arguments, start, count);
  }

  out->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();

  return xmpErrorSuccess;
}

template<int32_t size, bool small>
class launch_ar {
  public:
  launch_ar(xmpHandle_t handle, ar_arguments_t ar_arguments, uint32_t start, uint32_t count);
};

template<int32_t size>
class launch_ar<size, true> {
  public:
  launch_ar(xmpHandle_t handle, ar_arguments_t ar_arguments, uint32_t start, uint32_t count) {
    dim3 blocks(DIV_ROUND_UP(count, AR_GEOMETRY)), threads(AR_GEOMETRY);

    // size if words*width
    warpmp_small_ar_kernel<size><<<blocks, threads, 0, handle->stream>>>(ar_arguments, start, count);
  }
};

template<int32_t size>
class launch_ar<size, false> {
  public:
  launch_ar(xmpHandle_t handle, ar_arguments_t ar_arguments, uint32_t start, uint32_t count) {
    dim3 blocks(DIV_ROUND_UP(count, AR_GEOMETRY)), threads(AR_GEOMETRY);

    // size is ignored
    warpmp_large_ar_kernel<AR_DIGIT><<<blocks, threads, 0, handle->stream>>>(ar_arguments, start, count);
  }
};

template<int32_t geometry, int32_t min_blocks, int32_t width, int32_t words>
xmpError_t internalPowmWarpDistributedMP(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t start, uint32_t count, xmpLimb_t *out_buffer, uint32_t *instances_per_block, uint32_t *blocks_per_sm) {
  // geometry - # of threads per block
  // min_blocks - used for launch bounds
  // width - threads per instance
  // words - words per thread

  xmpExecutionPolicy_t policy=handle->policy;
  int32_t              bits, windowBits, windowSize;
  size_t               windowBytes, scratchBytes=0;
  int32_t              precision=out->precision;
  ar_arguments_t       ar_arguments;
  powm_arguments_t     powm_arguments;
  copy_out_arguments_t copy_out_arguments;
  xmpError_t           error;

  XMP_SET_DEVICE(handle);

  a->requireFormat(handle, xmpFormatStrided);
  mod->requireFormat(handle, xmpFormatStrided);
  exp->requireFormat(handle, xmpFormatStrided);

  bits=exp->precision;
  windowBits=windowBitsForPrecision(bits);

  if(precision>SMALL_AR_SIZE) {
    scratchBytes=(width*words*5+AR_DIGIT)*4;
    scratchBytes*=ROUND_UP(count, AR_GEOMETRY);
  }
  while(true) {
    windowSize=(1<<windowBits) + 1 + DIV_ROUND_UP(bits, words*width*32);
    windowBytes=windowSize*width*words*4;
    windowBytes*=ROUND_UP(count, geometry);

    error=xmpSetNecessaryScratchSize(handle, windowBytes + scratchBytes);
    if(error==xmpErrorSuccess)
      break;
    if(error!=xmpErrorIncreaseScratchLimit || windowBits<=2)
      return error;

    // try a smaller window size%
    windowBits--;
  }

  if(instances_per_block!=NULL)
    *instances_per_block=geometry/width;

  if(blocks_per_sm!=NULL) {
    determineMaxBlocks(warpmp_powm_kernel<geometry, min_blocks, words>, geometry, (int32_t *)blocks_per_sm);
    XMP_CHECK_CUDA();
  }

  if(instances_per_block!=NULL || blocks_per_sm!=NULL)
    return xmpErrorSuccess;

  // package up the arguments
  ar_arguments.window_data=(xmpLimb_t *)handle->scratch;
  ar_arguments.window_bits=windowBits;
  ar_arguments.window_size=windowSize;
  ar_arguments.a_data=a->slimbs;
  ar_arguments.a_len=DIV_ROUND_UP(a->precision, 32);
  ar_arguments.a_stride=a->stride;
  ar_arguments.a_count=a->count;
  ar_arguments.exp_data=exp->slimbs;
  ar_arguments.exp_len=DIV_ROUND_UP(exp->precision, 32);
  ar_arguments.exp_stride=exp->stride;
  ar_arguments.exp_count=exp->count;
  ar_arguments.mod_data=mod->slimbs;
  ar_arguments.mod_len=DIV_ROUND_UP(mod->precision, 32);
  ar_arguments.mod_stride=mod->stride;
  ar_arguments.mod_count=mod->count;
  ar_arguments.a_indices=policy->indices[1];
  ar_arguments.exp_indices=policy->indices[2];
  ar_arguments.mod_indices=policy->indices[3];
  ar_arguments.a_indices_count=policy->indices_count[1];
  ar_arguments.exp_indices_count=policy->indices_count[2];
  ar_arguments.mod_indices_count=policy->indices_count[3];
  ar_arguments.width=width;
  ar_arguments.precision=words*width; // needed for large ar
  ar_arguments.scratch_data=(xmpLimb_t *)handle->scratch + (windowBytes/4);  // needed for large ar

  if(words*width*32>SMALL_AR_SIZE && (words*width % AR_DIGIT)!=0) {
    // Internal error - the warpmp_large_ar_kernel will fail if word*width is not evenly divisible by the digit size
    return xmpErrorUnsupported;
  }

  launch_ar<words*width, words*width*32<=SMALL_AR_SIZE>(handle, ar_arguments, start, count);

/*
  if(words*width*32<=SMALL_AR_SIZE) {
    dim3 blocks(DIV_ROUND_UP(count, AR_GEOMETRY)), threads(AR_GEOMETRY);

    warpmp_small_ar_kernel<width*words><<<blocks, threads, 0, handle->stream>>>(ar_arguments, start, count);
  }
  else {
    dim3 blocks(DIV_ROUND_UP(count, AR_GEOMETRY)), threads(AR_GEOMETRY);

    if(words*width*32>SMALL_AR_SIZE && (words*width % AR_DIGIT)!=0) {
      // Internal error - the warpmp_large_ar_kernel will fail if word*width is not evenly divisible by the digit size
      return xmpErrorUnsupported;
    }
    warpmp_large_ar_kernel<AR_DIGIT><<<blocks, threads, 0, handle->stream>>>(ar_arguments, start, count);
  }
*/

  powm_arguments.mod_count=0;
  powm_arguments.window_data=(xmpLimb_t *)handle->scratch;
  powm_arguments.width=width;
  powm_arguments.bits=exp->precision;
  powm_arguments.window_bits=windowBits;
  powm_arguments.window_size=windowSize;

  {
    dim3 blocks(DIV_ROUND_UP(count*width, geometry)), threads(geometry);

    warpmp_powm_kernel<geometry, min_blocks, words><<<blocks, threads, 0, handle->stream>>>(powm_arguments, start, count);
  }

  copy_out_arguments.out_data=out_buffer;
  copy_out_arguments.out_len=DIV_ROUND_UP(precision, 32);
  copy_out_arguments.out_stride=out->stride;
  copy_out_arguments.out_indices=policy->indices[0];
  copy_out_arguments.window_data=(xmpLimb_t *)handle->scratch;
  copy_out_arguments.window_bits=windowBits;
  copy_out_arguments.window_size=windowSize;
  copy_out_arguments.width=width;

  {
    dim3 blocks(DIV_ROUND_UP(count, COPY_OUT_GEOMETRY)), threads(COPY_OUT_GEOMETRY);

    warpmp_copy_out_kernel<words><<<blocks, threads, 0, handle->stream>>>(copy_out_arguments, start, count);
  }

  out->setFormat(xmpFormatStrided);

  XMP_CHECK_CUDA();

  return xmpErrorSuccess;
}

xmpError_t XMPAPI xmpIntegersPowmAsync(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t count);

struct Latency {
  uint32_t alg_index;
  float latency;
  uint32_t instances_per_sm;

  Latency(uint32_t alg_index, float latency, uint32_t instances_per_sm) : alg_index(alg_index), latency(latency), instances_per_sm(instances_per_sm) {}
};

typedef xmpError_t (*xmpPowmFunc)(xmpHandle_t handle, xmpIntegers_t out, const xmpIntegers_t a, const xmpIntegers_t exp, const xmpIntegers_t mod, uint32_t start, uint32_t count, xmpLimb_t *out_buffer, uint32_t *instances_per_block, uint32_t *blocks_per_sm);

struct xmpPowmAlgorithm {
  xmpAlgorithm_t alg;
  xmpPowmFunc pfunc;
  uint32_t min_precision;
  uint32_t max_precision;

  xmpPowmAlgorithm(xmpAlgorithm_t alg, xmpPowmFunc pfunc, uint32_t min_precision, uint32_t max_precision) :
    alg(alg), pfunc(pfunc), min_precision(min_precision), max_precision(max_precision) {}
};


extern xmpPowmAlgorithm xmpPowmAlgorithms[];
extern uint32_t xmpPowmAlgorithmsCount;

extern uint32_t xmpPowmPrecisions[];
extern uint32_t xmpPowmPrecisionsCount;


