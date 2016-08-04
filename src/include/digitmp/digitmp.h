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

#if __CUDA_ARCH__<350
  // __ldg replacement for less than sm_35
  __device__ __forceinline__ uint32_t __ldg(uint32_t *ptr) {
    return *ptr;
  }

  __device__ __forceinline__ uint2 __ldg(uint2 *ptr) {
    return *ptr;
  }

  __device__ __forceinline__ uint4 __ldg(uint4 *ptr) {
    return *ptr;
  }
#endif

  // internal means that you don't have to do bounds checking, and there are no truncated digits
  // tex means use texture cache (__ldg) for loading
  // v1 means use 32 bit loads, v4 means use 128 bits loads

  // more of these can be added as needed

  typedef enum {
    xmpDigitStorage_compact_v1,
    xmpDigitStorage_compact_tex_v1,

    xmpDigitStorage_compact_internal_v1,
    xmpDigitStorage_compact_internal_v4,
    xmpDigitStorage_compact_internal_tex_v1,
    xmpDigitStorage_compact_internal_tex_v4,

    xmpDigitStorage_strided_v1,

    xmpDigitStorage_warp_strided_internal_v1,
    xmpDigitStorage_warp_strided_internal_v4,

    xmpDigitStorage_shared_internal_v1,
    xmpDigitStorage_shared_internal_v2,

    xmpDigitStorage_shared_chunked_internal_v2,    // chunked only works if _size is divisible by 8

    xmpDigitStorage_warp_distributed_v1,
    xmpDigitStorage_warp_distributed_v2,
    xmpDigitStorage_warp_distributed_v4,

    xmpDigitStorage_warp_distributed_transpose_v1,
  } xmpDigitStorage_t;

  template<uint32_t denominator>
  __device__ __forceinline__ uint32_t divide(uint32_t numerator) {
    uint32_t est=0xFFFFFFFF/denominator;

    // not exact, but ok for den<1024 and num<2^20
    return __umulhi((uint32_t)est, numerator+1);
  }

  template<uint32_t denominator>
  __device__ __forceinline__ uint32_t remainder(uint32_t numerator) {

    // not exact, but ok for den<1024 and num<2^20
    return numerator-divide<denominator>(numerator)*denominator;
  }

  template<uint32_t denominator>
  __device__ __forceinline__ uint32_t roundup(uint32_t numerator) {

    // not exact, but ok for den<1024 and num<2^20
    return divide<denominator>(numerator+denominator-1)*denominator;
  }

  template<uint32_t _size>
  class DigitMP {
    public:
      xmpDigitStorage_t  _storage;
      uint32_t          *_base;
      int32_t            _start;
      int32_t            _digits;
      int32_t            _length;
      uint32_t           _stride;
      int32_t            _width;
      int32_t            _words;

      __device__ __forceinline__ DigitMP() {
      }

      // used for constructing external, i.e., bounds checking is required
      __device__ __forceinline__ DigitMP(bool compact, bool useTextureCache, uint32_t *base, int32_t length, int32_t stride, int32_t count, uint32_t thread=0xFFFFFFFF) {
        int32_t digits=divide<_size>(length+_size-1);

        if(thread==0xFFFFFFFF)
          thread=blockIdx.x*blockDim.x+threadIdx.x;

        if(compact) {
          if(useTextureCache)
            _storage=xmpDigitStorage_compact_tex_v1;
          else
            _storage=xmpDigitStorage_compact_v1;

          _base=base + (thread % count) * length;
        }
        else {
          _storage=xmpDigitStorage_strided_v1;

          _base=base + (thread % count);
        }

        _start=0;
        _digits=digits;
        _length=length;
        _stride=stride;
        _width=0;
        _words=0;
      }

      // used for constructing internal values, i.e., no bounds checking required
      __device__ __forceinline__ DigitMP(bool compact, bool useTextureCache, uint32_t *base) {
        if(compact) {
          if(useTextureCache)
            if(_size%4!=0)
              _storage=xmpDigitStorage_compact_internal_tex_v1;
            else
              _storage=xmpDigitStorage_compact_internal_tex_v4;
          else
            if(_size%4!=0)
              _storage=xmpDigitStorage_compact_internal_v1;
            else
              _storage=xmpDigitStorage_compact_internal_v4;
        }
        else {
          if(_size%4!=0)
            _storage=xmpDigitStorage_warp_strided_internal_v1;
          else
            _storage=xmpDigitStorage_warp_strided_internal_v4;
        }

        _base=base;
        _start=0;
        _digits=0x7FFFFFFF;
        _length=0;
        _stride=32;
        _width=0;
        _words=0;
      }

      // used for constructing internal values, i.e., no bounds checking required
      __device__ __forceinline__ DigitMP(bool compact, bool useTextureCache, uint32_t *base, int32_t digits, uint32_t thread=0xFFFFFFFF) {
        PTXInliner inliner;
        uint32_t   words;
        uint64_t   offset;

        if(thread==0xFFFFFFFF)
          thread=blockIdx.x*blockDim.x+threadIdx.x;

        if(compact) {
          if(useTextureCache) {
            if(_size%4!=0)
              _storage=xmpDigitStorage_compact_internal_tex_v1;
            else
              _storage=xmpDigitStorage_compact_internal_tex_v4;
          }
          else {
            if(_size%4!=0)
              _storage=xmpDigitStorage_compact_internal_v1;
            else
              _storage=xmpDigitStorage_compact_internal_v4;
          }

          // FIX FIX FIX - this looks broken, shouldn't it be offset by thread?

          offset=0;
          words=digits*_size;
          inliner.MADWIDE(offset, thread, words, offset);

          _base=base + offset;
        }
        else {
          if(_size%4!=0) {
            _storage=xmpDigitStorage_warp_strided_internal_v1;

            offset=thread & 0x1F;
            thread=thread & ~0x1F;
            words=digits*_size;
            inliner.MADWIDE(offset, thread, words, offset);

            _base=base + offset;
          }
          else {
            _storage=xmpDigitStorage_warp_strided_internal_v4;

            offset=(thread & 0x1F)*4;
            thread=thread & ~0x1F;
            words=digits*_size;
            inliner.MADWIDE(offset, thread, words, offset);

            _base=base + offset;
          }
        }

        _start=0;
        _digits=digits;
        _length=0;
        _stride=32;
        _width=0;
        _words=0;
      }

      // used to construct a warp_distributed internal value with no bounds checking
      __device__ __forceinline__ DigitMP(uint32_t *base, int32_t width, int32_t words, int32_t index, int32_t count, int32_t size, bool transpose=false, uint32_t thread=0xFFFFFFFF) {
        PTXInliner inliner;
        int32_t    digits=divide<_size>(width*words);    // remainder must be zero
        uint32_t   window;
        uint64_t   offset;

        if(thread==0xFFFFFFFF)
          thread=blockIdx.x*blockDim.x+threadIdx.x;

        thread=thread*width;
        offset=thread & 0x1F;
        thread=thread & ~0x1F;
        window=size*words;
        inliner.MADWIDE(offset, thread, window, offset);

        _base=base + offset;
        if(transpose)
          _storage=xmpDigitStorage_warp_distributed_transpose_v1;
        else
          _storage=xmpDigitStorage_warp_distributed_v1;
        _start=index*digits;
        _digits=count*digits;
        _length=0;
        _stride=32;
        _width=width;
        _words=words;
      }

      // used for constructing shared internal values, i.e., no bound checking required
      __device__ __forceinline__ DigitMP(int32_t digits, int32_t count, int32_t thread=-1) {
        extern __shared__ uint32_t base[];

        if(thread<0)
          thread=threadIdx.x;

        if(_size%8==0) {       // use chunks of 8 words
          _storage=xmpDigitStorage_shared_chunked_internal_v2;
          _base=base + thread%count*8;
        }
        else if(_size%2==0) {  // if the digit size is not divisible by 8, you won't get too many bank conflicts
          _storage=xmpDigitStorage_shared_internal_v2;
          _base=base + thread%count*2;
        }
        else {
          _storage=xmpDigitStorage_shared_internal_v1;
          _base=base + thread%count;
        }

        _start=0;
        _digits=digits;
        _length=0;
        _stride=count;
        _width=0;
        _words=0;
      }

      __device__ __forceinline__ DigitMP(DigitMP<_size> source, int32_t start, int32_t digits) {
        _storage=source._storage;
        _base=source._base;
        _start=source._start + start;
        _digits=digits;
        _length=source._length;
        _stride=source._stride;
        _width=0;
        _words=0;
      }

      __device__ __forceinline__ void set(DigitMP<_size> source) {
        _storage=source._storage;
        _base=source._base;
        _start=source._start;
        _digits=source._digits;
        _length=source._length;
        _stride=source._stride;
        _width=0;
        _words=0;
      }

      __device__ __forceinline__ int32_t digits() {
        return _digits;
      }

      __device__ __forceinline__ int32_t length() {
        return _length;
      }

      __device__ __forceinline__ void load_digit(RegMP x, int32_t digit) {
        if(x.length()!=_size) RMP_ERROR("load_digit() - length mismatch");

        // external storage modes
        if(_storage==xmpDigitStorage_compact_v1) {
          uint32_t *base=_base+(digit+_start)*_size;

          #pragma unroll
          for(int index=0;index<_size;index++) {
            x[index]=0;
            if((digit+_start)*_size+index<_length)
              x[index]=base[index];
          }
        }
        if(_storage==xmpDigitStorage_compact_tex_v1) {
          uint32_t *base=_base+(digit+_start)*_size;

          #pragma unroll
          for(int index=0;index<_size;index++) {
            x[index]=0;
            if((digit+_start)*_size+index<_length)
              x[index]=__ldg(base+index);
          }
        }
        if(_storage==xmpDigitStorage_strided_v1) {
          int64_t   offset=(digit+_start)*_size;
          uint32_t *base=_base+offset*_stride;

          #pragma unroll
          for(int index=0;index<_size;index++) {
            x[index]=0;
            if((digit+_start)*_size+index<_length)
              x[index]=base[index*_stride];
          }
        }

        // internal storge modes
        if(_storage==xmpDigitStorage_compact_internal_v1) {
          uint32_t *base=_base+(digit+_start)*_size;

          #pragma unroll
          for(int index=0;index<_size;index++)
            x[index]=base[index];
        }
        if(_storage==xmpDigitStorage_compact_internal_v4) {
          uint4 *base=(uint4 *)(_base+(digit+_start)*_size);
          uint4  data4;

          #pragma unroll
          for(int index=0;index<_size/4;index++) {
            data4=base[index];
            x[index*4+0]=data4.x;
            x[index*4+1]=data4.y;
            x[index*4+2]=data4.z;
            x[index*4+3]=data4.w;
          }
        }
        if(_storage==xmpDigitStorage_compact_internal_tex_v1) {
          uint32_t *base=_base+(digit+_start)*_size;

          #pragma unroll
          for(int index=0;index<_size;index++)
            x[index]=__ldg(base+index);
        }
        if(_storage==xmpDigitStorage_compact_internal_tex_v4) {
          uint4 *base=(uint4 *)(_base+(digit+_start)*_size);
          uint4  data4;

          #pragma unroll
          for(int index=0;index<_size/4;index++) {
            data4=__ldg(base+index);
            x[index*4+0]=data4.x;
            x[index*4+1]=data4.y;
            x[index*4+2]=data4.z;
            x[index*4+3]=data4.w;
          }
        }
        if(_storage==xmpDigitStorage_warp_strided_internal_v1) {
          uint32_t *base=_base+(digit+_start)*_size*32;

          #pragma unroll
          for(int index=0;index<_size;index++)
            x[index]=base[index*32];
        }
        if(_storage==xmpDigitStorage_warp_strided_internal_v4) {
          uint4 *base=(uint4 *)(_base+(digit+_start)*_size*32);
          uint4  data4;

          #pragma unroll
          for(int index=0;index<_size/4;index++) {
            data4=base[index*32];
            x[index*4+0]=data4.x;
            x[index*4+1]=data4.y;
            x[index*4+2]=data4.z;
            x[index*4+3]=data4.w;
          }
        }

        // shared memory routines
        if(_storage==xmpDigitStorage_shared_internal_v1) {
          uint32_t *base=_base+(digit+_start)*_size*_stride;

          #pragma unroll
          for(int index=0;index<_size;index++)
            x[index]=base[index*_stride];
        }
        if(_storage==xmpDigitStorage_shared_internal_v2) {
          uint2 *base=(uint2 *)(_base+(digit+_start)*_size*_stride);
          uint2  data2;

          #pragma unroll
          for(int index=0;index<_size/2;index++) {
            data2=base[index*_stride];
            x[index*2+0]=data2.x;
            x[index*2+1]=data2.y;
          }
        }
        if(_storage==xmpDigitStorage_shared_chunked_internal_v2) {
          uint2 *base=(uint2 *)(_base+(digit+_start)*_size*_stride);
          uint2  data2;

          #pragma unroll
          for(int chunk=0;chunk<_size/8;chunk++) {
            #pragma unroll
            for(int index=0;index<4;index++) {
              data2=base[chunk*4+index];
              x[chunk*8+index*2+0]=data2.x;
              x[chunk*8+index*2+1]=data2.y;
            }
            base=base+_stride*4;
          }
        }

        // warp distributed
        if(_storage==xmpDigitStorage_warp_distributed_v1) {
          RMP_ERROR("load_digit() - can't load warp_distributed");
        }
        if(_storage==xmpDigitStorage_warp_distributed_v2) {
          RMP_ERROR("load_digit() - can't load warp_distributed");
        }
        if(_storage==xmpDigitStorage_warp_distributed_v4) {
          RMP_ERROR("load_digit() - can't load warp_distributed");
        }
        if(_storage==xmpDigitStorage_warp_distributed_transpose_v1) {
          RMP_ERROR("load_digit() - can't load warp_distributed");
        }
      }

      __device__ __forceinline__ void store_digit(RegMP x, int32_t digit) {
        if(x.length()!=_size) RMP_ERROR("store_digit() - length mismatch");

        // external storage modes
        if(_storage==xmpDigitStorage_compact_v1) {
          uint32_t *base=_base+(digit+_start)*_size;

          #pragma unroll
          for(int index=0;index<_size;index++)
            if((digit+_start)*_size+index<_length)
              base[index]=x[index];
        }
        if(_storage==xmpDigitStorage_compact_tex_v1) {
          RMP_ERROR("store_digit() - can't store to tex");
        }
        if(_storage==xmpDigitStorage_strided_v1) {
          int64_t   offset=(digit+_start)*_size;
          uint32_t *base=_base+offset*_stride;

          #pragma unroll
          for(int index=0;index<_size;index++)
            if((digit+_start)*_size+index<_length)
              base[index*_stride]=x[index];
        }

        // internal storge modes
        if(_storage==xmpDigitStorage_compact_internal_v1) {
          uint32_t *base=_base+(digit+_start)*_size;

          #pragma unroll
          for(int index=0;index<_size;index++)
            base[index]=x[index];
        }
        if(_storage==xmpDigitStorage_compact_internal_v4) {
          uint4 *base=(uint4 *)(_base+(digit+_start)*_size);
          uint4  data4;

          #pragma unroll
          for(int index=0;index<_size/4;index++) {
            data4.x=x[index*4+0];
            data4.y=x[index*4+1];
            data4.z=x[index*4+2];
            data4.w=x[index*4+3];
            base[index]=data4;
          }
        }
        if(_storage==xmpDigitStorage_compact_internal_tex_v1) {
          RMP_ERROR("store_digit() - can't store to tex");
        }
        if(_storage==xmpDigitStorage_compact_internal_tex_v4) {
          RMP_ERROR("store_digit() - can't store to tex");
        }
        if(_storage==xmpDigitStorage_warp_strided_internal_v1) {
          uint32_t *base=_base+(digit+_start)*_size*32;

          #pragma unroll
          for(int index=0;index<_size;index++)
            base[index*32]=x[index];
        }
        if(_storage==xmpDigitStorage_warp_strided_internal_v4) {
          uint4 *base=(uint4 *)(_base+(digit+_start)*_size*32);
          uint4  data4;

          #pragma unroll
          for(int index=0;index<_size/4;index++) {
            data4.x=x[index*4+0];
            data4.y=x[index*4+1];
            data4.z=x[index*4+2];
            data4.w=x[index*4+3];
            base[index*32]=data4;
          }
        }

        // shared memory routines
        if(_storage==xmpDigitStorage_shared_internal_v1) {
          uint32_t *base=_base+(digit+_start)*_size*_stride;

          #pragma unroll
          for(int index=0;index<_size;index++)
            base[index*_stride]=x[index];
        }
        if(_storage==xmpDigitStorage_shared_internal_v2) {
          uint2 *base=(uint2 *)(_base+(digit+_start)*_size*_stride);
          uint2  data2;

          #pragma unroll
          for(int index=0;index<_size/2;index++) {
            data2.x=x[index*2+0];
            data2.y=x[index*2+1];
            base[index*_stride]=data2;
          }
        }
        if(_storage==xmpDigitStorage_shared_chunked_internal_v2) {
          uint2 *base=(uint2 *)(_base+(digit+_start)*_size*_stride);
          uint2  data2;

          #pragma unroll
          for(int chunk=0;chunk<_size/8;chunk++) {
            #pragma unroll
            for(int index=0;index<4;index++) {
              data2.x=x[chunk*8+index*2+0];
              data2.y=x[chunk*8+index*2+1];
              base[chunk*4+index]=data2;
            }
            base=base+_stride*4;
          }
        }

        // warp distributed
        if(_storage==xmpDigitStorage_warp_distributed_v1) {
          uint32_t *base=_base + _start*_size*(32/_width);
          int32_t   offset=digit*_size;

          #pragma nounroll
          for(int32_t word=0;word<_size;word++)
            base[(word+offset)/_words + (word+offset)%_words*32]=x[word];
        }
        if(_storage==xmpDigitStorage_warp_distributed_v2) {
          RMP_ERROR("store_digit() - store v2 not supported yet");
        }
        if(_storage==xmpDigitStorage_warp_distributed_v4) {
          RMP_ERROR("load_digit() - store v4 not supported yet");
        }
        if(_storage==xmpDigitStorage_warp_distributed_transpose_v1) {
          uint32_t *base=_base + _start*_size*(32/_width);
          int32_t   offset=digit*_size;

          #pragma nounroll
          for(int32_t word=0;word<_size;word++)
            base[(word+offset)/_width*32 + (word+offset)%_width]=x[word];
        }

      }

      __device__ __forceinline__ uint32_t load_digit_word(int32_t digit, int32_t offset) {
        // external storage modes
        if(_storage==xmpDigitStorage_compact_v1) {
          uint32_t *base=_base+(digit+_start)*_size;

          if((digit+_start)*_size+offset<_length)
            return base[offset];
          return 0;
        }
        if(_storage==xmpDigitStorage_compact_tex_v1) {
          uint32_t *base=_base+(digit+_start)*_size;

          if((digit+_start)*_size+offset<_length)
            return __ldg(base+offset);
          return 0;
        }
        if(_storage==xmpDigitStorage_strided_v1) {
          int64_t   word=(digit+_start)*_size;
          uint32_t *base=_base+word*_stride;

          if((digit+_start)*_size+offset<_length)
            return base[offset*_stride];
          return 0;
        }

        // internal storge modes
        if(_storage==xmpDigitStorage_compact_internal_v1) {
          uint32_t *base=_base+(digit+_start)*_size;

          return base[offset];
        }
        if(_storage==xmpDigitStorage_compact_internal_v4) {
          uint4 *base=(uint4 *)(_base+(digit+_start)*_size);
          uint4  data4;

          data4=base[offset/4];
          if((offset & 0x03)==0)
            return data4.x;
          if((offset & 0x03)==1)
            return data4.y;
          if((offset & 0x03)==2)
            return data4.z;
          if((offset & 0x03)==3)
            return data4.w;
        }
        if(_storage==xmpDigitStorage_compact_internal_tex_v1) {
          uint32_t *base=_base+(digit+_start)*_size;

          return __ldg(base+offset);
        }
        if(_storage==xmpDigitStorage_compact_internal_tex_v4) {
          uint4 *base=(uint4 *)(_base+(digit+_start)*_size);
          uint4  data4;

          data4=__ldg(base+offset/4);
          if((offset & 0x03)==0)
            return data4.x;
          if((offset & 0x03)==1)
            return data4.y;
          if((offset & 0x03)==2)
            return data4.z;
          if((offset & 0x03)==3)
            return data4.w;
        }
        if(_storage==xmpDigitStorage_warp_strided_internal_v1) {
          uint32_t *base=_base+(digit+_start)*_size*32;

          return base[offset*32];
        }
        if(_storage==xmpDigitStorage_warp_strided_internal_v4) {
          uint4 *base=(uint4 *)(_base+(digit+_start)*_size*32);
          uint4  data4;

          data4=base[offset/4*32];
          if((offset & 0x03)==0)
            return data4.x;
          if((offset & 0x03)==1)
            return data4.y;
          if((offset & 0x03)==2)
            return data4.z;
          if((offset & 0x03)==3)
            return data4.w;
        }

        // shared memory implementations
        if(_storage==xmpDigitStorage_shared_internal_v1) {
          uint32_t *base=_base+(digit+_start)*_size*_stride;

          return base[offset*_stride];
        }
        if(_storage==xmpDigitStorage_shared_internal_v2) {
          uint2 *base=(uint2 *)(_base+(digit+_start)*_size*_stride);
          uint2  data2;

          data2=base[offset/2*_stride];
          if((offset & 0x01)==0)
            return data2.x;
          else
            return data2.y;
        }
        if(_storage==xmpDigitStorage_shared_chunked_internal_v2) {
          uint2 *base=(uint2 *)(_base+(digit+_start)*_size*_stride);
          uint2  data2;

          data2=base[((offset & ~0x07)*_stride + (offset & 0x07))/2];
          if((offset & 0x01)==0)
            return data2.x;
          else
            return data2.y;
        }

        // warp distributed
        if(_storage==xmpDigitStorage_warp_distributed_v1) {
          RMP_ERROR("load_digit_word() - can't load warp_distributed");
        }
        if(_storage==xmpDigitStorage_warp_distributed_v2) {
          RMP_ERROR("load_digit_word() - can't load warp_distributed");
        }
        if(_storage==xmpDigitStorage_warp_distributed_v4) {
          RMP_ERROR("load_digit_word() - can't load warp_distributed");
        }
        if(_storage==xmpDigitStorage_warp_distributed_transpose_v1) {
          RMP_ERROR("load_digit_word() - can't load warp_distributed");
        }

        return 0;
      }

      __device__ __forceinline__ uint32_t get_bits(int32_t bitOffset, int32_t bitLength) {
        uint32_t low, high;

        // external storage modes
        if(_storage==xmpDigitStorage_compact_v1) {
          RMP_ERROR("get_bits() - compact_v1 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_compact_tex_v1) {
          RMP_ERROR("get_bits() - compact_tex_v1 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_strided_v1) {
          RMP_ERROR("get_bits() - strided_v1 not supported");
          return 0xFFFFFFFF;
        }

        // internal storge modes
        if(_storage==xmpDigitStorage_compact_internal_v1) {
          RMP_ERROR("get_bits() - compact_internal_v1 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_compact_internal_v4) {
          RMP_ERROR("get_bits() - compact_internal_v4 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_compact_internal_tex_v1) {
          RMP_ERROR("get_bits() - compact_internal_tex_v1 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_compact_internal_tex_v4) {
          RMP_ERROR("get_bits() - compact_internal_tex_v4 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_warp_strided_internal_v1) {
          uint32_t *base=_base + ((bitOffset>>5) + _start*_size)*32;

          bitOffset=bitOffset & 0x1F;
          if(32-bitOffset>=bitLength) {
            low=base[0];
            high=0;
          }
          else {
            low=base[0];
            high=base[32];
          }
        }
        if(_storage==xmpDigitStorage_warp_strided_internal_v4) {
          uint4 *base=(uint4 *)(_base + ((bitOffset>>7)*4 + _start*_size)*32);

          bitOffset=bitOffset & 0x7F;
          if(128-bitOffset>=bitLength) {
            uint4 data4=base[0];

            if((bitOffset & 0x60)==0x00) {
              low=data4.x;
              high=data4.y;
            }
            if((bitOffset & 0x60)==0x20) {
              low=data4.y;
              high=data4.z;
            }
            if((bitOffset & 0x60)==0x40) {
              low=data4.z;
              high=data4.w;
            }
            if((bitOffset & 0x60)==0x60) {
              low=data4.w;
              high=0;
            }
          }
          else {
            low=base[0].w;
            high=base[32].x;
          }
          bitOffset=bitOffset & 0x1F;
        }

        // shared memory implementations
        if(_storage==xmpDigitStorage_shared_internal_v1) {
          RMP_ERROR("get_bits() - shared_internal_v2 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_shared_internal_v2) {
          RMP_ERROR("get_bits() - shared_internal_v2 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_shared_chunked_internal_v2) {
          RMP_ERROR("get_bits() - shared_chunked_internal_v2 not supported");
          return 0xFFFFFFFF;
        }

        // warp distributed
        if(_storage==xmpDigitStorage_warp_distributed_v1) {
          RMP_ERROR("get_bits() - warp_distributed_v1 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_warp_distributed_v2) {
          RMP_ERROR("get_bits() - warp_distributed_v2 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_warp_distributed_v4) {
          RMP_ERROR("get_bits() - warp_distributed_v4 not supported");
          return 0xFFFFFFFF;
        }
        if(_storage==xmpDigitStorage_warp_distributed_transpose_v1) {
          RMP_ERROR("get_bits() - warp_distributed_transpose_v1 not supported");
          return 0xFFFFFFFF;
        }

        low=low>>bitOffset;
        high=high<<32-bitOffset;
        return (low | high) & (1<<bitLength)-1;
      }
  };
}

#include "DigitMP_impl.h"
#include "assignment.h"
#include "clz.h"
#include "add.h"
#include "sub.h"
#include "shift.h"
#include "mul.h"
#include "sqr.h"
#include "div.h"
#include "reduce.h"

