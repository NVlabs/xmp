/***
Copyright (c) 2014-2015, Niall Emmart.  All rights reserved.
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
  __device__ __forceinline__ void shift_right_const(RegMP r, RegMP x, int32_t words) {
    if(r.length()!=x.length()) RMP_ERROR("shift_right_const() - length mismatch");

    #pragma unroll
    for(int index=0;index<r.length();index++)
      r[index]=(index+words<x.length()) ? x[index+words] : 0;
  }

  __device__ __forceinline__ void shift_left_const(RegMP r, RegMP x, int32_t words) {
    if(r.length()!=x.length()) RMP_ERROR("shift_left_const() - length mismatch");

    #pragma unroll
    for(int index=r.length()-1;index>=0;index--)
      r[index]=(index>=words) ? x[index-words] : 0;
  }

  __device__ __forceinline__ void shift_right_words(RegMP r, RegMP x, int32_t words) {
    if(r.length()!=x.length()) RMP_ERROR("shift_right_words() - length mismatch");

    #pragma unroll
    for(int index=0;index<x.length()-1;index++)
      r[index]=((words & 0x01)==0) ? x[index] : x[index+1];
    r[x.length()-1]=((words & 0x01)==0) ? x[x.length()-1] : 0;

    #pragma unroll
    for(int index=0;index<x.length()-2;index++)
      r[index]=((words & 0x02)==0) ? r[index] : r[index+2];
    r[x.length()-2]=((words & 0x02)==0) ? r[x.length()-2] : 0;
    r[x.length()-1]=((words & 0x02)==0) ? r[x.length()-1] : 0;

    words=words & ~0x03;
    while(words>=4) {
      shift_right_const(r, r, 4);
      words=words-4;
    }
  }

  __device__ __forceinline__ void shift_left_words(RegMP r, RegMP x, int32_t words) {
    if(r.length()!=x.length()) RMP_ERROR("shift_left_words() - length mismatch");

    #pragma unroll
    for(int index=x.length()-1;index>=1;index--)
      r[index]=((words & 0x01)==0) ? x[index] : x[index-1];
    r[0]=((words & 0x01)==0) ? x[0] : 0;

    #pragma unroll
    for(int index=x.length()-1;index>=2;index--)
      r[index]=((words & 0x02)==0) ? r[index] : r[index-2];
    r[1]=((words & 0x02)==0) ? r[1] : 0;
    r[0]=((words & 0x02)==0) ? r[0] : 0;

    words=words & ~0x03;
    while(words>=4) {
      shift_left_const(r, r, 4);
      words=words-4;
    }
  }

  __device__ __forceinline__ void shift_right_bits(RegMP r, RegMP x, int32_t bits) {
    PTXInliner inliner;
    uint32_t   shiftBits=bits;

    if(r.length()!=x.length()) RMP_ERROR("shift_right_bits() - length mismatch");

    if(bits==0)
      return;
#if __CUDA_ARCH__<350
    #pragma unroll
    for(int index=0;index<x.length()-1;index++)
      r[index]=(x[index]>>shiftBits) | (x[index+1]<<32-shiftBits);
    r[x.length()-1]=x[x.length()-1]>>shiftBits;
#else
    #pragma unroll
    for(int index=0;index<x.length()-1;index++)
      inliner.SHF_R_WRAP(r[index], x[index], x[index+1], shiftBits);
    r[x.length()-1]=x[x.length()-1]>>shiftBits;
#endif
  }

  __device__ __forceinline__ void shift_left_bits(RegMP r, RegMP x, int32_t bits) {
    PTXInliner inliner;
    uint32_t   shiftBits=bits;

    if(r.length()!=x.length()) RMP_ERROR("shift_left_bits() - length mismatch");

    if(bits==0)
      return;
#if __CUDA_ARCH__<350
    #pragma unroll
    for(int index=x.length()-1;index>=1;index--)
      r[index]=(x[index]<<shiftBits) | (x[index-1]>>32-shiftBits);
    r[0]=x[0]<<shiftBits;
#else
    #pragma unroll
    for(int index=x.length()-1;index>=1;index--)
      inliner.SHF_L_WRAP(r[index], x[index-1], x[index], shiftBits);
    r[0]=x[0]<<shiftBits;
#endif
  }
}
