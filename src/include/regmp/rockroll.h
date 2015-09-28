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
  __device__ __forceinline__ void _rock(RegMP rr) {
    PTXInliner inliner;
    int        length=rr.length();
    uint32_t   zero=0, permutation=0x5432;

    // shift RR left by 16 bits
    #pragma unroll
    for(int word=length-1;word>0;word--)
      inliner.PERMUTE(rr[word], rr[word-1], rr[word], permutation);
    inliner.PERMUTE(rr[0], zero, rr[0], permutation);
  }

  __device__ __forceinline__ void _roll(RegMP rr) {
    PTXInliner inliner;
    int        length=rr.length();
    uint32_t   permutation=0x5432;

    // shift RR right by 16 bits
    #pragma unroll
    for(int word=0;word<length-1;word++)
      inliner.PERMUTE(rr[word], rr[word], rr[word+1], permutation);
    rr[length-1]=((int32_t)rr[length-1])>>16;
  }
}
