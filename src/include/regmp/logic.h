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
  __device__ __forceinline__ void bitwise_not(RegMP r, RegMP a) {
    uint32_t index;

    if(r.length()!=a.length()) RMP_ERROR("bitwise_not() - length mismatch");
    for(index=0;index<r.length();index++)
      r[index]=~a[index];
  }

  __device__ __forceinline__ void bitwise_and(RegMP r, RegMP a, RegMP b) {
    if(r.length()!=a.length() || r.length()!=b.length()) RMP_ERROR("bitwise_and() - length mismatch");

    for(int index=0;index<r.length();index++)
      r[index]=a[index] & b[index];
  }

  __device__ __forceinline__ void bitwise_and(RegMP r, RegMP a, uint32_t b) {
    if(r.length()!=a.length()) RMP_ERROR("bitwise_and() - length mismatch");

    for(int index=0;index<r.length();index++)
      r[index]=a[index] & b;
  }

  __device__ __forceinline__ void bitwise_or(RegMP r, RegMP a, RegMP b) {
    uint32_t index;

    if(r.length()!=a.length() || r.length()!=b.length()) RMP_ERROR("bitwise_or() - length mismatch");
    for(index=0;index<r.length();index++)
      r[index]=a[index] | b[index];
  }

  __device__ __forceinline__ void bitwise_or(RegMP r, RegMP a, uint32_t b) {
    if(r.length()!=a.length()) RMP_ERROR("bitwise_or() - length mismatch");

    for(int index=0;index<r.length();index++)
      r[index]=a[index] | b;
  }

  __device__ __forceinline__ void bitwise_xor(RegMP r, RegMP a, RegMP b) {
    uint32_t index;

    if(r.length()!=a.length() || r.length()!=b.length()) RMP_ERROR("bitwise_xor() - length mismatch");
    for(index=0;index<r.length();index++)
      r[index]=a[index] ^ b[index];
  }

  __device__ __forceinline__ void bitwise_xor(RegMP r, RegMP a, uint32_t b) {
    if(r.length()!=a.length()) RMP_ERROR("bitwise_xor() - length mismatch");

    for(int index=0;index<r.length();index++)
      r[index]=a[index] ^ b;
  }
}
