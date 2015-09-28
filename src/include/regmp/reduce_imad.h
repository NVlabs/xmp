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
  __device__ __forceinline__ void reduce(RegMP r, RegMP xx, RegMP n, uint32_t np0) {
    PTXInliner inliner;
    uint32_t   temp, carry, zero=0, ff=0xFFFFFFFF;
    int        index, count, size=n.length();

    if(n.length()!=r.length()) RMP_ERROR("reduce() - length mismatch");

    #pragma unroll
    for(count=0;count<xx.length()-n.length();count++) {
      inliner.MULLO(temp, xx[count], np0);
      PTXChain chain1(size+2);
      chain1.ADD(xx[count], xx[count], ff);
      #pragma unroll
      for(index=1;index<size;index++)
        chain1.MADLO(xx[count+index], n[index], temp, xx[count+index]);
      chain1.ADD(xx[count+index], xx[count+index], (count==0) ? zero : carry);
      chain1.ADD(carry, zero, zero);
      chain1.end();

      PTXChain chain2(size+1);
      #pragma unroll
      for(index=0;index<size;index++)
        chain2.MADHI(xx[count+index+1], n[index], temp, xx[count+index+1]);
      chain2.ADD(carry, carry, zero);
      chain2.end();
    }

    carry=-carry;
    #pragma unroll
    for(index=0;index<n.length();index++)
      n[index]=n[index] & carry;

    PTXChain chain3(size);
    #pragma unroll
    for(index=0;index<n.length();index++)
      chain3.SUB(r[index], xx[count+index], n[index]);
    chain3.end();
  }

  __device__ __forceinline__ uint32_t _reduce(RegMP xx, RegMP coefficients, RegMP n, uint32_t np0) {
    PTXInliner inliner;
    uint32_t   carry, zero=0, ff=0xFFFFFFFF;
    int        index, count, size=n.length();

    #pragma unroll
    for(count=0;count<xx.length()-n.length();count++) {
      inliner.MULLO(coefficients[count], xx[count], np0);
      PTXChain chain1(size+2);
      chain1.ADD(xx[count], xx[count], ff);
      #pragma unroll
      for(index=1;index<size;index++)
        chain1.MADLO(xx[count+index], n[index], coefficients[count], xx[count+index]);
      chain1.ADD(xx[count+index], xx[count+index], (count==0) ? zero : carry);
      chain1.ADD(carry, zero, zero);
      chain1.end();

      PTXChain chain2(size+1);
      #pragma unroll
      for(index=0;index<size;index++)
        chain2.MADHI(xx[count+index+1], n[index], coefficients[count], xx[count+index+1]);
      chain2.ADD(carry, carry, zero);
      chain2.end();
    }
    return carry;
  }
}
