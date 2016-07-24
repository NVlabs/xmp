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
  __device__ __forceinline__ void reduce(RegMP r, RegMP xx, RegMP n, RegMP temp, uint32_t np0) {
    PTXInliner inliner;
    int32_t    i, j, size=n.length();
    uint32_t   q, carry=0, add=0, zero=0, permutation=0x5432;

    if(n.length()!=r.length()) RMP_ERROR("reduce() - length mismatch");

    set_ui(temp, 0);
    #pragma unroll
    for(i=0;i<size;i++) {
      inliner.XMADLL(q, xx[i], np0, zero);

      PTXChain chain1(size+2);
      chain1.XMADLL(xx[0], q, n[0], xx[i]);
      inliner.PERMUTE(add, xx[0], add, permutation);
      #pragma unroll
      for(j=1;j<size;j++)
        chain1.XMADLL(xx[i+j], q, n[j], xx[i+j]);
      chain1.ADD(xx[i+size], xx[i+size], carry);
      chain1.ADD(carry, zero, zero);
      chain1.end();

      PTXChain chain2(size);
      #pragma unroll
      for(j=0;j<size;j++)
        chain2.XMADLH(temp[j], q, n[j], temp[j]);
      chain2.end();

      inliner.ADD_CC(temp[0], temp[0], add);
      inliner.ADDC(add, zero, zero);

      inliner.XMADLL(q, temp[0], np0, zero);

      PTXChain chain3(size+1);
      chain3.XMADLL(temp[0], q, n[0], temp[0]);
      inliner.PERMUTE(add, temp[0], add, permutation);
      #pragma unroll
      for(j=1;j<size;j++)
        chain3.XMADLL(temp[j-1], q, n[j], temp[j]);
      chain3.ADD(temp[size-1], zero, zero);
      chain3.end();

      PTXChain chain4(size+1);
      #pragma unroll
      for(j=0;j<size;j++)
        chain4.XMADLH(xx[i+j+1], q, n[j], xx[i+j+1]);
      chain4.ADD(carry, carry, zero);
      chain4.end();

      if(i<size-1) {
        inliner.ADD_CC(xx[i+1], xx[i+1], add);
        inliner.ADDC(add, zero, zero);
      }
    }

    PTXChain chainX(size+1);
    chainX.ADD(xx[size], xx[size], add);
    #pragma unroll
    for(int j=1;j<size;j++)
      chainX.ADD(xx[j+size], xx[j+size], zero);
    chainX.ADD(carry, carry, zero);
    chainX.end();

    // shift RR left by 16 bits
    #pragma unroll
    for(int j=size-1;j>0;j--)
      inliner.PERMUTE(temp[j], temp[j-1], temp[j], permutation);
    inliner.PERMUTE(temp[0], zero, temp[0], permutation);

    PTXChain chain5(size+1);
    #pragma unroll
    for(j=0;j<size;j++)
      chain5.ADD(xx[j+size], xx[j+size], temp[j]);
    chain5.ADD(carry, carry, zero);
    chain5.end();

    carry=-carry;
    bitwise_and(n, n, carry);

    PTXChain chain6(size);
    #pragma unroll
    for(j=0;j<size;j++)
      chain6.SUB(r[j], xx[j+size], n[j]);
    chain6.end();
  }

/*
  There is a rare carry case that causes this routine to fail.
  To reproduce, run mod-exp at 512 with the following example:
    a=b1e451f4e5dbbf65e7901088e87820ed3ce644365acfae8f372e42dfb5c24232c4a1bc48ce5471b20dfc1827c1a5545a19eadbbf7d99579d82b02f036c18d95d
    b=8ddcf087f521bd3df8748641c40e1d8224548611769b9eb4d0cde0c6bdab8cb2845ea8d4a8585ae9b105a3176125895d30a3d95aba31e458500dd7b7fb665d32
    c=ffdba45c89ec1b188d3c8f549a32ccafd43b06fb42ec7cb0585c4e2ad78fa438d36b7796b9ea4adcc67e2a97823e5f01adfc2367c241f8e1c83ed4a11a70dde9

  __device__ __forceinline__ void reduce(RegMP r, RegMP xx, RegMP n, uint32_t np0) {
    PTXInliner inliner;
    uint32_t   temp, sum, high, carry, zero=0, permuteShift16=0x5432, permuteHighHigh=0x7632, permuteLowLow=0x5410;
    int        i, j, size=n.length();

    if(n.length()!=r.length()) RMP_ERROR("reduce() - length mismatch");

    #pragma unroll
    for(i=0;i<size;i++) {
      temp=0;
      inliner.XMADLL(temp, xx[i], np0, temp);

      PTXChain chain1(size+2);
      #pragma unroll
      for(j=0;j<size;j++)
        chain1.XMADLL(xx[i+j], temp, n[j], xx[i+j]);
      chain1.ADD(xx[i+size], xx[i+size], (i==0) ? zero : high);
      chain1.ADD(high, zero, zero);
      chain1.end();

      inliner.PERMUTE(sum, xx[i], xx[i+1], permuteShift16);

      if(i==0) {
        PTXChain chain2(2);
        chain2.XMADLH(sum, temp, n[0], sum);
        chain2.ADD(carry, zero, zero);
        chain2.end();
      }
      else {
        inliner.XMADLH(carry, temp, n[0], carry);
        PTXChain chain3(2);
        chain3.ADD(sum, sum, carry);
        chain3.ADD(carry, zero, zero);
        chain3.end();
      }

      #pragma unroll
      for(j=1;j<size;j++) {   // arg!  should be for(j=1;j<=i;j++)
        if(j<=i) {            // so, this is the compiler work around.  Full length loop with an extra if
          PTXChain chain4(2);
          chain4.XMADLH(sum, n[j], xx[i-j], sum);
          chain4.ADD(carry, carry, zero);
          chain4.end();

          PTXChain chain5(2);
          chain5.XMADHL(sum, n[j], xx[i-j], sum);
          chain5.ADD(carry, carry, zero);
          chain5.end();
        }
      }

      xx[i]=0;
      inliner.XMADLL(xx[i], sum, np0, xx[i]);
      inliner.PERMUTE(xx[i], temp, xx[i], permuteLowLow);

      PTXChain chain5(2);
      chain5.XMADHL(sum, xx[i], n[0], sum);
      chain5.ADD(carry, carry, zero);
      chain5.end();

      inliner.PERMUTE(xx[i+1], sum, xx[i+1], permuteHighHigh);

      PTXChain chain6(size+1);
      #pragma unroll
      for(j=0;j<size;j++)
        chain6.XMADHH(xx[i+j+1], xx[i], n[j], xx[i+j+1]);
      chain6.ADD(high, high, zero);
      chain6.end();
    }

    inliner.PERMUTE(high, xx[2*size-1], high, permuteShift16);
    #pragma unroll
    for(i=2*size-1;i>size;i--)
      inliner.PERMUTE(xx[i], xx[i-1], xx[i], permuteShift16);
    inliner.PERMUTE(xx[size], zero, xx[size], permuteShift16);

    PTXChain chain7(2);
    chain7.ADD(xx[size+1], xx[size+1], carry);
    chain7.ADD(carry, zero, zero);
    chain7.end();

    #pragma unroll
    for(i=0;i<size;i++) {
      if(i>0) {
        PTXChain chain8(i+1);
        #pragma unroll
        for(j=0;j<size;j++)
          if(j<i)   // work around for the compiler problem
            chain8.XMADLH(xx[size+j+1], xx[i], n[size-i+j], xx[size+j+1]);
        chain8.ADD(carry, carry, zero);
        chain8.end();
      }

      PTXChain chain9(i, false, true);   // chain length might be i+1 or i+2, so leak the last carry
      #pragma unroll
      for(j=0;j<size;j++)
        if(j<i)  // work around
          chain9.XMADHL(xx[size+j+1], xx[i], n[size-i+j], xx[size+j+1]);
      chain9.end();

      if(i<size-1) {
        inliner.ADDC_CC(xx[size+i+1], xx[size+i+1], carry);
        inliner.ADDC(carry, zero, zero);
      }
      else
        inliner.ADDC(high, high, carry);
    }

    #pragma unroll
    for(i=size;i<2*size-1;i++)
      inliner.PERMUTE(xx[i], xx[i], xx[i+1], permuteShift16);
    inliner.PERMUTE(xx[2*size-1], xx[2*size-1], high, permuteShift16);
    inliner.PERMUTE(high, high, zero, permuteShift16);

    high=-high;
    #pragma unroll
    for(int index=0;index<size;index++)
      n[index]=n[index] & high;

    PTXChain chain10(size);
    #pragma unroll
    for(int index=0;index<size;index++)
      chain10.SUB(r[index], xx[size+index], n[index]);
    chain10.end();
  }
*/

  // FIX FIX FIX - this is the IMAD algorithm
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
