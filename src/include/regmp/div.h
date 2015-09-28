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
  /*
    uint64_t invert(uint32_t x) {
      uint32_t x0, q0, q1;
      uint32_t l, h, p;
      uint64_t q, prod;

      x0=x>>16;
      q0=0xFFFFFFFF/x0;
      while(HI(q0, x)>=0x00010000)
        q0--;

      l=0-LO(q0, x);
      h=0x10000-HI(q0, x)-((l!=0)?1:0);
      p=(h<<16) + (l>>16);

      q1=p*q0;
      p=(q0<<16)+(q1>>16);
      q=0x100000000L + (uint64_t)p;

      l=LO(p, x);
      h=HI(p, x)+x;
      while(h!=0) {
        h=h+((l+x<l)?1:0);
        l=l+x;
        q=q+1;
      };
      return q;
    }
  */


  //
  //   invert(x) computes ceiling(1<<64 / x) - 2^32 - 1, when 0x80000000 <= x <= 0xFFFFFFFF
  //   using algorithm 3.5 from Modern Computer Arithmetic, by Brent and Zimmermann
  //

  __device__ __forceinline__ void _invert(uint32_t& inv, uint32_t x) {
    PTXInliner inliner;
    uint32_t   x0, q0, q1;
    uint32_t   l, h, p;
    uint32_t   zero=0, b16=0x10000, permuteShift16=0x5432;

    x0=x>>16;
    q0=0xFFFFFFFF/x0;
    while(1) {
      inliner.MULHI(h, q0, x);
      if(h<0x10000)
        break;
      q0--;
    }
    inliner.MULLO(l, q0, x);

    inliner.SUB_CC(l, zero, l);
    inliner.SUBC(h, b16, h);
    inliner.PERMUTE(p, l, h, permuteShift16);

    q1=q0*p;
    inliner.PERMUTE(inv, q1, q0, permuteShift16);

    // refine the estimate
    inliner.MULLO(l, inv, x);
    inliner.MADHI(h, inv, x, x);
    while(h!=0) {
      inv++;
      inliner.ADD_CC(l, l, x);
      inliner.ADDC(h, h, zero);
    }
    inv--;
  }

  __device__ __forceinline__ void _barrett_invert(uint32_t& inv, RegMP x) {
    PTXInliner inliner;
    uint32_t   l, h, zero=0;

    _invert(inv, x[x.length()-1]);

    // refine the estimate
    while(1) {
      inliner.MADHI_CC(l, inv, x[0], x[0]);
      inliner.ADDC(h, zero, zero);

      #pragma unroll
      for(int index=1;index<x.length();index++) {
        inliner.MADLO_CC(l, inv, x[index], l);
        inliner.MADHIC_CC(l, inv, x[index], h);
        inliner.ADDC(h, zero, zero);
        inliner.ADD_CC(l, l, x[index]);
        inliner.ADDC(h, h, zero);
      }

      if(h==0)
        break;
      inv--;
    }
  }

/*
  __device__ __forceinline__ int _bit_left_shift(RegMP x, int amount) {
    PTXInliner inliner;
    uint32_t   shiftBits=amount;

    if(amount==0)
      return 0;
#if __CUDA_ARCH__<350
    #pragma unroll
    for(int index=x.length()-1;index>=1;index--)
      x[index]=(x[index]<<shiftBits) | (x[index-1]>>32-shiftBits);
    x[0]=x[0]<<shiftBits;
#else
    #pragma unroll
    for(int index=x.length()-1;index>=1;index--)
      inliner.SHF_L_WRAP(x[index], x[index-1], x[index], shiftBits);
    x[0]=x[0]<<shiftBits;
#endif
    return amount;
  }

  __device__ __forceinline__ int _bit_right_shift(RegMP x, int amount) {
    PTXInliner inliner;
    uint32_t   shiftBits=amount;

#if __CUDA__ARCH__<350
    #pragma unroll
    for(int index=0;index<x.length()-1;index++)
      x[index]=(x[index]>>shiftBits) | (x[index+1]<<32-shiftBits);
    x[x.length()-1]=x[x.length()-1]>>shiftBits;
#else
    #pragma unroll
    for(int index=0;index<x.length()-1;index++)
      inliner.SHF_R_WRAP(x[index], x[index], x[index+1], shiftBits);
    x[x.length()-1]=x[x.length()-1]>>shiftBits;
#endif
    return -amount;
  }

  __device__ __forceinline__ int _word_right_shift(RegMP x, int amount) {
    if(x.length()>4) {
      while(amount>=4) {
        #pragma unroll
        for(int index=4;index<x.length();index++)
          x[index-4]=x[index];
        x[x.length()-1]=0;
        x[x.length()-2]=0;
        x[x.length()-3]=0;
        x[x.length()-4]=0;
        amount=amount-4;
      }
    }
    if(amount>=2) {
      #pragma unroll
      for(int index=2;index<x.length();index++)
        x[index-2]=x[index];
      x[x.length()-1]=0;
      x[x.length()-2]=0;
      amount=amount-2;
    }
    if(amount>=1) {
      #pragma unroll
      for(int index=1;index<x.length();index++)
        x[index-1]=x[index];
      x[x.length()-1]=0;
    }
    return -amount*32;
  }

  __device__ __forceinline__ int _normalize(RegMP x) {
    uint32_t bits=0;

    if(x.length()>4) {
      while(bits+128<(x.length()<<5) && (x[x.length()-1] | x[x.length()-2] | x[x.length()-3] | x[x.length()-4])==0) {
        #pragma unroll
        for(int index=x.length()-1;index>=4;index--)
          x[index]=x[index-4];
        x[0]=0;
        x[1]=0;
        x[2]=0;
        x[3]=0;
        bits+=128;
      }
    }
    if(bits+64<(x.length()<<5) && (x[x.length()-1] | x[x.length()-2])==0) {
      #pragma unroll
      for(int index=x.length()-1;index>=2;index--)
        x[index]=x[index-2];
      x[0]=0;
      x[1]=0;
      bits+=64;
    }
    if(bits+32<(x.length()<<5) && x[x.length()-1]==0) {
      #pragma unroll
      for(int index=x.length()-1;index>=1;index--)
        x[index]=x[index-1];
      x[0]=0;
      bits+=32;
    }
    if(x[x.length()-1]==0)
      return -1;

    return bits + _bit_left_shift(x, __clz(x[x.length()-1]));
  }

  __device__ __forceinline__ void _mul_k(RegMP r, uint32_t k, RegMP x) {
    PTXInliner inliner;
    uint32_t   temp, l, h, zero=0;
    int        index;

    temp=x[0];
    inliner.MULLO(r[0], temp, k);
    inliner.MULHI(h, temp, k);
    #pragma unroll
    for(index=1;index<x.length();index++) {
      inliner.ADD_CC(l, h, temp);
      inliner.ADDC(h, zero, zero);
      temp=x[index];
      inliner.MADLO_CC(r[index], temp, k, l);
      inliner.MADHIC(h, temp, k, h);
    }
    if(r.length()>index)
      inliner.ADD_CC(r[index], h, temp);
    #pragma unroll
    for(index=x.length()+1;index<r.length();index++)
      inliner.ADDC_CC(r[index], zero, zero);
  }

  __device__ __forceinline__ void _accumulate(RegMP acc, RegMP x) {
    int      index;
    uint32_t zero=0;

    if(x.length()>=acc.length()) RMP_ERROR("_accumulate(): size mismatch");

    PTXChain chain1(acc.length());
    #pragma unroll
    for(index=0;index<x.length();index++)
      chain1.ADD(acc[index], acc[index], x[index]);
    #pragma unroll
    for(index=x.length();index<acc.length();index++)
      chain1.ADD(acc[index], acc[index], zero);
    chain1.end();
  }

  __device__ __forceinline__ void _decumulate(RegMP acc, RegMP x) {
    int      index;
    uint32_t zero=0;

    if(x.length()>=acc.length()) RMP_ERROR("_decumulate(): size mismatch");

    PTXChain chain1(acc.length());
    #pragma unroll
    for(index=0;index<x.length();index++)
      chain1.SUB(acc[index], acc[index], x[index]);
    #pragma unroll
    for(index=x.length();index<acc.length();index++)
      chain1.SUB(acc[index], acc[index], zero);
    chain1.end();
  }
  */

  __device__ __forceinline__ void p(RegMP x) {
    #pragma unroll
    for(int index=x.length()-1;index>=0;index--)
      printf("%08X", x[index]);
    printf("\n");
  }

/*

  __device__ __forceinline__ bool div_setup(RegMP d, RegMP temp, uint32_t& k, int32_t& shifted) {
    // temp must be two words larger than d
    if(temp.length()!=d.length()+2) RMP_ERROR("div_setup() - temp length mismatch");

    shifted=0;
    shifted=_normalize(d);
    if(shifted==-1)
      return false;

    _invert(k, d[d.length()-1]);
    _mul_k(temp, k, d);
    // FIX FIX FIX - only need 3x?
    if(temp[temp.length()-1]==1) {
      _decumulate(temp, d);
      k--;
    }
    if(temp[temp.length()-1]==1) {
      _decumulate(temp, d);
      k--;
    }
    if(temp[temp.length()-1]==1) {
      _decumulate(temp, d);
      k--;
    }
    if(temp[temp.length()-1]==1) {
      _decumulate(temp, d);
      k--;
    }
    _accumulate(temp, d);
    k++;

    #pragma unroll
    for(int index=0;index<d.length();index++)
      d[index]=temp[index];

    return true;
  }

  __device__ __forceinline__ void div(RegMP xx, RegMP d, uint32_t k, int32_t shifted) {
    PTXInliner inliner;
    uint32_t   correction, temp, zero=0, ff=0xFFFFFFFF;
    uint32_t   delta=xx.length()-d.length()-2;

    // Call div_setup before this routine

    // top three words of XX must be 0

    _bit_left_shift(xx, shifted & 0x1F);

    if(k==0) {
      #pragma unroll
      for(int index=xx.length()-1;index>=1;index--)
        xx[index]=xx[index-1];

      PTXChain chain1(xx.length());
      #pragma unroll
      for(int index=0;index<xx.length();index++)
        chain1.ADD(xx[index], xx[index], xx[index]);
      chain1.end();
    }
    else
      _mul_k(xx, k, xx);

    for(int index=0;index<=delta+(shifted>>5);index++) {
      inliner.ADD_CC(temp, ff, ff);
      PTXChain chain1(d.length()+2, true, true);
      #pragma unroll
      for(int word=0;word<d.length();word++) {
        inliner.MULLO(temp, d[word], xx[xx.length()-1]);
        chain1.SUB(xx[delta+word], xx[delta+word], temp);
      }
      chain1.SUB(xx[delta+d.length()], xx[delta+d.length()], zero);
      chain1.SUB(correction, zero, zero);
      chain1.end();

      inliner.ADD_CC(temp, ff, ff);
      PTXChain chain2(d.length()+1, true, true);
      #pragma unroll
      for(int word=0;word<d.length();word++) {
        inliner.MULHI(temp, d[word], xx[xx.length()-1]);
        chain2.SUB(xx[delta+word+1], xx[delta+word+1], temp);
      }
      chain2.SUB(correction, correction, zero);
      chain2.end();

      PTXChain chain3(d.length()+1);
      #pragma unroll
      for(int word=0;word<d.length();word++) {
        temp=correction & d[word];
        chain3.ADD(xx[delta+word], xx[delta+word], temp);
      }
      chain3.ADD(xx[delta+d.length()], xx[delta+d.length()], zero);
      chain3.end();

      temp=xx[xx.length()-1]+correction;
      #pragma unroll
      for(int word=xx.length()-1;word>=1;word--)
        xx[word]=xx[word-1];
      xx[0]=temp;
    }

    temp=delta+(shifted>>5);
    #pragma unroll
    for(int word=xx.length()-1;word>=0;word--)
      xx[word]=(word<=temp) ? xx[word] : 0;
  }
*/

  __device__ __forceinline__ bool rem(RegMP xx, RegMP d) {
    PTXInliner inliner;
    uint32_t   k, q, temp, correction, zero=0, ff=0xFFFFFFFF;
    uint32_t   xx_len=xx.length(), d_len=d.length(), delta=xx_len-d_len-1;
    int32_t    shifted;

    // xx must be two words longer than the dividend, both top words must be set to zero

    shifted=count_leading_zeros(d);
    if(shifted==d.length()*32)
      return false;
    shift_left_words(d, d, shifted>>5);
    shift_left_bits(d, d, shifted & 0x1F);
    _barrett_invert(k, d);

    shift_left_bits(xx, xx, shifted & 0x1F);

    for(int index=0;index<=delta+(shifted>>5);index++) {
      correction=0;
      if(xx[xx_len-1]>d[d_len-1] || (xx[xx_len-1]==d[d_len-1] && xx[xx_len-2]>=d[d_len-2])) {
        PTXChain chain1(d_len+1);
        #pragma unroll
        for(int word=0;word<d_len;word++)
          chain1.SUB(xx[delta+word+1], xx[delta+word+1], d[word]);
        chain1.SUB(correction, zero, zero);
        chain1.end();
      }
      if(correction==0) {
        uint32_t t0, t1;

        // k is an approximation of the inverse.
        // use a Barrett reduction

        inliner.MADHI(q, k, xx[xx_len-1], xx[xx_len-1]);
        q=q-1;

        // improve q
        do {
          q++;
          if(q==0xFFFFFFFF)
            break;
          inliner.MULLO(t0, q, d[d_len-1]);
          inliner.MADHI_CC(t0, q, d[d_len-2], t0);
          inliner.MADHIC(t1, q, d[d_len-1], zero);
        } while(t1<xx[xx_len-1] || (t1==xx[xx_len-1] && t0<xx[xx_len-2]));

        inliner.ADD_CC(temp, ff, ff);
        PTXChain chain1(d.length()+2, true, true);
        #pragma unroll
        for(int word=0;word<d.length();word++) {
          inliner.MULLO(temp, d[word], q);
          chain1.SUB(xx[delta+word], xx[delta+word], temp);
        }
        chain1.SUB(xx[delta+d.length()], xx[delta+d.length()], zero);
        chain1.SUB(correction, zero, zero);
        chain1.end();

        inliner.ADD_CC(temp, ff, ff);
        PTXChain chain2(d.length()+1, true, true);
        #pragma unroll
        for(int word=0;word<d.length();word++) {
          inliner.MULHI(temp, d[word], q);
          chain2.SUB(xx[delta+word+1], xx[delta+word+1], temp);
        }
        chain2.SUB(correction, correction, zero);
        chain2.end();
      }

      PTXChain chain3(d.length()+1);
      #pragma unroll
      for(int word=0;word<d.length();word++) {
        temp=correction & d[word];
        chain3.ADD(xx[delta+word], xx[delta+word], temp);
      }
      chain3.ADD(xx[delta+d.length()], xx[delta+d.length()], zero);
      chain3.end();

      #pragma unroll
      for(int word=xx.length()-1;word>=1;word--)
        xx[word]=xx[word-1];
      xx[0]=0;
    }

    #pragma unroll
    for(int word=0;word<d.length();word++)
      xx[word]=xx[xx_len-d_len+word];

    #pragma unroll
    for(int word=d_len;word<xx.length();word++)
      xx[word]=0;

    shift_right_bits(xx.lower(d_len), xx.lower(d_len), shifted & 0x1F);
    shift_right_words(xx.lower(d_len), xx.lower(d_len), shifted>>5);

    return true;
  }

  __device__ __forceinline__ bool div(RegMP xx, RegMP d) {
    PTXInliner inliner;
    uint32_t   k, q, temp, correction, zero=0, ff=0xFFFFFFFF;
    uint32_t   xx_len=xx.length(), d_len=d.length(), delta=xx_len-d_len-1;
    int32_t    shifted;

    // xx must be two words longer than the dividend, both top words must be set to zero

    shifted=count_leading_zeros(d);
    if(shifted==d.length()*32)
      return false;
    shift_left_words(d, d, shifted>>5);
    shift_left_bits(d, d, shifted & 0x1F);
    _barrett_invert(k, d);

    shift_left_bits(xx, xx, shifted & 0x1F);

    for(int index=0;index<=delta+(shifted>>5);index++) {
      correction=0;
      q=0;
      if(d[d_len-1]<xx[xx_len-1] || (d[d_len-1]==xx[xx_len-1] && d[d_len-2]<=xx[xx_len-2])) {
        PTXChain chain1(d_len+1);
        #pragma unroll
        for(int word=0;word<d_len;word++)
          chain1.SUB(xx[delta+word+1], xx[delta+word+1], d[word]);
        chain1.SUB(correction, zero, zero);
        chain1.end();
        xx[0]++;
      }
      if(correction==0) {
        uint32_t t0, t1;

        // k is an approximation of the inverse.
        // use a Barrett reduction

        inliner.MADHI(q, k, xx[xx_len-1], xx[xx_len-1]);
        q=q-1;

        // improve q
        do {
          q++;
          if(q==0xFFFFFFFF)
            break;
          inliner.MULLO(t0, q, d[d_len-1]);
          inliner.MADHI_CC(t0, q, d[d_len-2], t0);
          inliner.MADHIC(t1, q, d[d_len-1], zero);
        } while(t1<xx[xx_len-1] || (t1==xx[xx_len-1] && t0<xx[xx_len-2]));
        inliner.ADD_CC(temp, ff, ff);
        PTXChain chain1(d.length()+2, true, true);
        #pragma unroll
        for(int word=0;word<d.length();word++) {
          inliner.MULLO(temp, d[word], q);
          chain1.SUB(xx[delta+word], xx[delta+word], temp);
        }
        chain1.SUB(xx[delta+d.length()], xx[delta+d.length()], zero);
        chain1.SUB(correction, zero, zero);
        chain1.end();

        inliner.ADD_CC(temp, ff, ff);
        PTXChain chain2(d.length()+1, true, true);
        #pragma unroll
        for(int word=0;word<d.length();word++) {
          inliner.MULHI(temp, d[word], q);
          chain2.SUB(xx[delta+word+1], xx[delta+word+1], temp);
        }
        chain2.SUB(correction, correction, zero);
        chain2.end();
      }

      PTXChain chain3(d.length()+1);
      #pragma unroll
      for(int word=0;word<d.length();word++) {
        temp=correction & d[word];
        chain3.ADD(xx[delta+word], xx[delta+word], temp);
      }
      chain3.ADD(xx[delta+d.length()], xx[delta+d.length()], zero);
      chain3.end();

      #pragma unroll
      for(int word=xx.length()-1;word>=1;word--)
        xx[word]=xx[word-1];
      if(q==0 && correction!=0)
        xx[1]--;
      xx[0]=q+correction;
    }

    temp=delta+(shifted>>5);
    #pragma unroll
    for(int word=xx.length()-1;word>=0;word--)
      xx[word]=(word<=temp) ? xx[word] : 0;
    return false;
  }

  __device__ __forceinline__ bool div_rem(RegMP xx, RegMP d) {
    PTXInliner inliner;
    uint32_t   k, q, temp, correction, zero=0, ff=0xFFFFFFFF;
    uint32_t   xx_len=xx.length(), d_len=d.length(), delta=xx_len-d_len-1;
    int32_t    shifted;

    // xx must be two words longer than the dividend, both top words must be set to zero

    shifted=count_leading_zeros(d);
    if(shifted==d.length()*32)
      return false;
    shift_left_words(d, d, shifted>>5);
    shift_left_bits(d, d, shifted & 0x1F);
    _barrett_invert(k, d);

    shift_left_bits(xx, xx, shifted & 0x1F);

    for(int index=0;index<=delta+(shifted>>5);index++) {
      correction=0;
      q=0;
      if(d[d_len-1]<xx[xx_len-1] || (d[d_len-1]==xx[xx_len-1] && d[d_len-2]<=xx[xx_len-2])) {
        PTXChain chain1(d_len+1);
        #pragma unroll
        for(int word=0;word<d_len;word++)
          chain1.SUB(xx[delta+word+1], xx[delta+word+1], d[word]);
        chain1.SUB(correction, zero, zero);
        chain1.end();
        xx[0]++;
      }
      if(correction==0) {
        uint32_t t0, t1;

        // k is an approximation of the inverse.
        // use a Barrett reduction

        inliner.MADHI(q, k, xx[xx_len-1], xx[xx_len-1]);
        q=q-1;

        // improve q
        do {
          q++;
          if(q==0xFFFFFFFF)
            break;
          inliner.MULLO(t0, q, d[d_len-1]);
          inliner.MADHI_CC(t0, q, d[d_len-2], t0);
          inliner.MADHIC(t1, q, d[d_len-1], zero);
        } while(t1<xx[xx_len-1] || (t1==xx[xx_len-1] && t0<xx[xx_len-2]));
        inliner.ADD_CC(temp, ff, ff);
        PTXChain chain1(d.length()+2, true, true);
        #pragma unroll
        for(int word=0;word<d.length();word++) {
          inliner.MULLO(temp, d[word], q);
          chain1.SUB(xx[delta+word], xx[delta+word], temp);
        }
        chain1.SUB(xx[delta+d.length()], xx[delta+d.length()], zero);
        chain1.SUB(correction, zero, zero);
        chain1.end();

        inliner.ADD_CC(temp, ff, ff);
        PTXChain chain2(d.length()+1, true, true);
        #pragma unroll
        for(int word=0;word<d.length();word++) {
          inliner.MULHI(temp, d[word], q);
          chain2.SUB(xx[delta+word+1], xx[delta+word+1], temp);
        }
        chain2.SUB(correction, correction, zero);
        chain2.end();
      }

      PTXChain chain3(d.length()+1);
      #pragma unroll
      for(int word=0;word<d.length();word++) {
        temp=correction & d[word];
        chain3.ADD(xx[delta+word], xx[delta+word], temp);
      }
      chain3.ADD(xx[delta+d.length()], xx[delta+d.length()], zero);
      chain3.end();

      #pragma unroll
      for(int word=xx.length()-1;word>=1;word--)
        xx[word]=xx[word-1];
      if(q==0 && correction!=0)
        xx[1]--;
      xx[0]=q+correction;
    }

    #pragma unroll
    for(int word=0;word<d.length();word++)
      d[word]=xx[xx_len-d_len+word];

    shift_right_bits(d, d, shifted & 0x1F);
    shift_right_words(d, d, shifted>>5);

    temp=delta+(shifted>>5);
    #pragma unroll
    for(int word=xx.length()-1;word>=0;word--)
      xx[word]=(word<=temp) ? xx[word] : 0;
    return false;
  }
}
