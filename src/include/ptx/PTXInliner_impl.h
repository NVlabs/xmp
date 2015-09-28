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
  __device__ __forceinline__ PTXInliner::PTXInliner() {
  }

  __device__ __forceinline__ void PTXInliner::ADD(uint32_t& r, uint32_t& a, uint32_t& b) {
    asm volatile ("add.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  }

  __device__ __forceinline__ void PTXInliner::ADD_CC(uint32_t& r, uint32_t& a, uint32_t& b) {
    asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  }

  __device__ __forceinline__ void PTXInliner::ADDC(uint32_t& r, uint32_t& a, uint32_t& b) {
    asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  }

  __device__ __forceinline__ void PTXInliner::ADDC_CC(uint32_t& r, uint32_t& a, uint32_t& b) {
    asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  }


  __device__ __forceinline__ void PTXInliner::SUB(uint32_t& r, uint32_t& a, uint32_t& b) {
    asm volatile ("sub.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  }

  __device__ __forceinline__ void PTXInliner::SUB_CC(uint32_t& r, uint32_t& a, uint32_t& b) {
    asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  }

  __device__ __forceinline__ void PTXInliner::SUBC(uint32_t& r, uint32_t& a, uint32_t& b) {
    asm volatile ("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  }

  __device__ __forceinline__ void PTXInliner::SUBC_CC(uint32_t& r, uint32_t& a, uint32_t& b) {
    asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  }


  __device__ __forceinline__ void PTXInliner::MULLO(uint32_t& r, uint32_t& a, uint32_t& b) {
    asm volatile ("mul.lo.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  }

  __device__ __forceinline__ void PTXInliner::MULHI(uint32_t& r, uint32_t& a, uint32_t& b) {
    asm volatile ("mul.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  }



  __device__ __forceinline__ void PTXInliner::MADLO(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }

  __device__ __forceinline__ void PTXInliner::MADLO_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }

  __device__ __forceinline__ void PTXInliner::MADLOC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }

  __device__ __forceinline__ void PTXInliner::MADLOC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }


  __device__ __forceinline__ void PTXInliner::MADHI(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }

  __device__ __forceinline__ void PTXInliner::MADHI_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }

  __device__ __forceinline__ void PTXInliner::MADHIC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }

  __device__ __forceinline__ void PTXInliner::MADHIC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }


  __device__ __forceinline__ void PTXInliner::XMADLL(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %al, %bl;\n\t"
                  "add.u32       %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLL_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %al, %bl;\n\t"
                  "add.cc.u32    %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLLC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %al, %bl;\n\t"
                  "addc.u32      %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLLC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c)  {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %al, %bl;\n\t"
                  "addc.cc.u32   %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }



  __device__ __forceinline__ void PTXInliner::XMADLH(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %al, %bh;\n\t"
                  "add.u32       %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLH_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %al, %bh;\n\t"
                  "add.cc.u32    %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLHC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %al, %bh;\n\t"
                  "addc.u32      %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLHC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c)  {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %al, %bh;\n\t"
                  "addc.cc.u32   %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }



  __device__ __forceinline__ void PTXInliner::XMADHL(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %ah, %bl;\n\t"
                  "add.u32       %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHL_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %ah, %bl;\n\t"
                  "add.cc.u32    %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHLC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %ah, %bl;\n\t"
                  "addc.u32      %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHLC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c)  {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %ah, %bl;\n\t"
                  "addc.cc.u32   %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }



  __device__ __forceinline__ void PTXInliner::XMADHH(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %ah, %bh;\n\t"
                  "add.u32       %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHH_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %ah, %bh;\n\t"
                  "add.cc.u32    %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHHC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %ah, %bh;\n\t"
                  "addc.u32      %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHHC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c)  {
    HACKSAW_SYNC();
    asm volatile ("{\n\t"
                  ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                  "mov.b32       {%al,%ah},%1;\n\t"
                  "mov.b32       {%bl,%bh},%2;\n\t"
                  "mul.wide.u16  %0, %ah, %bh;\n\t"
                  "addc.cc.u32   %0, %0, %3;\n\t"
                  "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    HACKSAW_SYNC();
  }



  __device__ __forceinline__ void PTXInliner::PERMUTE(uint32_t& r, uint32_t& a, uint32_t& b, uint32_t &c) {
    asm volatile ("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }

  __device__ __forceinline__ void PTXInliner::SHF_L_WRAP(uint32_t& r, uint32_t& a, uint32_t& b, uint32_t &c) {
    asm volatile ("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }

  __device__ __forceinline__ void PTXInliner::SHF_R_WRAP(uint32_t& r, uint32_t& a, uint32_t& b, uint32_t &c) {
    asm volatile ("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  }
}