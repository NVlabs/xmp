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
    _predicate=0;
    _set=true;
  }

  __device__ __forceinline__ PTXInliner::PTXInliner(uint32_t predicate, bool set) {
    _predicate=predicate;
    _set=set;
  }

  __device__ __forceinline__ PTXInliner PTXInliner::pnot() {
    return PTXInliner(_predicate, !_set);
  }

  __device__ __forceinline__ void PTXInliner::nestStart() {
    asm volatile ("{");
  }

  __device__ __forceinline__ void PTXInliner::nestEnd() {
    asm volatile ("}");
  }

  __device__ __forceinline__ void PTXInliner::declarePredicate() {
    if(_predicate==0) {
    }
    else if(_predicate==1)
      asm volatile (".reg .pred %c1;");
    else if(_predicate==2)
      asm volatile (".reg .pred %c2;");
    else
      PTX_ERROR("PTXInliner: declareCondition - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::ADD(uint32_t& r, uint32_t& a, uint32_t& b) {
    uint32_t res;

    if(_predicate==0)
      asm volatile ("add.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    else if(_predicate==1 && _set) {
      res=r;
      asm volatile ("@%%c1 add.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==1 && !_set) {
      res=r;
      asm volatile ("@!%%c1 add.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && _set) {
      res=r;
      asm volatile ("@%%c2 add.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && !_set) {
      res=r;
      asm volatile ("@!%%c2 add.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else
      PTX_ERROR("PTXInliner: ADD - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::ADD_CC(uint32_t& r, uint32_t& a, uint32_t& b) {
    uint32_t res;

    if(_predicate==0)
      asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    else if(_predicate==1 && _set) {
      res=r;
      asm volatile ("@%%c1 add.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==1 && !_set) {
      res=r;
      asm volatile ("@!%%c1 add.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && _set) {
      res=r;
      asm volatile ("@%%c2 add.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && !_set) {
      res=r;
      asm volatile ("@!%%c2 add.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else
      PTX_ERROR("PTXInliner: ADD_CC - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::ADDC(uint32_t& r, uint32_t& a, uint32_t& b) {
    uint32_t res;

    if(_predicate==0)
      asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    else if(_predicate==1 && _set) {
      res=r;
      asm volatile ("@%%c1 addc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==1 && !_set) {
      res=r;
      asm volatile ("@!%%c1 addc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && _set) {
      res=r;
      asm volatile ("@%%c2 addc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && !_set) {
      res=r;
      asm volatile ("@!%%c2 addc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else
      PTX_ERROR("PTXInliner: ADDC - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::ADDC_CC(uint32_t& r, uint32_t& a, uint32_t& b) {
    uint32_t res;

    if(_predicate==0)
      asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    else if(_predicate==1 && _set) {
      res=r;
      asm volatile ("@%%c1 addc.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==1 && !_set) {
      res=r;
      asm volatile ("@!%%c1 addc.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && _set) {
      res=r;
      asm volatile ("@%%c2 addc.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && !_set) {
      res=r;
      asm volatile ("@!%%c2 addc.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else
      PTX_ERROR("PTXInliner: ADDC_CC - unsupported predicate");
  }


  __device__ __forceinline__ void PTXInliner::SUB(uint32_t& r, uint32_t& a, uint32_t& b) {
    uint32_t res;

    if(_predicate==0)
      asm volatile ("sub.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    else if(_predicate==1 && _set) {
      res=r;
      asm volatile ("@%%c1 sub.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==1 && !_set) {
      res=r;
      asm volatile ("@!%%c1 sub.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && _set) {
      res=r;
      asm volatile ("@%%c2 sub.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && !_set) {
      res=r;
      asm volatile ("@!%%c2 sub.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else
      PTX_ERROR("PTXInliner: SUB - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::SUB_CC(uint32_t& r, uint32_t& a, uint32_t& b) {
    uint32_t res;

    if(_predicate==0)
      asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    else if(_predicate==1 && _set) {
      res=r;
      asm volatile ("@%%c1 sub.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==1 && !_set) {
      res=r;
      asm volatile ("@!%%c1 sub.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && _set) {
      res=r;
      asm volatile ("@%%c2 sub.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && !_set) {
      res=r;
      asm volatile ("@!%%c2 sub.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else
      PTX_ERROR("PTXInliner: SUB_CC - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::SUBC(uint32_t& r, uint32_t& a, uint32_t& b) {
    uint32_t res;

    if(_predicate==0)
      asm volatile ("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    else if(_predicate==1 && _set) {
      res=r;
      asm volatile ("@%%c1 subc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==1 && !_set) {
      res=r;
      asm volatile ("@!%%c1 subc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && _set) {
      res=r;
      asm volatile ("@%%c2 subc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && !_set) {
      res=r;
      asm volatile ("@!%%c2 subc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else
      PTX_ERROR("PTXInliner: SUBC - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::SUBC_CC(uint32_t& r, uint32_t& a, uint32_t& b) {
    uint32_t res;

    if(_predicate==0)
      asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    else if(_predicate==1 && _set) {
      res=r;
      asm volatile ("@%%c1 subc.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==1 && !_set) {
      res=r;
      asm volatile ("@!%%c1 subc.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && _set) {
      res=r;
      asm volatile ("@%%c2 subc.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else if(_predicate==2 && !_set) {
      res=r;
      asm volatile ("@!%%c2 subc.cc.u32 %0, %1, %2;" : "+r"(res) : "r"(a), "r"(b));
      r=res;
    }
    else
      PTX_ERROR("PTXInliner: SUBC_CC - unsupported predicate");
  }


  __device__ __forceinline__ void PTXInliner::MULLO(uint32_t& r, uint32_t& a, uint32_t& b) {
    if(_predicate==0)
      asm volatile ("mul.lo.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    else
      PTX_ERROR("PTXInliner: MULLO - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::MULHI(uint32_t& r, uint32_t& a, uint32_t& b) {
    if(_predicate==0)
      asm volatile ("mul.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    else
      PTX_ERROR("PTXInliner: MULHI - unsupported predicate");
  }



  __device__ __forceinline__ void PTXInliner::MADLO(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    if(_predicate==0)
      asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: MADLO - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::MADLO_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    if(_predicate==0)
      asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: MADLO_CC - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::MADLOC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    if(_predicate==0)
      asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: MADLOC - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::MADLOC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    if(_predicate==0)
      asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: MADLOC_CC - unsupported predicate");
  }


  __device__ __forceinline__ void PTXInliner::MADHI(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    if(_predicate==0)
      asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: MADHI - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::MADWIDE(uint64_t& r, uint32_t& a, uint32_t& b, uint64_t& c) {
    if(_predicate==0)
      asm volatile ("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c));
    else
      PTX_ERROR("PTXInliner: MADWIDE - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::MADHI_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    if(_predicate==0)
      asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: MADHI_CC - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::MADHIC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    if(_predicate==0)
      asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: MADHIC - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::MADHIC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    if(_predicate==0)
      asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: MADHIC_CC - unsupported predicate");
  }


  __device__ __forceinline__ void PTXInliner::XMADLL(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %al, %bl;\n\t"
                    "add.u32       %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADLL - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLL_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %al, %bl;\n\t"
                    "add.cc.u32    %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADLL_CC - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLLC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %al, %bl;\n\t"
                    "addc.u32      %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADLLC - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLLC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c)  {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %al, %bl;\n\t"
                    "addc.cc.u32   %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADLLC_CC - unsupported predicate");
    HACKSAW_SYNC();
  }



  __device__ __forceinline__ void PTXInliner::XMADLH(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %al, %bh;\n\t"
                    "add.u32       %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADLH - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLH_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %al, %bh;\n\t"
                    "add.cc.u32    %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADLH_CC - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLHC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %al, %bh;\n\t"
                    "addc.u32      %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADLHC - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADLHC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c)  {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %al, %bh;\n\t"
                    "addc.cc.u32   %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADLHC_CC - unsupported predicate");
    HACKSAW_SYNC();
  }



  __device__ __forceinline__ void PTXInliner::XMADHL(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %ah, %bl;\n\t"
                    "add.u32       %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADHL - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHL_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %ah, %bl;\n\t"
                    "add.cc.u32    %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADHL_CC - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHLC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %ah, %bl;\n\t"
                    "addc.u32      %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADHLC - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHLC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c)  {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %ah, %bl;\n\t"
                    "addc.cc.u32   %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADHLC_CC - unsupported predicate");
    HACKSAW_SYNC();
  }



  __device__ __forceinline__ void PTXInliner::XMADHH(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %ah, %bh;\n\t"
                    "add.u32       %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADHH - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHH_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %ah, %bh;\n\t"
                    "add.cc.u32    %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADHH_CC - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHHC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %ah, %bh;\n\t"
                    "addc.u32      %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADHHC - unsupported predicate");
    HACKSAW_SYNC();
  }

  __device__ __forceinline__ void PTXInliner::XMADHHC_CC(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c)  {
    HACKSAW_SYNC();
    if(_predicate==0)
      asm volatile ("{\n\t"
                    ".reg .u16     %al, %ah, %bl, %bh;\n\t"
                    "mov.b32       {%al,%ah},%1;\n\t"
                    "mov.b32       {%bl,%bh},%2;\n\t"
                    "mul.wide.u16  %0, %ah, %bh;\n\t"
                    "addc.cc.u32   %0, %0, %3;\n\t"
                    "}" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: XMADHHC_CC - unsupported predicate");
    HACKSAW_SYNC();
  }


  __device__ __forceinline__ void PTXInliner::SETP_EQ(uint32_t& a, uint32_t& b) {
    if(_predicate==1)
      asm volatile ("setp.eq.u32 %%c1, %0, %1;" : : "r"(a), "r"(b));
    else if(_predicate==2)
      asm volatile ("setp.eq.u32 %%c1, %0, %1;" : : "r"(a), "r"(b));
    else
      PTX_ERROR("PTXInliner: SETP_EQ - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::SETP_NE(uint32_t& a, uint32_t& b) {
    if(_predicate==1)
      asm volatile ("setp.ne.u32 %%c1, %0, %1;" : : "r"(a), "r"(b));
    else if(_predicate==2)
      asm volatile ("setp.ne.u32 %%c1, %0, %1;" : : "r"(a), "r"(b));
    else
      PTX_ERROR("PTXInliner: SETP_NE - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::SELP(uint32_t& r, uint32_t& a, uint32_t& b) {
    if(_predicate==1)
      asm volatile ("selp.u32 %0, %1, %2, %%c1;" : "=r"(r) : "r"(a), "r"(b));
    if(_predicate==2)
      asm volatile ("selp.u32 %0, %1, %2, %%c2;" : "=r"(r) : "r"(a), "r"(b));
    else
      PTX_ERROR("PTXInliner: SELP - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::BFE(uint32_t& r, uint32_t& a, uint32_t &start, uint32_t &len) {
    if(_predicate==0)
      asm volatile ("bfe.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(start), "r"(len));
    else
      PTX_ERROR("PTXInliner: BFE - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::BFE_S(uint32_t& r, uint32_t& a, uint32_t &start, uint32_t &len) {
    if(_predicate==0)
      asm volatile ("bfe.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(start), "r"(len));
    else
      PTX_ERROR("PTXInliner: BFE - unsupported predicate");
  }


  __device__ __forceinline__ void PTXInliner::PERMUTE(uint32_t& r, uint32_t& a, uint32_t& b, uint32_t &c) {
    if(_predicate==0)
      asm volatile ("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: PERMUTE - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::SHF_L_WRAP(uint32_t& r, uint32_t& a, uint32_t& b, uint32_t &c) {
    if(_predicate==0)
      asm volatile ("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: SHF_L_WRAP - unsupported predicate");
  }

  __device__ __forceinline__ void PTXInliner::SHF_R_WRAP(uint32_t& r, uint32_t& a, uint32_t& b, uint32_t &c) {
    if(_predicate==0)
      asm volatile ("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    else
      PTX_ERROR("PTXInliner: SHF_R_WRAP - unsupported predicate");
  }
}