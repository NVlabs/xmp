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

#if defined(_WIN64) || defined(__LP64__)  //64 bit build
#define ASM_PTR "l"
#else
#define ASM_PTR "r"                       //32 bit build
#endif


namespace xmp {
  __device__ __forceinline__ bool overlap(Registers file1, Registers file2) {
    return file1.tag==file2.tag && file1.start + file1.length>file2.start && file1.start<file2.start + file2.length;
  }

  __device__ __forceinline__ RegMP::RegMP() {
    _lower.registers=NULL;
    _lower.tag=0;
    _lower.start=0;
    _lower.length=0;
    _upper.registers=NULL;
    _upper.tag=0;
    _upper.start=0;
    _upper.length=0;
  }

  __device__ __forceinline__ RegMP::RegMP(uint32_t *registers, uint32_t tag, uint32_t length) {
    if(length==0) RMP_ERROR("zero length not allowed");
    _lower.registers=registers;
    _lower.tag=tag;
    _lower.start=0;
    _lower.length=length;
    _upper.registers=NULL;
    _upper.tag=0;
    _upper.start=0;
    _upper.length=0;
  }

  __device__ __forceinline__ RegMP::RegMP(uint32_t *registers, uint32_t tag, uint32_t start, uint32_t length) {
    if(length==0) RMP_ERROR("zero length not allowed");
    _lower.registers=registers;
    _lower.tag=tag;
    _lower.start=start;
    _lower.length=length;
    _upper.registers=NULL;
    _upper.tag=0;
    _upper.start=0;
    _upper.length=0;
  }

  __device__ __forceinline__ RegMP::RegMP(RegMP mp, int32_t length) {
    if(length==0) RMP_ERROR("zero length not allowed");

    // positive length: LOWER
    if(length>0) {
      if(length>mp._lower.length + mp._upper.length) RMP_ERROR("length overrun");
      _lower.registers=mp._lower.registers;
      _lower.tag=mp._lower.tag;
      _lower.start=mp._lower.start;
      if(length<=mp._lower.length) {
        _lower.length=length;
        _upper.registers=NULL;
        _upper.tag=0;
        _upper.start=0;
        _upper.length=0;
      }
      else {
        _lower.length=mp._lower.length;
        _upper.registers=mp._upper.registers;
        _upper.tag=mp._upper.tag;
        _upper.start=mp._upper.start;
        _upper.length=length-mp._lower.length;
      }
    }
    // negative length: UPPER
    else if(length<0) {
      length=-length;
      if(length>mp._lower.length + mp._upper.length) RMP_ERROR("length overrun");
      if(length<=mp._upper.length) {
        _lower.registers=mp._upper.registers;
        _lower.tag=mp._upper.tag;
        _lower.start=mp._upper.start + mp._upper.length - length;
        _lower.length=length;
        _upper.registers=NULL;
        _upper.tag=0;
        _upper.start=0;
        _upper.length=0;
      }
      else {
        _lower.registers=mp._lower.registers;
        _lower.tag=mp._lower.tag;
        _lower.start=mp._lower.start + mp._lower.length + mp._upper.length - length;
        _lower.length=length - mp._upper.length;
        _upper.registers=mp._upper.registers;
        _upper.tag=mp._upper.tag;
        _upper.start=mp._upper.start;
        _upper.length=mp._upper.length;
      }
    }
  }

  __device__ __forceinline__ RegMP RegMP::concatenate(RegMP upper) {
    RegMP res;

    if(_upper.length!=0 || upper._upper.length!=0) RMP_ERROR("double concatenation not allowed");
    if(_lower.tag==upper._lower.tag && _lower.start + _lower.length==upper._lower.start) {
      res._lower.registers=_lower.registers;
      res._lower.tag=_lower.tag;
      res._lower.start=_lower.start;
      res._lower.length=_lower.length + upper._lower.length;
      res._upper.registers=NULL;
      res._upper.tag=0;
      res._upper.start=0;
      res._upper.length=0;
    }
    else {
      res._lower.registers=_lower.registers;
      res._lower.tag=_lower.tag;
      res._lower.start=_lower.start;
      res._lower.length=_lower.length;
      res._upper.registers=upper._lower.registers;
      res._upper.tag=upper._lower.tag;
      res._upper.start=upper._lower.start;
      res._upper.length=upper._lower.length;
    }
    return res;
  }

  __device__ __forceinline__ RegMP RegMP::lower(uint32_t length) {
    RegMP res;

    if(length==0) RMP_ERROR("lower() - zero length not allowed");
    if(length>0) {
      if(length>_lower.length + _upper.length) RMP_ERROR("lower() - length overrun");
      res._lower.registers=_lower.registers;
      res._lower.tag=_lower.tag;
      res._lower.start=_lower.start;
      if(length<=_lower.length) {
        res._lower.length=length;
        res._upper.registers=NULL;
        res._upper.tag=0;
        res._upper.start=0;
        res._upper.length=0;
      }
      else {
        res._lower.length=_lower.length;
        res._upper.registers=_upper.registers;
        res._upper.tag=_upper.tag;
        res._upper.start=_upper.start;
        res._upper.length=length-_lower.length;
      }
    }
    return res;
  }

  __device__ __forceinline__ RegMP RegMP::upper(uint32_t length) {
    RegMP res;

    if(length==0) RMP_ERROR("upper() - zero length not allowed");
    if(length>0) {
      if(length>_lower.length + _upper.length) RMP_ERROR("upper() - length overrun");
      if(length<=_upper.length) {
        res._lower.registers=_upper.registers;
        res._lower.tag=_upper.tag;
        res._lower.start=_upper.start + _upper.length - length;
        res._lower.length=length;
        res._upper.registers=NULL;
        res._upper.tag=0;
        res._upper.start=0;
        res._upper.length=0;
      }
      else {
        res._lower.registers=_lower.registers;
        res._lower.tag=_lower.tag;
        res._lower.start=_lower.start + _lower.length + _upper.length - length;
        res._lower.length=length - _upper.length;
        res._upper.registers=_upper.registers;
        res._upper.tag=_upper.tag;
        res._upper.start=_upper.start;
        res._upper.length=_upper.length;
      }
    }
    return res;
  }

  __device__ __forceinline__ bool RegMP::overlap(RegMP check) {
    return xmp::overlap(_lower, check._lower) || ::xmp::overlap(_lower, check._upper) || ::xmp::overlap(_upper, check._lower) || ::xmp::overlap(_upper, check._upper);
  }

  __device__ __forceinline__ bool RegMP::lowerAligned(RegMP check) {
    if(_lower.tag!=check._lower.tag || _lower.start!=check._lower.start || _lower.length<check._lower.length) return false;
    if(_lower.length>check._lower.length)
      return check._upper.length==0;
    else
     return _upper.tag==check._upper.tag && _upper.start==check._upper.start && _upper.length>=check._upper.length;
  }

  __device__ __forceinline__ bool RegMP::upperAligned(RegMP check) {
    if(_upper.length==0 && check._upper.length!=0) return false;
    if(_upper.length==0 && check._upper.length==0) {
      return _lower.tag==check._lower.tag && _lower.length<=check._lower.length && _lower.start + _lower.length==check._lower.start + check._lower.length;
    }
    else if(_upper.length!=0 && check._upper.length!=0) {
      return (_upper.tag==check._upper.tag && _upper.length==check._upper.length && _upper.start==check._upper.start) &&
             (_lower.tag==check._lower.tag && _lower.length<=check._lower.length && _lower.start + _lower.length==check._lower.start + check._lower.length);
    }
    else {
      return _upper.tag==check._lower.tag && _upper.length<=check._lower.length && _upper.start + _upper.length==check._lower.start + check._lower.length;
    }
  }

  __device__ __forceinline__ int RegMP::alignment(RegMP check) {
    if(!overlap(check))
      return RMP_NO_OVERLAP;
    else if(lowerAligned(check))
      return RMP_LOWER_ALIGNED;
    else if(upperAligned(check))
      return RMP_UPPER_ALIGNED;
    else
      return RMP_UNALIGNED;
  }

/*
  __device__ __forceinline__ void RegMP::set(RegMP mp) {
     _lower.registers=mp._lower.registers;
     _lower.tag=mp._lower.tag;
     _lower.start=mp._lower.start;
     _lower.length=mp._lower.length;
     _upper.registers=mp._upper.registers;
     _upper.tag=mp._upper.tag;
     _upper.start=mp._upper.start;
     _upper.length=mp._upper.length;
  }
*/

  __device__ __forceinline__ uint32_t RegMP::length() {
     return _lower.length + _upper.length;
  }

  __device__ __forceinline__ uint32_t& RegMP::operator[] (const int index) {
    if(index<0 || index>=_lower.length + _upper.length) RMP_ERROR("index out of range");
    if(index<_lower.length)
      return _lower.registers[_lower.start + index];
    else
      return _upper.registers[_upper.start + index - _lower.length];
  }

  __device__ __forceinline__ uint32_t *RegMP::registers() {
    if(_upper.length!=0) RMP_ERROR("concatenated registers");
    return _lower.registers + _lower.start;
  }

  __device__ __forceinline__ void RegMP::print() {
    #pragma unroll
    for(int index=length()-1;index>=0;index--)
      if(index<_lower.length)
        printf("%08X", _lower.registers[_lower.start + index]);
      else
        printf("%08X", _upper.registers[_upper.start + index - _lower.length]);
    printf("\n");
  }

  __device__ __forceinline__ void RegMP::print(const char *text) {
    printf("%s=", text);
    #pragma unroll
    for(int index=length()-1;index>=0;index--)
      if(index<_lower.length)
        printf("%08X", _lower.registers[_lower.start + index]);
      else
        printf("%08X", _upper.registers[_upper.start + index - _lower.length]);
    printf("\n");
  }


  __device__ __forceinline__ void instruction_store(int instruction, uint32_t *address, uint32_t& a) {
    if(instruction==RMP_ST_GLOBAL)
      asm volatile ("st.global.u32  [%0],%1;" : : ASM_PTR(address), "r"(a));
    else
      RMP_ERROR("instruction_store() - unknown instruction");
  }

  __device__ __forceinline__ void instruction_store(int instruction, uint32_t *address, uint32_t& a, uint32_t& b) {
    if(instruction==RMP_ST_GLOBAL_V2)
      asm volatile ("st.global.v2.u32  [%0],{%1,%2};" : : ASM_PTR(address), "r"(a), "r"(b));
    else
      RMP_ERROR("instruction_store() - unknown instruction");
  }

  __device__ __forceinline__ void instruction_store(int instruction, uint32_t *address, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    if(instruction==RMP_ST_GLOBAL_V4)
      asm volatile ("st.global.v4.u32  [%0],{%1,%2,%3,%4};" : : ASM_PTR(address), "r"(a), "r"(b), "r"(c), "r"(d));
    else
      RMP_ERROR("instruction_store() - unknown instruction");
  }
}