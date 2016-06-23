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
  __device__ __forceinline__ PTXChain::PTXChain() {
    _predicate=0;
    _set=true;
    _size=-1;
    _position=-1;
  }

  __device__ __forceinline__ PTXChain::PTXChain(int32_t size) {
    _predicate=0;
    _set=true;
    _size=size;
    _position=0;
    _carryIn=false;
    _carryOut=false;
  }

  __device__ __forceinline__ PTXChain::PTXChain(int32_t size, bool carryIn, bool carryOut) {
    _predicate=0;
    _set=true;
    _size=size;
    _position=0;
    _carryIn=carryIn;
    _carryOut=carryOut;
  }

  __device__ __forceinline__ PTXChain::PTXChain(PTXInliner inliner) {
    _predicate=inliner._predicate;
    _set=inliner._set;
    _size=-1;
    _position=-1;
  }

  __device__ __forceinline__ PTXChain::PTXChain(PTXInliner inliner, int32_t size) {
    _predicate=inliner._predicate;
    _set=inliner._set;
    _size=size;
    _position=0;
    _carryIn=false;
    _carryOut=false;
  }

  __device__ __forceinline__ PTXChain::PTXChain(PTXInliner inliner, int32_t size, bool carryIn, bool carryOut) {
    _predicate=inliner._predicate;
    _set=inliner._set;
    _size=size;
    _position=0;
    _carryIn=carryIn;
    _carryOut=carryOut;
  }

  __device__ __forceinline__ void PTXChain::start(int32_t size) {
    if(_size!=-1) PTX_ERROR("PTXChain.start() - previous chain not ended");
    _size=size;
    _position=0;
    _carryIn=false;
    _carryOut=false;
  }

  __device__ __forceinline__ void PTXChain::start(int32_t size, bool carryIn, bool carryOut) {
    if(_size!=-1) PTX_ERROR("PTXChain.start() - previous chain not ended");
    _size=size;
    _position=0;
    _carryIn=carryIn;
    _carryOut=carryOut;
  }

  __device__ __forceinline__ void PTXChain::end() {
    if(_position<_size) PTX_ERROR("PTXChain.end() - chain too short");
    if(_position>_size) PTX_ERROR("PTXChain.end() - chain too long");
    _size=-1;
  }

  __device__ __forceinline__ void PTXChain::ADD(uint32_t& r, uint32_t& a, uint32_t& b) {
    PTXInliner inliner;

    if(_size==-1) PTX_ERROR("PTXChain.ADD() - chain not started");
    _position++;

    if(_position==1 && _size==1 && !_carryIn && !_carryOut)
      inliner.ADD(r, a, b);
    else if(_position==1 && !_carryIn)
      inliner.ADD_CC(r, a, b);
    else if(_position<_size || _carryOut)
      inliner.ADDC_CC(r, a, b);
    else
      inliner.ADDC(r, a, b);
  }

  __device__ __forceinline__ void PTXChain::SUB(uint32_t& r, uint32_t& a, uint32_t& b) {
    PTXInliner inliner;

    if(_size==-1) PTX_ERROR("PTXChain.SUB() - chain not started");
    _position++;

    if(_position==1 && _size==1 && !_carryIn && !_carryOut)
      inliner.SUB(r, a, b);
    else if(_position==1 && !_carryIn)
      inliner.SUB_CC(r, a, b);
    else if(_position<_size || _carryOut)
      inliner.SUBC_CC(r, a, b);
    else
      inliner.SUBC(r, a, b);
  }

  __device__ __forceinline__ void PTXChain::MADLO(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    PTXInliner inliner;

    if(_size==-1) PTX_ERROR("PTXChain.MADLO() - chain not started");
    _position++;

    if(_position==1 && _size==1 && !_carryIn && !_carryOut)
      inliner.MADLO(r, a, b, c);
    else if(_position==1 && !_carryIn)
      inliner.MADLO_CC(r, a, b, c);
    else if(_position<_size || _carryOut)
      inliner.MADLOC_CC(r, a, b, c);
    else
     inliner.MADLOC(r, a, b, c);
  }

  __device__ __forceinline__ void PTXChain::MADHI(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    PTXInliner inliner;

    if(_size==-1) PTX_ERROR("PTXChain.MADHI() - chain not started");
    _position++;

    if(_position==1 && _size==1 && !_carryIn && !_carryOut)
      inliner.MADHI(r, a, b, c);
    else if(_position==1 && !_carryIn)
      inliner.MADHI_CC(r, a, b, c);
    else if(_position<_size || _carryOut)
      inliner.MADHIC_CC(r, a, b, c);
    else
      inliner.MADHIC(r, a, b, c);
  }

  __device__ __forceinline__ void PTXChain::XMADLL(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    PTXInliner inliner;

    if(_size==-1) PTX_ERROR("PTXChain.XMADLL() - chain not started");
    _position++;

    if(_position==1 && _size==1 && !_carryIn && !_carryOut)
      inliner.XMADLL(r, a, b, c);
    else if(_position==1 && !_carryIn)
      inliner.XMADLL_CC(r, a, b, c);
    else if(_position<_size || _carryOut)
      inliner.XMADLLC_CC(r, a, b, c);
    else
      inliner.XMADLLC(r, a, b, c);
  }

  __device__ __forceinline__ void PTXChain::XMADLH(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    PTXInliner inliner;

    if(_size==-1) PTX_ERROR("PTXChain.XMADLH() - chain not started");
    _position++;

    if(_position==1 && _size==1 && !_carryIn && !_carryOut)
      inliner.XMADLH(r, a, b, c);
    else if(_position==1 && !_carryIn)
      inliner.XMADLH_CC(r, a, b, c);
    else if(_position<_size || _carryOut)
      inliner.XMADLHC_CC(r, a, b, c);
    else
      inliner.XMADLHC(r, a, b, c);
  }

  __device__ __forceinline__ void PTXChain::XMADHL(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    PTXInliner inliner;

    if(_size==-1) PTX_ERROR("PTXChain.XMADHL() - chain not started");
    _position++;

    if(_position==1 && _size==1 && !_carryIn && !_carryOut)
      inliner.XMADHL(r, a, b, c);
    else if(_position==1 && !_carryIn)
      inliner.XMADHL_CC(r, a, b, c);
    else if(_position<_size || _carryOut)
      inliner.XMADHLC_CC(r, a, b, c);
    else
      inliner.XMADHLC(r, a, b, c);
  }

  __device__ __forceinline__ void PTXChain::XMADHH(uint32_t& r, uint32_t a, uint32_t& b, uint32_t& c) {
    PTXInliner inliner;

    if(_size==-1) PTX_ERROR("PTXChain.XMADHH() - chain not started");
    _position++;

    if(_position==1 && _size==1 && !_carryIn && !_carryOut)
      inliner.XMADHH(r, a, b, c);
    else if(_position==1 && !_carryIn)
      inliner.XMADHH_CC(r, a, b, c);
    else if(_position<_size || _carryOut)
      inliner.XMADHHC_CC(r, a, b, c);
    else
      inliner.XMADHHC(r, a, b, c);
  }
}
