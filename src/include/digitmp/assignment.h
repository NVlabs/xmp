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
  template<int size>
  __device__ __forceinline__ void set_ui(uint32_t *registers, DigitMP<size> r, uint32_t ui) {
    RegMP UI(registers, 0, 0, size);

    set_ui(UI, ui);
    #pragma nounroll
    for(int index=0;index<r.digits();index++) {
      r.store_digit(UI, index);
      UI[0]=0;
    }
  }

  template<int size>
  __device__ __forceinline__ void set(uint32_t *registers, DigitMP<size> r, DigitMP<size> x) {
    RegMP   digit(registers, 0, 0, size);
    int32_t index;

    #pragma nounroll
    for(index=0;index<min(r.digits(), x.digits());index++) {
      x.load_digit(digit, index);
      r.store_digit(digit, index);
    }
    set_ui(digit, 0);
    while(index<r.digits())
      r.store_digit(digit, index++);
  }
}
