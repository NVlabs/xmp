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


#include "three_n.h"
#include "digitized.h"
#include "warp_distributed.h"

#define RMP_STATE_SQUARE            0x01000000
#define RMP_STATE_WINDOW_MULTIPLY   0x00800000
#define RMP_STATE_REDUCE            0x00400000
#define RMP_STATE_CONSTANT_MULTIPLY 0x00200000
#define RMP_STATE_EXIT              0x00100000
#define RMP_STATE_ODDS              0x000F0000
#define RMP_STATE_EVENS             0x000E0000
#define RMP_STATE_START             0x000D0000

namespace xmp {
  template <class Model, bool storeResults>
  __device__ __forceinline__ void fwe(int32_t   mod_count,
                                      uint32_t *window,
                                      int32_t   size, int32_t width, int32_t bits, int32_t window_bits) {
    Model    model(size, width, bits, window_bits);
    int      index;
    uint32_t state;

    model.initialize(window, mod_count);
    model.computeRModuloN(window);
    model.storeWindow(window, 0);

    model.loadWindow(window, 1);
    state=0xFF000000 + RMP_STATE_START + 2;

//    index=2;
//    state=RMP_STATE_CONSTANT_MULTIPLY + RMP_STATE_START + 1;

    // This state machine is intricate, be careful with changes.

    while(true) {
      // if(blockIdx.x==0 && threadIdx.x>=32 && threadIdx.x<40) { printf("state=%08X ", state); model.printCurrent(window, "current"); }
      if(state>=RMP_STATE_SQUARE) {
        model.squareCurrent(window);
        state=state + 0x01000000;
      }
      else if(state>=RMP_STATE_WINDOW_MULTIPLY) {
        model.multiplyCurrentByWindow(window, index);
        state=state-RMP_STATE_WINDOW_MULTIPLY;
        if(state==0)
          state=RMP_STATE_REDUCE + RMP_STATE_EXIT;
        else if(state<0x10000) {
          state=state - window_bits;
          index=model.getBits(window, state, window_bits);
          state=state + (-window_bits<<24) + RMP_STATE_WINDOW_MULTIPLY;
        }
      }
      else if(state<RMP_STATE_EXIT) {
        model.storeWindow(window, state & 0xFFFF);
        if(state>=RMP_STATE_ODDS) {
          if((state & 0xFFFF)<((1<<window_bits)-1)) {
            index=2;
            state=state + RMP_STATE_WINDOW_MULTIPLY + 2;
          }
          else {
            model.loadWindow(window, 2);
            state=0xFF000000 + RMP_STATE_EVENS + 4;
          }
        }
        else if(state>=RMP_STATE_EVENS) {
          if((state & 0xFFFF)<((1<<window_bits)-2)) {
            state=state + 0xFF000000 + 2;
            model.loadWindow(window, (state & 0xFFFF)/2);
          }
          else {
            int offset=(bits%window_bits==0) ? window_bits : bits%window_bits;

            model.loadWindow(window, model.getBits(window, bits-offset, offset));
            index=model.getBits(window, bits-offset-window_bits, window_bits);
            state=(-window_bits<<24) + RMP_STATE_WINDOW_MULTIPLY + bits - offset - window_bits;
          }
        }
        else if(state==RMP_STATE_START+2) {
          index=1;
          state=0x00800000 + RMP_STATE_ODDS + 3;
        }
        else
          state=0xFF000000 + RMP_STATE_START + 2;
        continue;
      }
      else if(state>=RMP_STATE_REDUCE) {
        model.multiplyCurrentByOne(window);
        state=state-RMP_STATE_REDUCE;
      }
      else if(state>=RMP_STATE_CONSTANT_MULTIPLY) {
        model.multiplyCurrentByConstant(window, index);
        state=state-RMP_STATE_CONSTANT_MULTIPLY;
        if(state>=RMP_STATE_EXIT)
          break;
      }
      else if(state>=RMP_STATE_EXIT)
        break;

      model.reduceCurrent(window, mod_count);
    }

    if(storeResults)
      model.storeWindow(window, 0);
  }
}

