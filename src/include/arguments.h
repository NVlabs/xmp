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


// place pointers first to avoid any alignment issues

struct one_argument {
  xmpLimb_t *out_data;
  xmpLimb_t *a_data;
  int32_t    out_len;
  int32_t    out_stride;
  int32_t    a_len;
  int32_t    a_stride;
  int32_t    a_count;
};

struct two_arguments {
  xmpLimb_t *out_data;
  xmpLimb_t *a_data;
  xmpLimb_t *b_data;
  int32_t    out_len;
  int32_t    out_stride;
  int32_t    a_len;
  int32_t    a_stride;
  int32_t    a_count;
  int32_t    b_len;
  int32_t    b_stride;
  int32_t    b_count;
};

struct division_arguments {
  xmpLimb_t *out_data;
  xmpLimb_t *a_data;
  xmpLimb_t *b_data;
  xmpLimb_t *scratch;
  int32_t    out_len;
  int32_t    out_stride;
  int32_t    a_len;
  int32_t    a_stride;
  int32_t    a_count;
  int32_t    b_len;
  int32_t    b_stride;
  int32_t    b_count;
};

struct divmod_arguments {
  xmpLimb_t *q_data;
  xmpLimb_t *m_data;
  xmpLimb_t *a_data;
  xmpLimb_t *b_data;
  xmpLimb_t *scratch;
  int32_t    q_len;
  int32_t    q_stride;
  int32_t    m_len;
  int32_t    m_stride;
  int32_t    a_len;
  int32_t    a_stride;
  int32_t    a_count;
  int32_t    b_len;
  int32_t    b_stride;
  int32_t    b_count;
};

struct ar_arguments {
  xmpLimb_t *window_data;
  xmpLimb_t *a_data;
  xmpLimb_t *mod_data;
  int32_t    window_bits;
  int32_t    a_len;
  int32_t    a_stride;
  int32_t    a_count;
  int32_t    mod_len;
  int32_t    mod_stride;
  int32_t    mod_count;
};

struct powm_arguments {
  xmpLimb_t *out_data;
  xmpLimb_t *exp_data;
  xmpLimb_t *window_data;
  int32_t    out_len;
  int32_t    out_stride;
  int32_t    exp_stride;
  int32_t    exp_count;
  int32_t    mod_count;   // specifying a non-zero mod_count means to cache the modulus in shared mem
  int32_t    digits;
  int32_t    bits;
  int32_t    window_bits;
};

struct cmp_arguments {
  int32_t   *out_data;
  xmpLimb_t *a_data;
  xmpLimb_t *b_data;
  int32_t    a_len;
  int32_t    a_stride;
  int32_t    a_count;
  int32_t    b_len;
  int32_t    b_stride;
  int32_t    b_count;
  int32_t    negate;
};

struct shf_arguments {
  xmpLimb_t *out_data;
  xmpLimb_t *a_data;
  int32_t   *shift_data;
  int32_t    out_len;
  int32_t    out_stride;
  int32_t    a_len;
  int32_t    a_stride;
  int32_t    a_count;
  int32_t    shift_count;
};

struct popc_arguments {
  uint32_t  *out_data;
  xmpLimb_t *a_data;
  int32_t    a_len;
  int32_t    a_stride;
  int32_t    a_count;
};

typedef struct two_arguments add_arguments_t;
typedef struct two_arguments sub_arguments_t;
typedef struct one_argument  sqr_arguments_t;
typedef struct two_arguments mul_arguments_t;
typedef struct division_arguments div_arguments_t;
typedef struct division_arguments mod_arguments_t;
typedef struct divmod_arguments divmod_arguments_t;
typedef struct ar_arguments ar_arguments_t;
typedef struct powm_arguments powm_arguments_t;
typedef struct cmp_arguments cmp_arguments_t;
typedef struct two_arguments ior_arguments_t;
typedef struct two_arguments and_arguments_t;
typedef struct two_arguments xor_arguments_t;
typedef struct one_argument not_arguments_t;
typedef struct popc_arguments popc_arguments_t;
typedef struct shf_arguments shf_arguments_t;
