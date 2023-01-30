// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <immintrin.h>

#include "ppl/kernel/x86/fp32/conv2d/fma/conv2d_n16cx_direct_kernel_fp32_fma.h"
#include "ppl/kernel/x86/common/array_param_helper.h"

namespace ppl { namespace kernel { namespace x86 {

typedef conv2d_n16cx_direct_kernel_fp32_fma::param_def ker_p_def;
typedef conv2d_n16cx_direct_kernel_fp32_fma::config ker_cfg;
typedef conv2d_n16cx_direct_kernel_fp32_fma::flag ker_flag;

template <bool nt_store, int32_t spec_stride_w, int32_t u_oc, int32_t u_w>
void conv2d_n16cx_direct_fp32_fma_blk1x6_kernel(int64_t *param)
{
#define IC_COMPUTE_STEP(IC) do {\
    if (u_ocr > 0) ymm14 = _mm256_loadu_ps(icb_flt + 0 * OC_REG_ELTS + (IC) * OC_DATA_BLK);\
    if (u_ocr > 1) ymm15 = _mm256_loadu_ps(icb_flt + 1 * OC_REG_ELTS + (IC) * OC_DATA_BLK);\
    _mm_prefetch((const char*)(icb_flt + (IC) * OC_DATA_BLK + IC_DATA_BLK * OC_DATA_BLK), _MM_HINT_T0);\
    if (u_w > 0) {\
        ymm12 = _mm256_set1_ps(k_src_w0[(IC) + 0 * src_sw_stride]);\
        if (u_ocr > 0) ymm0 = _mm256_fmadd_ps(ymm14, ymm12, ymm0);\
        if (u_ocr > 1) ymm1 = _mm256_fmadd_ps(ymm15, ymm12, ymm1);\
    }\
    if (u_w > 1) {\
        ymm13 = _mm256_set1_ps(k_src_w0[(IC) + 1 * src_sw_stride]);\
        if (u_ocr > 0) ymm2 = _mm256_fmadd_ps(ymm14, ymm13, ymm2);\
        if (u_ocr > 1) ymm3 = _mm256_fmadd_ps(ymm15, ymm13, ymm3);\
    }\
    if (u_w > 2) {\
        ymm12 = _mm256_set1_ps(k_src_w0[(IC) + 2 * src_sw_stride]);\
        if (u_ocr > 0) ymm4 = _mm256_fmadd_ps(ymm14, ymm12, ymm4);\
        if (u_ocr > 1) ymm5 = _mm256_fmadd_ps(ymm15, ymm12, ymm5);\
    }\
    if (u_w > 3) {\
        ymm13 = _mm256_set1_ps(k_src_w3[(IC) + 0 * src_sw_stride]);\
        if (u_ocr > 0) ymm6 = _mm256_fmadd_ps(ymm14, ymm13, ymm6);\
        if (u_ocr > 1) ymm7 = _mm256_fmadd_ps(ymm15, ymm13, ymm7);\
    }\
    if (u_w > 4) {\
        ymm12 = _mm256_set1_ps(k_src_w3[(IC) + 1 * src_sw_stride]);\
        if (u_ocr > 0) ymm8 = _mm256_fmadd_ps(ymm14, ymm12, ymm8);\
        if (u_ocr > 1) ymm9 = _mm256_fmadd_ps(ymm15, ymm12, ymm9);\
    }\
    if (u_w > 5) {\
        ymm13 = _mm256_set1_ps(k_src_w3[(IC) + 2 * src_sw_stride]);\
        if (u_ocr > 0) ymm10 = _mm256_fmadd_ps(ymm14, ymm13, ymm10);\
        if (u_ocr > 1) ymm11 = _mm256_fmadd_ps(ymm15, ymm13, ymm11);\
    }\
} while (0)

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    const int64_t IC_DATA_BLK = ker_cfg::IC_DATA_BLK;
    const int64_t OC_DATA_BLK = ker_cfg::OC_DATA_BLK;
    const int64_t OC_REG_ELTS = ker_cfg::OC_REG_ELTS;
    const int64_t u_ocr = div_up(u_oc, OC_REG_ELTS);
    const int64_t u_ic = 4;

    array_param_helper ker_p(param);

    const int64_t kernel_h = ker_p.pick<const int64_t>(ker_p_def::KH_IDX);
    const int64_t kernel_w = ker_p.pick<const int64_t>(ker_p_def::KW_IDX);
    const int64_t src_icb_stride = ker_p.pick<const int64_t>(ker_p_def::SRC_ICB_STRIDE_IDX);
    const int64_t src_dw_stride = ker_p.pick<const int64_t>(ker_p_def::SRC_DW_STRIDE_IDX);
    const int64_t src_dh_stride = ker_p.pick<const int64_t>(ker_p_def::SRC_DH_STRIDE_IDX) - kernel_w * src_dw_stride;
    const int64_t src_sw_stride = spec_stride_w ? (spec_stride_w * IC_DATA_BLK) : ker_p.pick<const int64_t>(ker_p_def::SRC_SW_STRIDE_IDX);
    const int64_t kh_start = ker_p.pick<const int64_t>(ker_p_def::KH_START_IDX);
    const int64_t kh_end = ker_p.pick<const int64_t>(ker_p_def::KH_END_IDX);
    const int64_t kernel_flags = ker_p.pick<const int64_t>(ker_p_def::FLAGS_IDX);

    const int64_t flt_icb_stride = (kernel_h - kh_end + kh_start) * kernel_w * IC_DATA_BLK * OC_DATA_BLK;
    const int64_t src_offset = kh_start * (src_dh_stride + kernel_w * src_dw_stride);
    const int64_t flt_offset = kh_start * kernel_w * IC_DATA_BLK * OC_DATA_BLK;

    int64_t dst_w    = ker_p.pick<int64_t>(ker_p_def::DST_WIDTH_IDX);
    const float *src = ker_p.pick<const float*>(ker_p_def::SRC_PTR_IDX);
    const float *his = ker_p.pick<const float*>(ker_p_def::HIS_PTR_IDX);
    float *dst       = ker_p.pick<float*>(ker_p_def::DST_PTR_IDX);

    do {
        his += u_w * OC_DATA_BLK;
        if (kernel_flags & ker_flag::HIS) {
            if (u_w > 0) {
                if (u_ocr > 0) ymm0 = _mm256_loadu_ps(his + 0 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK);
                if (u_ocr > 1) ymm1 = _mm256_loadu_ps(his + 0 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK);
            }
            if (u_w > 1) {
                if (u_ocr > 0) ymm2 = _mm256_loadu_ps(his + 1 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK);
                if (u_ocr > 1) ymm3 = _mm256_loadu_ps(his + 1 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK);
            }
            if (u_w > 2) {
                if (u_ocr > 0) ymm4 = _mm256_loadu_ps(his + 2 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK);
                if (u_ocr > 1) ymm5 = _mm256_loadu_ps(his + 2 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK);
            }
            if (u_w > 3) {
                if (u_ocr > 0) ymm6 = _mm256_loadu_ps(his + 3 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK);
                if (u_ocr > 1) ymm7 = _mm256_loadu_ps(his + 3 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK);
            }
            if (u_w > 4) {
                if (u_ocr > 0) ymm8 = _mm256_loadu_ps(his + 4 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK);
                if (u_ocr > 1) ymm9 = _mm256_loadu_ps(his + 4 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK);
            }
            if (u_w > 5) {
                if (u_ocr > 0) ymm10 = _mm256_loadu_ps(his + 5 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK);
                if (u_ocr > 1) ymm11 = _mm256_loadu_ps(his + 5 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK);
            }
        } else {
            if (u_w > 0) {
                if (u_ocr > 0) ymm0 = _mm256_setzero_ps();
                if (u_ocr > 1) ymm1 = _mm256_setzero_ps();
            }
            if (u_w > 1) {
                if (u_ocr > 0) ymm2 = _mm256_setzero_ps();
                if (u_ocr > 1) ymm3 = _mm256_setzero_ps();
            }
            if (u_w > 2) {
                if (u_ocr > 0) ymm4 = _mm256_setzero_ps();
                if (u_ocr > 1) ymm5 = _mm256_setzero_ps();
            }
            if (u_w > 3) {
                if (u_ocr > 0) ymm6 = _mm256_setzero_ps();
                if (u_ocr > 1) ymm7 = _mm256_setzero_ps();
            }
            if (u_w > 4) {
                if (u_ocr > 0) ymm8 = _mm256_setzero_ps();
                if (u_ocr > 1) ymm9 = _mm256_setzero_ps();
            }
            if (u_w > 5) {
                if (u_ocr > 0) ymm10 = _mm256_setzero_ps();
                if (u_ocr > 1) ymm11 = _mm256_setzero_ps();
            }
        }

        if (kernel_flags & ker_flag::BIAS) {
            const float* bias = ker_p.pick<const float*>(ker_p_def::BIAS_PTR_IDX);
            if (u_ocr > 0) ymm14 = _mm256_loadu_ps(bias + 0 * OC_REG_ELTS);
            if (u_ocr > 1) ymm15 = _mm256_loadu_ps(bias + 1 * OC_REG_ELTS);
            if (u_w > 0) {
                if (u_ocr > 0) ymm0 = _mm256_add_ps(ymm14, ymm0);
                if (u_ocr > 1) ymm1 = _mm256_add_ps(ymm15, ymm1);
            }
            if (u_w > 1) {
                if (u_ocr > 0) ymm2 = _mm256_add_ps(ymm14, ymm2);
                if (u_ocr > 1) ymm3 = _mm256_add_ps(ymm15, ymm3);
            }
            if (u_w > 2) {
                if (u_ocr > 0) ymm4 = _mm256_add_ps(ymm14, ymm4);
                if (u_ocr > 1) ymm5 = _mm256_add_ps(ymm15, ymm5);
            }
            if (u_w > 3) {
                if (u_ocr > 0) ymm6 = _mm256_add_ps(ymm14, ymm6);
                if (u_ocr > 1) ymm7 = _mm256_add_ps(ymm15, ymm7);
            }
            if (u_w > 4) {
                if (u_ocr > 0) ymm8 = _mm256_add_ps(ymm14, ymm8);
                if (u_ocr > 1) ymm9 = _mm256_add_ps(ymm15, ymm9);
            }
            if (u_w > 5) {
                if (u_ocr > 0) ymm10 = _mm256_add_ps(ymm14, ymm10);
                if (u_ocr > 1) ymm11 = _mm256_add_ps(ymm15, ymm11);
            }
        }

        int64_t channels     = ker_p.pick<int64_t>(ker_p_def::CHANNELS_IDX);
        const float *icb_flt = ker_p.pick<const float*>(ker_p_def::FLT_PTR_IDX) + flt_offset;
        const float *icb_src = src + src_offset;
        while (channels >= IC_DATA_BLK) {
            channels -= IC_DATA_BLK;
            const float *k_src_w0;
            const float *k_src_w3;
            if (u_w > 0) k_src_w0 = icb_src + 0 * src_sw_stride;
            if (u_w > 3) k_src_w3 = icb_src + 3 * src_sw_stride;
            for (int64_t kh = kh_start; kh < kh_end; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t ic = IC_DATA_BLK;
                    do {
                        ic -= u_ic;
                        IC_COMPUTE_STEP(0);
                        IC_COMPUTE_STEP(1);
                        IC_COMPUTE_STEP(2);
                        IC_COMPUTE_STEP(3);
                        if (u_w > 0) k_src_w0 += u_ic;
                        if (u_w > 3) k_src_w3 += u_ic;
                        icb_flt += u_ic * OC_DATA_BLK;
                    } while (ic);
                    if (u_w > 0) k_src_w0 += src_dw_stride - IC_DATA_BLK;
                    if (u_w > 3) k_src_w3 += src_dw_stride - IC_DATA_BLK;
                }
                if (u_w > 0) k_src_w0 += src_dh_stride;
                if (u_w > 3) k_src_w3 += src_dh_stride;
            }
            icb_src += src_icb_stride;
            icb_flt += flt_icb_stride;
        }
        src += u_w * src_sw_stride;

        if (channels > 0) {
            const float *k_src_w0;
            const float *k_src_w3;
            if (u_w > 0) k_src_w0 = icb_src + 0 * src_sw_stride;
            if (u_w > 3) k_src_w3 = icb_src + 3 * src_sw_stride;
            for (int64_t kh = kh_start; kh < kh_end; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t ic = channels;
                    while (ic >= u_ic) {
                        ic -= u_ic;
                        IC_COMPUTE_STEP(0);
                        IC_COMPUTE_STEP(1);
                        IC_COMPUTE_STEP(2);
                        IC_COMPUTE_STEP(3);
                        if (u_w > 0) k_src_w0 += u_ic;
                        if (u_w > 3) k_src_w3 += u_ic;
                        icb_flt += u_ic * OC_DATA_BLK;
                    }
                    if (ic >= 3) {
                        IC_COMPUTE_STEP(0);
                        if (u_w > 0) k_src_w0 += 1;
                        if (u_w > 3) k_src_w3 += 1;
                        icb_flt += OC_DATA_BLK;
                    }
                    if (ic >= 2) {
                        IC_COMPUTE_STEP(0);
                        if (u_w > 0) k_src_w0 += 1;
                        if (u_w > 3) k_src_w3 += 1;
                        icb_flt += OC_DATA_BLK;
                    }
                    if (ic >= 1) {
                        IC_COMPUTE_STEP(0);
                        if (u_w > 0) k_src_w0 += 1;
                        if (u_w > 3) k_src_w3 += 1;
                        icb_flt += OC_DATA_BLK;
                    }
                    if (u_w > 0) k_src_w0 += src_dw_stride - channels;
                    if (u_w > 3) k_src_w3 += src_dw_stride - channels;
                    icb_flt += (IC_DATA_BLK - channels) * OC_DATA_BLK;
                }
                if (u_w > 0) k_src_w0 += src_dh_stride;
                if (u_w > 3) k_src_w3 += src_dh_stride;
            }
        }

        dst += u_w * OC_DATA_BLK;
        if (kernel_flags & (ker_flag::RELU | ker_flag::RELU6)) {
            ymm14 = _mm256_setzero_ps();
            if (u_w > 0) {
                if (u_ocr > 0) ymm0 = _mm256_max_ps(ymm0, ymm14);
                if (u_ocr > 1) ymm1 = _mm256_max_ps(ymm1, ymm14);
            }
            if (u_w > 1) {
                if (u_ocr > 0) ymm2 = _mm256_max_ps(ymm2, ymm14);
                if (u_ocr > 1) ymm3 = _mm256_max_ps(ymm3, ymm14);
            }
            if (u_w > 2) {
                if (u_ocr > 0) ymm4 = _mm256_max_ps(ymm4, ymm14);
                if (u_ocr > 1) ymm5 = _mm256_max_ps(ymm5, ymm14);
            }
            if (u_w > 3) {
                if (u_ocr > 0) ymm6 = _mm256_max_ps(ymm6, ymm14);
                if (u_ocr > 1) ymm7 = _mm256_max_ps(ymm7, ymm14);
            }
            if (u_w > 4) {
                if (u_ocr > 0) ymm8 = _mm256_max_ps(ymm8, ymm14);
                if (u_ocr > 1) ymm9 = _mm256_max_ps(ymm9, ymm14);
            }
            if (u_w > 5) {
                if (u_ocr > 0) ymm10 = _mm256_max_ps(ymm10, ymm14);
                if (u_ocr > 1) ymm11 = _mm256_max_ps(ymm11, ymm14);
            }
        }

        if (kernel_flags & ker_flag::RELU6) {
            ymm15 = _mm256_set1_ps(6.0f);
            if (u_w > 0) {
                if (u_ocr > 0) ymm0 = _mm256_min_ps(ymm0, ymm15);
                if (u_ocr > 1) ymm1 = _mm256_min_ps(ymm1, ymm15);
            }
            if (u_w > 1) {
                if (u_ocr > 0) ymm2 = _mm256_min_ps(ymm2, ymm15);
                if (u_ocr > 1) ymm3 = _mm256_min_ps(ymm3, ymm15);
            }
            if (u_w > 2) {
                if (u_ocr > 0) ymm4 = _mm256_min_ps(ymm4, ymm15);
                if (u_ocr > 1) ymm5 = _mm256_min_ps(ymm5, ymm15);
            }
            if (u_w > 3) {
                if (u_ocr > 0) ymm6 = _mm256_min_ps(ymm6, ymm15);
                if (u_ocr > 1) ymm7 = _mm256_min_ps(ymm7, ymm15);
            }
            if (u_w > 4) {
                if (u_ocr > 0) ymm8 = _mm256_min_ps(ymm8, ymm15);
                if (u_ocr > 1) ymm9 = _mm256_min_ps(ymm9, ymm15);
            }
            if (u_w > 5) {
                if (u_ocr > 0) ymm10 = _mm256_min_ps(ymm10, ymm15);
                if (u_ocr > 1) ymm11 = _mm256_min_ps(ymm11, ymm15);
            }
        }

        dst_w -= u_w;
        if (nt_store) {
            if (u_w > 0) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 0 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm0);
                if (u_ocr > 1) _mm256_stream_ps(dst + 0 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm1);
            }
            if (u_w > 1) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 1 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm2);
                if (u_ocr > 1) _mm256_stream_ps(dst + 1 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm3);
            }
            if (u_w > 2) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 2 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm4);
                if (u_ocr > 1) _mm256_stream_ps(dst + 2 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm5);
            }
            if (u_w > 3) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 3 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm6);
                if (u_ocr > 1) _mm256_stream_ps(dst + 3 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm7);
            }
            if (u_w > 4) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 4 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm8);
                if (u_ocr > 1) _mm256_stream_ps(dst + 4 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm9);
            }
            if (u_w > 5) {
                if (u_ocr > 0) _mm256_stream_ps(dst + 5 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm10);
                if (u_ocr > 1) _mm256_stream_ps(dst + 5 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm11);
            }
        } else {
            if (u_w > 0) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 0 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm0);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 0 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm1);
            }
            if (u_w > 1) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 1 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm2);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 1 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm3);
            }
            if (u_w > 2) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 2 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm4);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 2 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm5);
            }
            if (u_w > 3) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 3 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm6);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 3 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm7);
            }
            if (u_w > 4) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 4 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm8);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 4 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm9);
            }
            if (u_w > 5) {
                if (u_ocr > 0) _mm256_storeu_ps(dst + 5 * OC_DATA_BLK + 0 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm10);
                if (u_ocr > 1) _mm256_storeu_ps(dst + 5 * OC_DATA_BLK + 1 * OC_REG_ELTS - u_w * OC_DATA_BLK, ymm11);
            }
        }
    } while (dst_w > 0);
    ker_p.pick<const float*>(ker_p_def::SRC_PTR_IDX) = src;
    ker_p.pick<const float*>(ker_p_def::HIS_PTR_IDX) = his;
    ker_p.pick<float*>(ker_p_def::DST_PTR_IDX) = dst;
#undef IC_COMPUTE_STEP
}

template <bool nt_store, int32_t u_oc>
void conv2d_n16cx_direct_fp32_fma_blk1x1_kernel(int64_t *param)
{
#define IC_COMPUTE_STEP(IC) do {\
    ymm2 = _mm256_set1_ps(ic_src[(IC)]);\
    if (u_ocr > 0) ymm0 = _mm256_fmadd_ps(_mm256_loadu_ps(ic_flt + 0 * OC_REG_ELTS + (IC) * OC_DATA_BLK), ymm2, ymm0);\
    if (u_ocr > 1) ymm1 = _mm256_fmadd_ps(_mm256_loadu_ps(ic_flt + 1 * OC_REG_ELTS + (IC) * OC_DATA_BLK), ymm2, ymm1);\
} while (0)

    __m256 ymm0, ymm1, ymm2, ymm3, ymm4;

    const int64_t IC_DATA_BLK = ker_cfg::IC_DATA_BLK;
    const int64_t OC_DATA_BLK = ker_cfg::OC_DATA_BLK;
    const int64_t OC_REG_ELTS = ker_cfg::OC_REG_ELTS;
    const int64_t u_ocr = div_up(u_oc, OC_REG_ELTS);

    array_param_helper ker_p(param);

    const int64_t kernel_h = ker_p.pick<const int64_t>(ker_p_def::KH_IDX);
    const int64_t kernel_w = ker_p.pick<const int64_t>(ker_p_def::KW_IDX);
    const int64_t src_icb_stride = ker_p.pick<const int64_t>(ker_p_def::SRC_ICB_STRIDE_IDX);
    const int64_t src_dh_stride = ker_p.pick<const int64_t>(ker_p_def::SRC_DH_STRIDE_IDX);
    const int64_t src_dw_stride = ker_p.pick<const int64_t>(ker_p_def::SRC_DW_STRIDE_IDX);
    const int64_t kh_start = ker_p.pick<const int64_t>(ker_p_def::KH_START_IDX);
    const int64_t kh_end = ker_p.pick<const int64_t>(ker_p_def::KH_END_IDX);
    const int64_t kw_start = ker_p.pick<const int64_t>(ker_p_def::KW_START_IDX);
    const int64_t kw_end = ker_p.pick<const int64_t>(ker_p_def::KW_END_IDX);
    const int64_t kernel_flags = ker_p.pick<const int64_t>(ker_p_def::FLAGS_IDX);

    if (kernel_flags & ker_flag::HIS) {
        const float* his = ker_p.pick<const float*>(ker_p_def::HIS_PTR_IDX);
        ker_p.pick<const float*>(ker_p_def::HIS_PTR_IDX) = his + OC_DATA_BLK;
        if (u_ocr > 0) ymm0 = _mm256_loadu_ps(his + 0 * OC_REG_ELTS);
        if (u_ocr > 1) ymm1 = _mm256_loadu_ps(his + 1 * OC_REG_ELTS);
    } else {
        if (u_ocr > 0) ymm0 = _mm256_setzero_ps();
        if (u_ocr > 1) ymm1 = _mm256_setzero_ps();
    }

    if (kernel_flags & ker_flag::BIAS) {
        const float* bias = ker_p.pick<const float*>(ker_p_def::BIAS_PTR_IDX);
        if (u_ocr > 0) ymm0 = _mm256_add_ps(_mm256_loadu_ps(bias + 0 * OC_REG_ELTS), ymm0);
        if (u_ocr > 1) ymm1 = _mm256_add_ps(_mm256_loadu_ps(bias + 1 * OC_REG_ELTS), ymm1);
    }

    int64_t channels = ker_p.pick<const int64_t>(ker_p_def::CHANNELS_IDX);
    const float *icb_src = ker_p.pick<const float*>(ker_p_def::SRC_PTR_IDX) + kh_start * src_dh_stride;
    const float *icb_flt = ker_p.pick<const float*>(ker_p_def::FLT_PTR_IDX) + kh_start * kernel_w * IC_DATA_BLK * OC_DATA_BLK;
    ker_p.pick<const float*>(ker_p_def::SRC_PTR_IDX) += ker_p.pick<const int64_t>(ker_p_def::SRC_SW_STRIDE_IDX);

    ymm3 = _mm256_setzero_ps();
    ymm4 = _mm256_set1_ps(6.0f);

    while (channels >= IC_DATA_BLK) {
        channels -= IC_DATA_BLK;
        const float *kh_src = icb_src;
        const float *kh_flt = icb_flt;
        for (int64_t kh = kh_start; kh < kh_end; ++kh) {
            const float *kw_src = kh_src + kw_start * src_dw_stride;
            const float *kw_flt = kh_flt + kw_start * IC_DATA_BLK * OC_DATA_BLK;
            for (int64_t kw = kw_start; kw < kw_end; ++kw) {
                const float *ic_src = kw_src;
                const float *ic_flt = kw_flt;
                IC_COMPUTE_STEP(0);
                IC_COMPUTE_STEP(1);
                IC_COMPUTE_STEP(2);
                IC_COMPUTE_STEP(3);

                IC_COMPUTE_STEP(4);
                IC_COMPUTE_STEP(5);
                IC_COMPUTE_STEP(6);
                IC_COMPUTE_STEP(7);

                IC_COMPUTE_STEP(8);
                IC_COMPUTE_STEP(9);
                IC_COMPUTE_STEP(10);
                IC_COMPUTE_STEP(11);

                IC_COMPUTE_STEP(12);
                IC_COMPUTE_STEP(13);
                IC_COMPUTE_STEP(14);
                IC_COMPUTE_STEP(15);
                kw_flt += IC_DATA_BLK * OC_DATA_BLK;
                kw_src += src_dw_stride;
            }
            kh_flt += kernel_w * IC_DATA_BLK * OC_DATA_BLK;
            kh_src += src_dh_stride;
        }
        icb_flt += kernel_h * kernel_w * IC_DATA_BLK * OC_DATA_BLK;
        icb_src += src_icb_stride;
    }

    if (channels > 0) {
        const float *kh_src = icb_src;
        const float *kh_flt = icb_flt;
        for (int64_t kh = kh_start; kh < kh_end; ++kh) {
            const float *kw_src = kh_src + kw_start * src_dw_stride;
            const float *kw_flt = kh_flt + kw_start * IC_DATA_BLK * OC_DATA_BLK;
            for (int64_t kw = kw_start; kw < kw_end; ++kw) {
                const float *ic_src = kw_src;
                const float *ic_flt = kw_flt;
                for (int64_t ic = 0; ic < channels; ++ic) {
                    IC_COMPUTE_STEP(0);
                    ic_src += 1;
                    ic_flt += OC_DATA_BLK;
                }
                kw_flt += IC_DATA_BLK * OC_DATA_BLK;
                kw_src += src_dw_stride;
            }
            kh_flt += kernel_w * IC_DATA_BLK * OC_DATA_BLK;
            kh_src += src_dh_stride;
        }
    }

    if (kernel_flags & (ker_flag::RELU | ker_flag::RELU6)) {
        if (u_ocr > 0) ymm0 = _mm256_max_ps(ymm0, ymm3);
        if (u_ocr > 1) ymm1 = _mm256_max_ps(ymm1, ymm3);
    }

    if (kernel_flags & ker_flag::RELU6) {
        if (u_ocr > 0) ymm0 = _mm256_min_ps(ymm0, ymm4);
        if (u_ocr > 1) ymm1 = _mm256_min_ps(ymm1, ymm4);
    }

    float* dst = ker_p.pick<float*>(ker_p_def::DST_PTR_IDX);
    ker_p.pick<float*>(ker_p_def::DST_PTR_IDX) = dst + OC_DATA_BLK;
    if (nt_store) {
        if (u_ocr > 0) _mm256_stream_ps(dst + 0 * OC_REG_ELTS, ymm0);
        if (u_ocr > 1) _mm256_stream_ps(dst + 1 * OC_REG_ELTS, ymm1);
    } else {
        if (u_ocr > 0) _mm256_storeu_ps(dst + 0 * OC_REG_ELTS, ymm0);
        if (u_ocr > 1) _mm256_storeu_ps(dst + 1 * OC_REG_ELTS, ymm1);
    }
#undef IC_COMPUTE_STEP
}

const conv2d_n16cx_direct_kernel_fp32_fma::func_t
    conv2d_n16cx_direct_kernel_fp32_fma::border_table_[config::NT_STORE_OPT][config::MAX_OC_REGS] =
{
    {
        conv2d_n16cx_direct_fp32_fma_blk1x1_kernel<false, 1 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS>,
        conv2d_n16cx_direct_fp32_fma_blk1x1_kernel<false, 2 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS>,
    },
    {
        conv2d_n16cx_direct_fp32_fma_blk1x1_kernel<true, 1 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS>,
        conv2d_n16cx_direct_fp32_fma_blk1x1_kernel<true, 2 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS>,
    },
};

#define DIRECT_KERNEL_TABLE_BLK(NT_STORE, STRIDE_W) \
{\
    {\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 1>,\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 2>,\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 3>,\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 4>,\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 5>,\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 6>,\
    },\
    {\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 1>,\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 2>,\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 3>,\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 4>,\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 5>,\
        conv2d_n16cx_direct_fp32_fma_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * conv2d_n16cx_direct_kernel_fp32_fma::config::OC_REG_ELTS, 6>,\
    },\
}

const conv2d_n16cx_direct_kernel_fp32_fma::func_t
    conv2d_n16cx_direct_kernel_fp32_fma::table_[config::NT_STORE_OPT][config::STRIDE_W_OPT][config::MAX_OC_REGS][config::MAX_W_REGS] =
{
    {
        DIRECT_KERNEL_TABLE_BLK(false, 0),
        DIRECT_KERNEL_TABLE_BLK(false, 1),
        DIRECT_KERNEL_TABLE_BLK(false, 2),
    },
    {
        DIRECT_KERNEL_TABLE_BLK(true, 0),
        DIRECT_KERNEL_TABLE_BLK(true, 1),
        DIRECT_KERNEL_TABLE_BLK(true, 2),
    },
};

}}};
