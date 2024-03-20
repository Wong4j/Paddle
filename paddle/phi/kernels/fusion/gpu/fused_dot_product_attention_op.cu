// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpudnn/mha_cudnn_frontend.h"

namespace phi {
namespace fusion {

__global__ void set_rng_state(std::pair<uint64_t, uint64_t> seed_offset,
                              int64_t *rng_state_ptr) {
  rng_state_ptr[0] = static_cast<int64_t>(seed_offset.first);
  rng_state_ptr[1] = static_cast<int64_t>(seed_offset.second);
}

const std::map<std::string, MHA_Bias_Type> kBiasTypeMap = {
    {"no_bias", MHA_Bias_Type::NO_BIAS},
    {"pre_scale_bias", MHA_Bias_Type::PRE_SCALE_BIAS},
    {"post_scale_bias", MHA_Bias_Type::POST_SCALE_BIAS}};

const std::map<std::string, MHA_Mask_Type> kMaskTypeMap = {
    {"causal", MHA_Mask_Type::CAUSAL_MASK},
    {"padding", MHA_Mask_Type::PADDING_MASK},
    {"none", MHA_Mask_Type::NO_MASK},
    {"padding_causal", MHA_Mask_Type::PADDING_CAUSAL_MASK}};

template <typename T, typename Context>
void FusedDotProductAttentionKernel(const Context &dev_ctx,
                                    const DenseTensor &q,
                                    const DenseTensor &k,
                                    const DenseTensor &v,
                                    const DenseTensor &cu_seqlen_q,
                                    const DenseTensor &cu_seqlen_kv,
                                    const paddle::optional<DenseTensor> &bias,
                                    float scaling_factor,
                                    float dropout_probability,
                                    bool is_training,
                                    const std::string &mask_type_str,
                                    const std::string &bias_type_str,
                                    DenseTensor *out,
                                    DenseTensor *softmax_out,
                                    DenseTensor *rng_state) {
  PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                    80,
                    phi::errors::PreconditionNotMet(
                        "This op only supports Ampere and later devices, "
                        "but got compute capability: %d.",
                        dev_ctx.GetComputeCapability()));
  auto cudnn_version = phi::backends::gpu::DnnVersion();
  PADDLE_ENFORCE_GE(cudnn_version,
                    8906,
                    phi::errors::PreconditionNotMet(
                        "This op only supports CUDNN version >= 8906, "
                        "but got %d.",
                        cudnn_version));

  // allocate output variables
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<float>(softmax_out);
  dev_ctx.template Alloc<int64_t>(rng_state);

  // get handles
  auto handle = dev_ctx.cudnn_handle();

  auto tensor_dtype = get_cudnn_fe_dtype(q.dtype());
  bool is_type_supported =
      (tensor_dtype == cudnn_frontend::DataType_t::HALF ||
       tensor_dtype == cudnn_frontend::DataType_t::BFLOAT16);
  PADDLE_ENFORCE_EQ(
      is_type_supported,
      true,
      phi::errors::InvalidArgument(
          "cuDNN fused attention Only supports FP16/BF16 currently"));
  auto mha_layout = MHA_Layout::BSHD_BSHD_BSHD;
  auto bias_type = MHA_Bias_Type::NO_BIAS;
  auto mask_type = MHA_Mask_Type::NO_MASK;
  auto bias_type_iter = kBiasTypeMap.find(bias_type_str);
  if (bias_type_iter != kBiasTypeMap.end()) {
    bias_type = bias_type_iter->second;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Invalid bias type: %s, only support no_bias, pre_scale_bias, "
        "post_scale_bias",
        bias_type_str));
  }
  if (bias.get_ptr() == nullptr) {
    bias_type = MHA_Bias_Type::NO_BIAS;
  }
  auto mask_type_iter = kMaskTypeMap.find(mask_type_str);
  if (mask_type_iter != kMaskTypeMap.end()) {
    mask_type = mask_type_iter->second;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Invalid mask type: %s, only support causal, padding, none, "
        "padding_causal",
        mask_type_str));
  }

  // q dim: {b, s_q, h, d};
  // k,v dim: {b, s_kv, h_kv, d};
  auto batch_size = q.dims()[0];
  auto q_seq_len = q.dims()[1];
  auto num_heads = q.dims()[2];
  auto num_heads_kv = k.dims()[2];
  auto head_size = q.dims()[3];
  auto kv_seq_len = k.dims()[1];

  // support bias shape: [b,1,s,s],[b,h,s,s], [1,1,s,s]
  size_t bias_b = 0;
  size_t bias_h = 0;
  void *bias_dev_ptr = nullptr;
  if (bias_type != MHA_Bias_Type::NO_BIAS) {
    bias_b = bias.get_ptr()->dims()[0];
    bias_h = bias.get_ptr()->dims()[1];
    bias_dev_ptr =
        reinterpret_cast<void *>(const_cast<T *>(bias.get_ptr()->data<T>()));
  }

  auto gen_cuda = dev_ctx.GetGenerator();
  const int rng_elts_per_thread = 16;
  auto seed_offset = gen_cuda->IncrementOffset(rng_elts_per_thread);
  set_rng_state<<<1, 1, 0, dev_ctx.stream()>>>(
      seed_offset, static_cast<int64_t *>(rng_state->data<int64_t>()));

  void *q_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(q.data<T>()));
  void *k_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(k.data<T>()));
  void *v_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(v.data<T>()));
  void *out_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(out->data<T>()));
  void *softmax_out_dev_ptr =
      reinterpret_cast<void *>(const_cast<float *>(softmax_out->data<float>()));
  void *cu_seqlen_q_dev_ptr = reinterpret_cast<void *>(
      const_cast<int32_t *>(cu_seqlen_q.data<int32_t>()));
  void *cu_seqlen_kv_dev_ptr = reinterpret_cast<void *>(
      const_cast<int32_t *>(cu_seqlen_kv.data<int32_t>()));
  // rng_state: {seed, offset}
  void *seed_dev_ptr = reinterpret_cast<void *>(
      const_cast<int64_t *>(rng_state->data<int64_t>()));
  void *offset_dev_ptr = reinterpret_cast<void *>(
      const_cast<int64_t *>(rng_state->data<int64_t>()) + 1);
  size_t workspace_size = 0;
  // call the first time to get the workspace size
  fused_attn_arbitrary_seqlen_fwd_impl(batch_size,
                                       num_heads,
                                       num_heads_kv,
                                       q_seq_len,
                                       kv_seq_len,
                                       head_size,
                                       bias_b,
                                       bias_h,
                                       is_training,
                                       scaling_factor,
                                       dropout_probability,
                                       mha_layout,
                                       bias_type,
                                       mask_type,
                                       q_dev_ptr,
                                       k_dev_ptr,
                                       v_dev_ptr,
                                       bias_dev_ptr,
                                       softmax_out_dev_ptr,
                                       out_dev_ptr,
                                       seed_dev_ptr,
                                       offset_dev_ptr,
                                       cu_seqlen_q_dev_ptr,
                                       cu_seqlen_kv_dev_ptr,
                                       tensor_dtype,
                                       nullptr,
                                       &workspace_size,
                                       dev_ctx.stream(),
                                       handle);
  DenseTensor workspace;
  workspace.Resize({workspace_size > 0 ? workspace_size : 1});
  dev_ctx.template Alloc<char>(&workspace);
  // call the second time to excute the kernel
  fused_attn_arbitrary_seqlen_fwd_impl(
      batch_size,
      num_heads,
      num_heads_kv,
      q_seq_len,
      kv_seq_len,
      head_size,
      bias_b,
      bias_h,
      is_training,
      scaling_factor,
      dropout_probability,
      mha_layout,
      bias_type,
      mask_type,
      q_dev_ptr,
      k_dev_ptr,
      v_dev_ptr,
      bias_dev_ptr,
      softmax_out_dev_ptr,
      out_dev_ptr,
      seed_dev_ptr,
      offset_dev_ptr,
      cu_seqlen_q_dev_ptr,
      cu_seqlen_kv_dev_ptr,
      tensor_dtype,
      reinterpret_cast<void *>(workspace.data<char>()),
      &workspace_size,
      dev_ctx.stream(),
      handle);
}

template <typename T, typename Context>
void FusedDotProductAttentionGradKernel(
    const Context &dev_ctx,
    const DenseTensor &q,
    const DenseTensor &k,
    const DenseTensor &v,
    const DenseTensor &cu_seqlen_q,
    const DenseTensor &cu_seqlen_kv,
    const DenseTensor &O,
    const DenseTensor &softmax_out,
    const DenseTensor &rng_state,
    const DenseTensor &dO,
    const paddle::optional<DenseTensor> &bias,
    float scaling_factor,
    float dropout_probability,
    const std::string &mask_type_str,
    const std::string &bias_type_str,
    DenseTensor *q_grad,
    DenseTensor *k_grad,
    DenseTensor *v_grad,
    DenseTensor *bias_grad) {
  auto sm_arch = dev_ctx.GetComputeCapability();
  PADDLE_ENFORCE_GE(sm_arch,
                    80,
                    phi::errors::PreconditionNotMet(
                        "This op only supports Ampere and later devices, "
                        "but got compute capability: %d.",
                        dev_ctx.GetComputeCapability()));
  auto cudnn_version = phi::backends::gpu::DnnVersion();
  PADDLE_ENFORCE_GE(cudnn_version,
                    8906,
                    phi::errors::PreconditionNotMet(
                        "This op only supports CUDNN version >= 8906, "
                        "but got %d.",
                        cudnn_version));

  // allocate output variables
  dev_ctx.template Alloc<T>(q_grad);
  dev_ctx.template Alloc<T>(k_grad);
  dev_ctx.template Alloc<T>(v_grad);

  // get handles
  auto handle = dev_ctx.cudnn_handle();

  auto tensor_dtype = get_cudnn_fe_dtype(q.dtype());
  bool is_type_supported =
      (tensor_dtype == cudnn_frontend::DataType_t::HALF ||
       tensor_dtype == cudnn_frontend::DataType_t::BFLOAT16);
  PADDLE_ENFORCE_EQ(
      is_type_supported,
      true,
      phi::errors::InvalidArgument(
          "cuDNN fused attention Only supports FP16/BF16 currently"));
  auto mha_layout = MHA_Layout::BSHD_BSHD_BSHD;
  auto bias_type = MHA_Bias_Type::NO_BIAS;
  auto mask_type = MHA_Mask_Type::NO_MASK;
  auto bias_type_iter = kBiasTypeMap.find(bias_type_str);
  if (bias_type_iter != kBiasTypeMap.end()) {
    bias_type = bias_type_iter->second;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Invalid bias type: %s, only support no_bias, pre_scale_bias, "
        "post_scale_bias",
        bias_type_str));
  }
  if (bias.get_ptr() == nullptr) {
    bias_type = MHA_Bias_Type::NO_BIAS;
  }
  auto mask_type_iter = kMaskTypeMap.find(mask_type_str);
  if (mask_type_iter != kMaskTypeMap.end()) {
    mask_type = mask_type_iter->second;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Invalid mask type: %s, only support causal, padding, none, "
        "padding_causal",
        mask_type_str));
  }

  // q dim: {b, s_q, h, d};
  // k, v dim: {b, s_kv, h_kv, d};
  auto batch_size = q.dims()[0];
  auto q_seq_len = q.dims()[1];
  auto num_heads = q.dims()[2];
  auto num_heads_kv = k.dims()[2];
  auto head_size = q.dims()[3];
  auto kv_seq_len = k.dims()[1];

  // bias dim: {b, h, s_q, s_kv}
  size_t bias_b = 0;
  size_t bias_h = 0;
  void *bias_dev_ptr = nullptr;
  void *dbias_dev_ptr = nullptr;
  if (bias_type != MHA_Bias_Type::NO_BIAS) {
    bias_b = bias_grad->dims()[0];
    bias_h = bias_grad->dims()[1];
    bias_dev_ptr =
        reinterpret_cast<void *>(const_cast<T *>(bias.get_ptr()->data<T>()));
    if (bias_grad != nullptr) {
      dev_ctx.template Alloc<T>(bias_grad);
      dbias_dev_ptr =
          reinterpret_cast<void *>(const_cast<T *>(bias_grad->data<T>()));
    }
  }

  void *q_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(q.data<T>()));
  void *k_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(k.data<T>()));
  void *v_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(v.data<T>()));
  void *dq_dev_ptr =
      reinterpret_cast<void *>(const_cast<T *>(q_grad->data<T>()));
  void *dk_dev_ptr =
      reinterpret_cast<void *>(const_cast<T *>(k_grad->data<T>()));
  void *dv_dev_ptr =
      reinterpret_cast<void *>(const_cast<T *>(v_grad->data<T>()));
  void *o_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(O.data<T>()));
  void *do_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(dO.data<T>()));
  void *softmax_out_dev_ptr =
      reinterpret_cast<void *>(const_cast<float *>(softmax_out.data<float>()));
  void *cu_seqlen_q_dev_ptr = reinterpret_cast<void *>(
      const_cast<int32_t *>(cu_seqlen_q.data<int32_t>()));
  void *cu_seqlen_kv_dev_ptr = reinterpret_cast<void *>(
      const_cast<int32_t *>(cu_seqlen_kv.data<int32_t>()));
  void *seed_dev_ptr = reinterpret_cast<void *>(
      const_cast<int64_t *>(rng_state.data<int64_t>()));
  void *offset_dev_ptr = reinterpret_cast<void *>(
      const_cast<int64_t *>(rng_state.data<int64_t>()) + 1);

  //  bool use_workspace_opt = false;
  //  if (sm_arch >= 90) {
  //    // quick estimate of dp workspace size
  //    size_t max_seqlen_div_up_q = ((q_seq_len + 64 - 1) / 64) * 64;
  //    size_t max_seqlen_div_up_kv = ((kv_seq_len + 64 - 1) / 64) * 64;
  //    size_t required_dp_workspace =
  //        (batch_size * num_heads * max_seqlen_div_up_q * max_seqlen_div_up_kv
  //        *
  //             2 +
  //         1048576 - 1) /
  //        1048576;
  //    // default upper limit for dp workspace 256MB
  //    size_t max_allowed_dp_workspace = 256;
  //    if (required_dp_workspace <= max_allowed_dp_workspace) {
  //      use_workspace_opt = true;
  //    }
  //    auto use_workspace_opt_str =
  //        std::getenv("CUDNN_FUSE_ATTN_USE_WORKSPACE_OPT");
  //    if (use_workspace_opt_str != nullptr) {
  //      use_workspace_opt =
  //      static_cast<bool>(std::stoi(use_workspace_opt_str));
  //    }
  //  }

  size_t workspace_size = 0;
  // call the first time to get the workspace size
  fused_attn_arbitrary_seqlen_bwd(batch_size,
                                  num_heads,
                                  num_heads_kv,
                                  q_seq_len,
                                  kv_seq_len,
                                  head_size,
                                  bias_b,
                                  bias_h,
                                  scaling_factor,
                                  dropout_probability,
                                  mha_layout,
                                  bias_type,
                                  mask_type,
                                  q_dev_ptr,
                                  k_dev_ptr,
                                  v_dev_ptr,
                                  o_dev_ptr,
                                  softmax_out_dev_ptr,
                                  bias_dev_ptr,
                                  dq_dev_ptr,
                                  dk_dev_ptr,
                                  dv_dev_ptr,
                                  do_dev_ptr,
                                  dbias_dev_ptr,
                                  seed_dev_ptr,
                                  offset_dev_ptr,
                                  cu_seqlen_q_dev_ptr,
                                  cu_seqlen_kv_dev_ptr,
                                  tensor_dtype,
                                  nullptr,
                                  &workspace_size,
                                  dev_ctx.stream(),
                                  handle);

  DenseTensor workspace;
  workspace.Resize({workspace_size > 0 ? workspace_size : 1});
  dev_ctx.template Alloc<char>(&workspace);
  fused_attn_arbitrary_seqlen_bwd(
      batch_size,
      num_heads,
      num_heads_kv,
      q_seq_len,
      kv_seq_len,
      head_size,
      bias_b,
      bias_h,
      scaling_factor,
      dropout_probability,
      mha_layout,
      bias_type,
      mask_type,
      q_dev_ptr,
      k_dev_ptr,
      v_dev_ptr,
      o_dev_ptr,
      softmax_out_dev_ptr,
      bias_dev_ptr,
      dq_dev_ptr,
      dk_dev_ptr,
      dv_dev_ptr,
      do_dev_ptr,
      dbias_dev_ptr,
      seed_dev_ptr,
      offset_dev_ptr,
      cu_seqlen_q_dev_ptr,
      cu_seqlen_kv_dev_ptr,
      tensor_dtype,
      reinterpret_cast<void *>(workspace.data<char>()),
      &workspace_size,
      dev_ctx.stream(),
      handle);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_dot_product_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDotProductAttentionKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(fused_dot_product_attention_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDotProductAttentionGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
