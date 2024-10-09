/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Tue Oct  8 02:03:33 2024
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x856fbbff0af9427d0e2e62d2b362cb5b"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Tue Oct  8 02:03:33 2024"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_conv2d_input0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 3072, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  nl_8_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  upsample_9_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  nl_10_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  upsample_11_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  nl_12_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  upsample_13_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  nl_14_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  upsample_15_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32768, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  nl_17_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 3072, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 864, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 864, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_12_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_weights, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 6144),
  1, &conv2d_14_weights_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_bias, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_14_bias_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_weights, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 3), AI_STRIDE_INIT(4, 4, 128, 384, 1152),
  1, &conv2d_16_weights_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_bias, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &conv2d_16_bias_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_scratch0, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 32, 2), AI_STRIDE_INIT(4, 4, 4, 128, 4096),
  1, &conv2d_0_scratch0_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 2), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &conv2d_2_scratch0_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 2), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_scratch0, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 4, 2), AI_STRIDE_INIT(4, 4, 4, 8, 32),
  1, &conv2d_6_scratch0_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_conv2d_input0_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 32, 32), AI_STRIDE_INIT(4, 4, 4, 12, 384),
  1, &serving_default_conv2d_input0_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 4, 4, 128, 2048),
  1, &conv2d_0_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &conv2d_2_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 4, 4), AI_STRIDE_INIT(4, 4, 4, 32, 128),
  1, &conv2d_4_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 2, 2), AI_STRIDE_INIT(4, 4, 4, 8, 16),
  1, &conv2d_6_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 2, 2), AI_STRIDE_INIT(4, 4, 4, 8, 16),
  1, &conv2d_8_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  nl_8_nl_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 2, 2), AI_STRIDE_INIT(4, 4, 4, 8, 16),
  1, &nl_8_nl_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  upsample_9_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 4, 4), AI_STRIDE_INIT(4, 4, 4, 8, 32),
  1, &upsample_9_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 4, 4), AI_STRIDE_INIT(4, 4, 4, 32, 128),
  1, &conv2d_10_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  nl_10_nl_output, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 4, 4), AI_STRIDE_INIT(4, 4, 4, 32, 128),
  1, &nl_10_nl_output_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  upsample_11_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 8, 8), AI_STRIDE_INIT(4, 4, 4, 32, 256),
  1, &upsample_11_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &conv2d_12_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  nl_12_nl_output, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 4, 4, 64, 512),
  1, &nl_12_nl_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  upsample_13_output, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &upsample_13_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 4, 4, 128, 2048),
  1, &conv2d_14_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  nl_14_nl_output, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 4, 4, 128, 2048),
  1, &nl_14_nl_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  upsample_15_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 32, 32), AI_STRIDE_INIT(4, 4, 4, 128, 4096),
  1, &upsample_15_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_output, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 32, 32), AI_STRIDE_INIT(4, 4, 4, 12, 384),
  1, &conv2d_16_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  nl_17_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 32, 32), AI_STRIDE_INIT(4, 4, 4, 12, 384),
  1, &nl_17_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_weights, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 3, 3, 3, 32), AI_STRIDE_INIT(4, 4, 12, 384, 1152),
  1, &conv2d_0_weights_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_bias, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_0_bias_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 32, 3, 3, 16), AI_STRIDE_INIT(4, 4, 128, 2048, 6144),
  1, &conv2d_2_weights_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_2_bias_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 3, 8), AI_STRIDE_INIT(4, 4, 64, 512, 1536),
  1, &conv2d_4_weights_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_4_bias_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_weights, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 8, 3, 3, 2), AI_STRIDE_INIT(4, 4, 32, 64, 192),
  1, &conv2d_6_weights_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_bias, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &conv2d_6_bias_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_weights, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 2, 3, 3, 2), AI_STRIDE_INIT(4, 4, 8, 16, 48),
  1, &conv2d_8_weights_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_bias, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &conv2d_8_bias_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_weights, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 2, 3, 3, 8), AI_STRIDE_INIT(4, 4, 8, 64, 192),
  1, &conv2d_10_weights_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_bias, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_10_bias_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_weights, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 8, 3, 3, 16), AI_STRIDE_INIT(4, 4, 32, 512, 1536),
  1, &conv2d_12_weights_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_17_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_17_layer, 17,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_17_chain,
  NULL, &nl_17_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &upsample_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_16_weights, &conv2d_16_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_16_layer, 16,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_16_chain,
  NULL, &nl_17_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float upsample_15_scales_data[] = { 2.0, 2.0, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    upsample_15_scales, AI_ARRAY_FORMAT_FLOAT,
    upsample_15_scales_data, upsample_15_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  upsample_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_14_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &upsample_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  upsample_15_layer, 15,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_nearest,
  &upsample_15_chain,
  NULL, &conv2d_16_layer, AI_STATIC, 
  .scales = &upsample_15_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_NEAREST, 
  .nearest_mode = AI_ROUND_FLOOR, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_14_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_14_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_14_nl_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_14_nl_chain,
  NULL, &upsample_15_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &upsample_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_14_weights, &conv2d_14_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_14_layer, 14,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_14_chain,
  NULL, &nl_14_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float upsample_13_scales_data[] = { 2.0, 2.0, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    upsample_13_scales, AI_ARRAY_FORMAT_FLOAT,
    upsample_13_scales_data, upsample_13_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  upsample_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_12_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &upsample_13_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  upsample_13_layer, 13,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_nearest,
  &upsample_13_chain,
  NULL, &conv2d_14_layer, AI_STATIC, 
  .scales = &upsample_13_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_NEAREST, 
  .nearest_mode = AI_ROUND_FLOOR, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_12_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_12_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_12_nl_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_12_nl_chain,
  NULL, &upsample_13_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &upsample_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_12_weights, &conv2d_12_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_12_layer, 12,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_12_chain,
  NULL, &nl_12_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float upsample_11_scales_data[] = { 2.0, 2.0, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    upsample_11_scales, AI_ARRAY_FORMAT_FLOAT,
    upsample_11_scales_data, upsample_11_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  upsample_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_10_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &upsample_11_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  upsample_11_layer, 11,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_nearest,
  &upsample_11_chain,
  NULL, &conv2d_12_layer, AI_STATIC, 
  .scales = &upsample_11_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_NEAREST, 
  .nearest_mode = AI_ROUND_FLOOR, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_10_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_10_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_10_nl_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_10_nl_chain,
  NULL, &upsample_11_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &upsample_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_10_weights, &conv2d_10_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_10_layer, 10,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_10_chain,
  NULL, &nl_10_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float upsample_9_scales_data[] = { 2.0, 2.0, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    upsample_9_scales, AI_ARRAY_FORMAT_FLOAT,
    upsample_9_scales_data, upsample_9_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  upsample_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_8_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &upsample_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  upsample_9_layer, 9,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_nearest,
  &upsample_9_chain,
  NULL, &conv2d_10_layer, AI_STATIC, 
  .scales = &upsample_9_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_NEAREST, 
  .nearest_mode = AI_ROUND_FLOOR, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_8_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_8_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_8_nl_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_8_nl_chain,
  NULL, &upsample_9_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_8_weights, &conv2d_8_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_8_chain,
  NULL, &nl_8_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_6_weights, &conv2d_6_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_6_layer, 7,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_6_chain,
  NULL, &conv2d_8_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 5,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_4_chain,
  NULL, &conv2d_6_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 3,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_2_chain,
  NULL, &conv2d_4_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_conv2d_input0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_0_weights, &conv2d_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_0_layer, 1,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_0_chain,
  NULL, &conv2d_2_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 54764, 1, 1),
    54764, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 163840, 1, 1),
    163840, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_conv2d_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_17_output),
  &conv2d_0_layer, 0xa6d0b9b8, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 54764, 1, 1),
      54764, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 163840, 1, 1),
      163840, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_conv2d_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_17_output),
  &conv2d_0_layer, 0xa6d0b9b8, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_conv2d_input0_output_array.data = AI_PTR(g_network_activations_map[0] + 37056);
    serving_default_conv2d_input0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 37056);
    conv2d_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 49344);
    conv2d_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 49344);
    conv2d_0_output_array.data = AI_PTR(g_network_activations_map[0] + 10432);
    conv2d_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10432);
    conv2d_2_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 43200);
    conv2d_2_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 43200);
    conv2d_2_output_array.data = AI_PTR(g_network_activations_map[0] + 45248);
    conv2d_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 45248);
    conv2d_4_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 10432);
    conv2d_4_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 10432);
    conv2d_4_output_array.data = AI_PTR(g_network_activations_map[0] + 10944);
    conv2d_4_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10944);
    conv2d_6_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 10432);
    conv2d_6_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 10432);
    conv2d_6_output_array.data = AI_PTR(g_network_activations_map[0] + 10496);
    conv2d_6_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10496);
    conv2d_8_output_array.data = AI_PTR(g_network_activations_map[0] + 10432);
    conv2d_8_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10432);
    nl_8_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 10464);
    nl_8_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10464);
    upsample_9_output_array.data = AI_PTR(g_network_activations_map[0] + 10496);
    upsample_9_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10496);
    conv2d_10_output_array.data = AI_PTR(g_network_activations_map[0] + 10624);
    conv2d_10_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10624);
    nl_10_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 11136);
    nl_10_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 11136);
    upsample_11_output_array.data = AI_PTR(g_network_activations_map[0] + 11648);
    upsample_11_output_array.data_start = AI_PTR(g_network_activations_map[0] + 11648);
    conv2d_12_output_array.data = AI_PTR(g_network_activations_map[0] + 13696);
    conv2d_12_output_array.data_start = AI_PTR(g_network_activations_map[0] + 13696);
    nl_12_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 17792);
    nl_12_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 17792);
    upsample_13_output_array.data = AI_PTR(g_network_activations_map[0] + 21888);
    upsample_13_output_array.data_start = AI_PTR(g_network_activations_map[0] + 21888);
    conv2d_14_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_14_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_14_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_14_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    upsample_15_output_array.data = AI_PTR(g_network_activations_map[0] + 32768);
    upsample_15_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32768);
    conv2d_16_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_16_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_17_output_array.data = AI_PTR(g_network_activations_map[0] + 12288);
    nl_17_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12288);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_0_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_0_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 3456);
    conv2d_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 3456);
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(g_network_weights_map[0] + 3584);
    conv2d_2_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 3584);
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(g_network_weights_map[0] + 22016);
    conv2d_2_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 22016);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(g_network_weights_map[0] + 22080);
    conv2d_4_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 22080);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(g_network_weights_map[0] + 26688);
    conv2d_4_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 26688);
    conv2d_6_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_weights_array.data = AI_PTR(g_network_weights_map[0] + 26720);
    conv2d_6_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 26720);
    conv2d_6_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_bias_array.data = AI_PTR(g_network_weights_map[0] + 27296);
    conv2d_6_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 27296);
    conv2d_8_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_weights_array.data = AI_PTR(g_network_weights_map[0] + 27304);
    conv2d_8_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 27304);
    conv2d_8_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_bias_array.data = AI_PTR(g_network_weights_map[0] + 27448);
    conv2d_8_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 27448);
    conv2d_10_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_10_weights_array.data = AI_PTR(g_network_weights_map[0] + 27456);
    conv2d_10_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 27456);
    conv2d_10_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_10_bias_array.data = AI_PTR(g_network_weights_map[0] + 28032);
    conv2d_10_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 28032);
    conv2d_12_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_12_weights_array.data = AI_PTR(g_network_weights_map[0] + 28064);
    conv2d_12_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 28064);
    conv2d_12_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_12_bias_array.data = AI_PTR(g_network_weights_map[0] + 32672);
    conv2d_12_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 32672);
    conv2d_14_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_weights_array.data = AI_PTR(g_network_weights_map[0] + 32736);
    conv2d_14_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 32736);
    conv2d_14_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_bias_array.data = AI_PTR(g_network_weights_map[0] + 51168);
    conv2d_14_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 51168);
    conv2d_16_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_16_weights_array.data = AI_PTR(g_network_weights_map[0] + 51296);
    conv2d_16_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 51296);
    conv2d_16_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_16_bias_array.data = AI_PTR(g_network_weights_map[0] + 54752);
    conv2d_16_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 54752);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 4395983,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xa6d0b9b8,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 4395983,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xa6d0b9b8,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

