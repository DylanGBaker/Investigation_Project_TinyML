/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "network.h"
#include "network_data.h"
#include "good_image_data_3.h"
#include "math.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

CRC_HandleTypeDef hcrc;

TIM_HandleTypeDef htim14;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_CRC_Init(void);
static void MX_TIM14_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

//__attribute__((section(".sdram_data"))) const float good_image_data[10][30000];

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
	uint16_t timer_val = 0;

	int final_outputs[13];

	int true_positives = 0;
	int false_negatives = 0;

	int false_positives = 0;
	int true_negatives = 0;

	int accumulated_time = 0;

	float mae_loss = 0.0;
	const float threshold_loss = 0.10709624;

	const int output_length = 3072;
	uint8_t final_output_val = 0;

  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* Enable the CPU Cache */

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_CRC_Init();
  MX_TIM14_Init();
  /* USER CODE BEGIN 2 */

  // THIS IS FOR THE COMPRESSION VALUE OF LOW!!!!!!!

  static ai_handle network = AI_HANDLE_NULL;

      /* Global c-array to handle the activations buffer */
      AI_ALIGNED(32)
      static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

      /* Array to store the data of the input tensor */
      AI_ALIGNED(32)
      static ai_float in_data[AI_NETWORK_IN_1_SIZE];
      /* or static ai_u8 in_data[AI_NETWORK_IN_1_SIZE_BYTES]; */

      /* c-array to store the data of the output tensor */
      AI_ALIGNED(32)
      static ai_float out_data[AI_NETWORK_OUT_1_SIZE];
    //  static ai_u8 out_data[AI_NETWORK_OUT_1_SIZE_BYTES];

      /* Array of pointer to manage the model's input/output tensors */
      static ai_buffer *ai_input;
      static ai_buffer *ai_output;

      int aiInit(void) {
          ai_error err;

          /* Create and initialize the c-model */
          const ai_handle acts[] = { activations };
          err = ai_network_create_and_init(&network, acts, NULL);
          if (err.type != AI_ERROR_NONE) {}

          /* Reteive pointers to the model's input/output tensors */
          ai_input = ai_network_inputs_get(network, NULL);
          ai_output = ai_network_outputs_get(network, NULL);

          return 0;
        }

      int aiRun(const void *in_data, void *out_data) {
          ai_i32 n_batch;
          ai_error err;

          /* 1 - Update IO handlers with the data payload */
          ai_input[0].data = AI_HANDLE_PTR(in_data);
          ai_output[0].data = AI_HANDLE_PTR(out_data);

          /* 2 - Perform the inference */
          n_batch = ai_network_run(network, &ai_input[0], &ai_output[0]);
          if (n_batch != 1) {
              err = ai_network_get_error(network);
          };

          return 0;
        }




    double calculate_mae(ai_float input[], ai_float output[]){
    	float absolute_err_sum = 0.0;

    	for (int i = 0; i < output_length; i++){
    		absolute_err_sum += fabs(input[i] - output[i]);
    	}

    	return (absolute_err_sum / output_length);
    }


    uint8_t get_output_detection(){

    	if (mae_loss >= threshold_loss){
    		return 0;
    	}
    	else{
    		return 1;
    	}

    }


    void calculate_true_positives(int output[], int size){

    	for(int i = 0; i < size; i++){
    		if (output[i] == 0){
    			true_positives++;
    		}
    	}
    }


    void calculate_false_negatives(int output[], int size){

        	for(int i = 0; i < size; i++){
        		if (output[i] == 1){
        			false_negatives++;
        		}
        	}
        }

    void calculate_false_positives(int output[], int size){

            	for(int i = 0; i < size; i++){
            		if (output[i] == 0){
            			false_positives++;
            		}
            	}
            }

    void calculate_true_negatives(int output[], int size){

            	for(int i = 0; i < size; i++){
            		if (output[i] == 1){
            			true_negatives++;
            		}
            	}
            }



      aiInit();


      HAL_TIM_Base_Start(&htim14);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  for (int i = 0; i < 13; i++){
		  for (int j = 0; j < AI_NETWORK_IN_1_SIZE; j++){
		  	  	  		  ((ai_float*)in_data)[j] = good_image_data[i][j];
		  	  	  	  }

		  	  	  	  timer_val = __HAL_TIM_GET_COUNTER(&htim14);
		  	  	  	  aiRun(in_data, out_data);
		  	  	  	  timer_val = __HAL_TIM_GET_COUNTER(&htim14) - timer_val;
		  	  	  	  accumulated_time += timer_val;

		  	  	  	  mae_loss = calculate_mae(in_data, out_data);
		  	  	  	  final_output_val = get_output_detection();

		  	  	  	  final_outputs[i] = final_output_val;
	  }

	  //acc = get_accuracy(counter, 5.0);
//	  calculate_true_positives(final_outputs, 2);
//	  calculate_false_negatives(final_outputs, 2);

	  calculate_false_positives(final_outputs, 13);
	  calculate_true_negatives(final_outputs, 13);
	  accumulated_time = 0;


    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 216;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  hcrc.Init.DefaultPolynomialUse = DEFAULT_POLYNOMIAL_ENABLE;
  hcrc.Init.DefaultInitValueUse = DEFAULT_INIT_VALUE_ENABLE;
  hcrc.Init.InputDataInversionMode = CRC_INPUTDATA_INVERSION_NONE;
  hcrc.Init.OutputDataInversionMode = CRC_OUTPUTDATA_INVERSION_DISABLE;
  hcrc.InputDataFormat = CRC_INPUTDATA_FORMAT_BYTES;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief TIM14 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM14_Init(void)
{

  /* USER CODE BEGIN TIM14_Init 0 */

  /* USER CODE END TIM14_Init 0 */

  /* USER CODE BEGIN TIM14_Init 1 */

  /* USER CODE END TIM14_Init 1 */
  htim14.Instance = TIM14;
  htim14.Init.Prescaler = 107;
  htim14.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim14.Init.Period = 65535;
  htim14.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim14.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim14) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM14_Init 2 */

  /* USER CODE END TIM14_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
