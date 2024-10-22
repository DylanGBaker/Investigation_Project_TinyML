# tinyML Investigation Project
Welcome to Shen Reddy and Dylan Bakers' 4th year tinyML investigation project. The repo contains Jupyter Notebooks where Python 3.9, Tensorflow 2.10.0, and Keras 2.10.1 were used to train three different machine learning models. The aim was to train them to detect anomalies in images. These models were then saved and exported so that they can be used on the STM32F769NI-DISCO board. We made use of the dataset provided by MVTec which can be found at this link: https://www.mvtec.com/company/research/datasets/mvtec-ad. They provide a large dataset to detect anomalies in images.

# Jupyter Notebooks
Out computers made use of the Windows OS. We also wanted to make use of GPU's for model training when we could. In order to do this with Windows, specific versions of Tensorflow need to be used due to limited GPU support with versions after 2.10. The steps for native Windows can be found at https://www.tensorflow.org/install/pip. If you do not have Tensorflow 2.10 we cannot say if all the code written will be compatible with a different version.

# Machine Learning
The three different machine learning models are as follows:

1. Neural Network
2. 2D Convolutional Neural Network
3. Autoencoder

The Jupyter Notebooks for the Neural Network and 2D Convolutional Neural Network can be found under the ```CV_Anomaly_Detection``` folder. The cable images used from MVTec are already in the folders. Do not move the folders unless you would like to change where you load the images from. The Jupyter Notebook for the Autoencoder can be found in the ```Autoencoder_Anomaly_Detection``` folder. Again, do not move the folders within unless you would like to change where you load the images from. When running the notebooks the outputs might be different from ours due to different hardware.


# STM32 Code
To run the STM32 code you need to make sure that you have STM32CubeIDE and STM32CubeMX installed. The IDE will allow you to see the code written and flash that code to an STM32 MCU. It also contains the ```.ioc``` file which shows how the pins have been assigned and the setup of the X-CUBE-AI extension. The ```.ioc``` file should give you all the necessary files to run the code, if you click generate code button in the STM32CubeIDE. Before generating the code make sure you have installed X-CUBE-AI 9.0.0 using STM32CubeMX. This is the specific version of X-Cube-AI used in the project. Once you have done that, open X-CUBE-AI in the .ioc file in the IDE. If you are not sure how to, you can follow the steps below:

1. Open the .ioc file within the STM project in STM32CubeIDE

2. Open the Middleware and Software list. You should see the something similar to the image below.

   
![XCUBE_Config_Option](https://github.com/user-attachments/assets/5d014107-2693-4a05-92f1-b9171d983c43)

3. Click on the X-CUBE-AI option and then open the dropdown for Artificial Intelligence X-CUBE-AI and check the box like in the image below.


![XCUBEAI_Choice](https://github.com/user-attachments/assets/f94a20b9-3333-43ca-8ae2-929f63644262)

4. Once you have done that, you should see a configuration tab appear on the right when you click on X-CUBE-AI. You can now click on ''add network'' and give a file location to a keras model.


![Browse_File_Options](https://github.com/user-attachments/assets/e535209c-df5c-48a3-9f1c-c8062054d852)


You need to make sure that the file path to the saved imported machine learning model you want to use is correct. It currently could be a wrong file path as you may have a different file structure. Please ensure that the saved keras model came from a Tensorflow 2.10 model. This version of Tensorflow uses a version of keras that is compatible with X-CUBE-AI 9.0.0. If you do not and an error occurs when analysing the model, you can convert the model to a Tensorflow Lite model and then STM should be able to analyse it. Once you have done this, either save the .ioc file or click on the generate code button in the STM32CubeIDE, and all the relevant files will be created for you so that the ```main.c``` file can run.
