# Deformable Convolution and Dilated Convolution Neural Network (DC2Net)

This is the code of the hyperspectral image classification network , used for early detection of crop diseases: Deformable Convolution and Dilated Convolution Neural Network (DC2Net).
The datasets required for the code can be found at soybean306.csv and soy3d_y.csv



## Model
![DC2Net](DC2Net.png)

The main idea of the proposed DC2Net is based on two novel designs: dilated convolution and deformable convolution fusion. These two designs give the proposed model three advantages: First, dilated convolution can expand the receptive field by introducing gaps in the standard convolutional kernels, allowing the convolutional layer to better capture sensitive wavelengths in the spectral dimension. Secondly, considering the randomly scattered appearance of soybean rust on leaves, we upgraded 2D deformable convolution to 3D deformable convolution. The research results demonstrated that, compared with 2D deformable convolution, 3D deformable convolution performed better in terms of accuracy. The integration of the deformable convolution and dilated convolution into the proposed DC2Net model allowed the extraction of sensitive spatial and spectral features, thereby enhancing the model's performance.


## Requirements 

 Python 3.9, Pytorch1.12.1, CUDA 11.3, cuDNN 8, NVCC, VNC

## License
Copyright (c) 2023 Jiarui Feng. Released under the MIT License.
