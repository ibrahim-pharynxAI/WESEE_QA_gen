## 3 Implementation

$$R e c a l l ( S e n s i t i v i t y ) = T P T P + F N$$

$$F _ 1 - s c o r e = 2 P r e c i s i o n T P + F P$$

The above formulae utilize True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) for computing performance metrics.

A custom CNN model was selected for each dataset and trained using our proposed strategy. Different pooling and activation strategies were employed to generate three distinct probability scores. The input was simultaneously passed through three parallel channels of the model, each utilizing a different pooling strategy, MaxPooling, MinPooling, and our novel MaxMinPooling, as illustrated in Fig. 3.

The MaxPooling model works by taking the maximum value over an input window, using a positive ReLU activation function at the output. In contrast, the MinPooling model emphasizes the minimum values in the feature maps and uses a negative ReLU activation function to retain the

Fig. 6 A block diagram of the custom CNN model for classifying X-ray and CT scan images into three categories: COVID-19, normal, and pneumonia. Conv2D refers to convolutional layer 2D followed by activation function, and BN stands for batch normalization

<!-- image -->

negative coefficients. Our innovative MaxMinPooling model combines both MaxPooling and MinPooling outputs using an interleaving approach, incorporating both positive and negative ReLU activation functions to retain both sets of features through concatenation.

of our proposed models, developed empirically to achieve optimal classification performance. The training process for each model is meticulously documented to ensure a clear understanding of the approach and its effectiveness in fitting well with the training curves:

- Data Preparation: All images were resized to uniform 32x32 pixels and normalized to have pixel values between 0 and 1 for CIFAR datasets. The input sizes for X-ray and CT scan images were 160x160 and 224x224, respectively. Various data augmentation techniques were applied to increase dataset robustness, as detailed in Table 3.
- Model Training: Training was conducted on an i9 processor with 16 GB NVIDIA RTX3080 Ti GPGPU to expedite the computation. The models were evaluated at the end of each epoch using a hold-out validation set, comprising 20% of the training data, to monitor performance and adjust hyperparameters if necessary.
- Optimizations and Monitoring: Real-time monitoring of the training process was enabled through TensorFlow callbacks, allowing for immediate adjustments based on performance metrics like loss and accuracy.
- Software Tools: The implementation was carried out using CUDA 11.8, Python 3.8, TensorFlow 2.4 and Keras as the primary libraries for building and training the neural networks.

Figures 5 and 6 showcase block diagrams of each custom CNN model tailored to the CIFAR-10, CIFAR-100, and medical (X-ray and CT scan) datasets, respectively. These diagrams clarify the intricate architecture of the pooling and activation strategies and the operational flow

The hyperparameters for each model were selected based on an empirical intuition derived from prior studies and multiple iterations. The learning rate was optimized using an adaptive learning rate method (Adam optimizer) to adjust as training progresses. Batch size was carefully chosen to balance the trade-off between training speed and

Table 3 Experimental hyperparameter specifications for the proposed MaxMinPooling model

| Hyperparameters    | CIFAR-10                                                                 | CIFAR-100                                            | X-ray                | CT Scan            |
|--------------------|--------------------------------------------------------------------------|------------------------------------------------------|----------------------|--------------------|
| Batchsize          | 64                                                                       | 64                                                   | 128                  | 128                |
| Learning rate      | 0.001                                                                    | 0.001                                                | 0.00001              | 0.00001            |
| Decay rate         | 0.9                                                                      | 0.91                                                 | 0.89                 | 0.827              |
| Optimizer          | Adam                                                                     | Adam                                                 | Adam                 | Adam               |
| Kernel_initializer | He_normal                                                                | He_Uniform                                           | He_normal            | He_normal          |
| Kernel_regularizer | L2(0.001)                                                                | -                                                    | L2(0.001)            | L2(0.001)          |
| Activation         | Softmax                                                                  | Softmax                                              | Softmax              | Softmax            |
| Loss               | categorical                                                              | categorical                                          | sparse_categorical   | sparse_categorical |
| Early stopping     | Yes (patience =20)                                                       | Yes (patience =10)                                   | Yes (patience =10)   | Yes (patience =10) |
| Library            | Tensorflow, keras                                                        | Tensorflow, keras                                    | Tensorflow, keras    | Tensorflow, keras  |
| Epochs             | 200                                                                      | 200                                                  | 100                  | 100                |
| Input size         | (32,32)                                                                  | (32,32)                                              | (160,160)            | (224,224)          |
| Augmentation       | rotation_range(20), horizontal_flip, width_shift(0.2), height_shift(0.2) | horizontal_flip, width_shift(0.2), height_shift(0.2) | RandomContrast (0.8) | -                  |

Table 4 Ablation study of performance metrics for various approaches on CIFAR-10 and CIFAR-100 datasets

| Pooling method        | Activation function   | CIFAR-10 dataset   | CIFAR-10 dataset   | CIFAR-10 dataset   | CIFAR-10 dataset   | CIFAR-100 dataset   | CIFAR-100 dataset   | CIFAR-100 dataset   | CIFAR-100 dataset   |
|-----------------------|-----------------------|--------------------|--------------------|--------------------|--------------------|---------------------|---------------------|---------------------|---------------------|
|                       |                       | Accuracy           | Precision          | Recall             | F1_Score           | Accuracy            | Precision           | Recall              | F1_Score            |
| MaxPooling            | Pos Relu              | 88.72              | 88.76              | 88.72              | 88.59              | 59.07               | 59.40               | 59.07               | 58.71               |
| MinPooling            | Neg Relu              | 89.08              | 89.05              | 89.08              | 88.94              | 58.66               | 58.78               | 58.66               | 58.27               |
| MaxMinPooling         | PosNeg Relu           | 88.80              | 88.81              | 88.80              | 88.69              | 63.96               | 65.04               | 64.72               | 64.37               |
| MinMaxPooling         | NegPos Relu           | 88.72              | 88.76              | 88.72              | 88.58              | 63.14               | 63.33               | 63.14               | 62.80               |
| Probabilistic Average | -                     | 90.08              | 90.10              | 90.10              | 89.96              | 64.72               | 65.04               | 64.72               | 64.37               |

Table 5 Ablation study of performance metrics of the experimented methods on X-ray and CT scan dataset

| Methods               | Activation function   | X-ray dataset   | X-ray dataset   | X-ray dataset   | X-ray dataset   | CT scan dataset   | CT scan dataset   | CT scan dataset   | CT scan dataset   |
|-----------------------|-----------------------|-----------------|-----------------|-----------------|-----------------|-------------------|-------------------|-------------------|-------------------|
|                       |                       | Accuracy        | Precision       | Recall          | F1_Score        | Accuracy          | Precision         | Recall            | F1_Score          |
| MaxPooling            | Pos Relu              | 88.60           | 88.60           | 88.55           | 88.50           | 95.61             | 96.34             | 96.54             | 96.40             |
| MinPooling            | Neg Relu              | 88.52           | 88.42           | 88.43           | 88.40           | 95.32             | 96.18             | 96.37             | 96.21             |
| MaxMinPooling         | PosNeg Relu           | 90.67           | 90.71           | 90.66           | 90.60           | 96.72             | 97.37             | 97.44             | 97.38             |
| MinMaxPooling         | NegPos Relu           | 90.0            | 90.0            | 90.0            | 89.95           | 96.57             | 97.20             | 97.30             | 97.24             |
| Probabilistic Average | -                     | 90.26           | 90.22           | 90.20           | 90.15           | 96.34             | 97.08             | 97.16             | 97.08             |

model performance stability. Each model was trained for up to 200 epochs, with early stopping implemented to prevent overfitting if the validation loss ceased to decrease for ten consecutive epochs. A detailed list of optimal hyperparameters for the MaxMinPooling model for the four different datasets is tabulated in Table 3, providing a comprehensive overview of the configuration settings employed in the experiments that ensures the reliability and reproducibility of the results.

The model training involved a rigorous process of selecting and iterating through various combinations of hyperparameters. The optimally tuned hyperparameters are detailed in Table 3. Additionally, learning curves for the datasets utilizing the proposed MaxMinPooling method are depicted in Fig. 8. These figures illustrate that both the training and validation curves for accuracy and loss converge as the number of epochs increases, signaling a wellfitting model that generalizes effectively across different data scenarios.
