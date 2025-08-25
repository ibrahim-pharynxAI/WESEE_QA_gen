## Deep Convolutional Neural Network based Ship Images Classification

Narendra Kumar Mishra * , Ashok Kumar, and Kishor Choudhury

Weapons   and Electronics Systems Engineering Establishment, New Delhi - 110 066, India * E-mail: narendra.lal@hqr.drdo.in

## ABSTRACT

Ships are an integral part of maritime traffic where they play both militaries as well as non-combatant roles. This vast maritime traffic needs to be managed and monitored by identifying and recognising vessels to ensure the maritime safety and security. As an approach to find an automated and efficient solution, a deep learning model exploiting convolutional neural network (CNN) as a basic building block, has been proposed in this paper. CNN has been predominantly used in image recognition due to its automatic high-level features extraction capabilities and exceptional performance. We have used transfer learning approach using pre-trained CNNs based on VGG16 architecture  to  develop  an  algorithm  that  performs  the  different  ship  types  classification. This  paper  adopts  data augmentation and fine-tuning to further improve and optimize the baseline VGG16 model. The proposed model attains  an  average  classification  accuracy  of  97.08%  compared  to  the  average  classification  accuracy  of  88.54% obtained from the baseline model.

Keywords: Ship classification; Convolutional neural network; Transfer learning; VGG16

## 1. INTRODUCTION

Apart from their conventional roles, modern naval forces are  also  actively  involved  in  maritime  security  operations, including  monitoring,  tracking,  detecting,  and  identifying ocean traffic, efficiently and effectively.

AI has turned up as one of the most promising technologies across diverse fields. In this paper, CNN based deep learning algorithm has been studied, and its performance is evaluated and analysed.

Vessel movements are currently monitored using automatic  identification  system  (AIS) 1 , synthetic aperture radar  (SAR) 2 ,  satellite-based  images 3-4 ,  and  optical  images captured by cameras. SAR or satellite images give the full view of maritime vessels and cover larger ocean areas than optical images.  For  maritime  surveillance,  the  optical  image-based classification would be an efficient solution due to its simplicity and easy availability. However, its successful realisation using conventional methods faces many challenges such as degraded quality of images due to environmental factors, the resemblance in the look and form of the class of ships, and the vastness of the ocean environment.

These factors call for a more reliable technology or system which can automatically classify ships based on their features, where artificial intelligence (AI) can play a significant role. AI system is capable of automatic identification and recognition of marine vessels and objects around it, like navigation-aids, boats, etc. that can lead to the enhanced situational awareness.

## 1.1  Convolutional  Neural  Network

With  the  advancement  in  technology,  in  terms  of  more robust  algorithms,  availability  of  large  volume  of  structured datasets, and the capability of handling large volumes of data more  efficiently  through  graphical  processing  units  (GPUs),

Received : 25 August 2020, Revised : 19 January 2021

Accepted : 01 February 2021, Online published : 10 March 2021

In  CNN  based  models,  input  images  are  minimally processed and fed directly to the system, where a suitable group of features is extracted through a learning 5 .  This CNN capability allows for  the  cascading  of  several  CNN  layers  making  it  a 'Deep' feature extractor while learning the essential features for the particular problem of interest. In Deep learning models, convolutional layers learn more generic features in the initial stages and learn features specific to the input training dataset in deeper stages that are further utilised to classify the test images that were not part of the training dataset. It is predominantly used in a broad range of image recognition applications due to its automatic high-level feature extraction capabilities and exceptional performance 6 .

Fundamentally, CNN architecture consists of sequences of layers that transform the pixel values of input images through various  processes  to  final  class  scores.  The  process  flow architecture of a typical CNN, and its basic building blocks are shown in Fig. 1. The details of fundamental blocks of CNN 7 are described as follows:

- (i) Convolutional Layer: This layer is responsible for learning the  important features from input images. It consists of several  learnable  filters  or  kernels  that  slides  spatially across  the  input  image  and  calculates  the  dot  products as  a  response  called  the  feature  maps.  A  schematic implementation  of  the  convolution  operation  is  shown in  Fig.  2.  Two  important  features  of  the  convolutional layer are local connectivity (at a time, filter weights are multiplied  to  only  a  local  area  of  the  input  image)  and

<!-- image -->

Figure 1. Process flow architecture of typical convolution neural network.

<!-- image -->

## 1.2    Transfer  Learning

Figure 2. Schematic representation  of  convolution  operation  in  a convolutional layer.

It is very uncommon to train a convolutional network from scratch because it needs a sufficiently large training dataset and GPU to execute and evaluate the deep learning model. Alternatively, a transfer learning approach can be used for a new classification task. In the transfer learningbased approach, the pretrained model weights, which have already  been  trained  optimally  on  similar  problems,  are used for the new image recognition task. A transfer learning approach is schematically represented in Fig. 5. In transfer learning,  either  a  convolutional  network  pretrained  using

Weight Sharing (the same filter weights are multiplied to every spatial location of the input image).  A Convolutional layer output is fed to the activation function (e.g., RelU) that  introduces  non-linearity  into  the  artificial  neural network.  The  output  size  of  the  convolutional  layer depends on the following four Hyperparameters:

millions of images could be used as a fixed feature extractor (where  pretrained  weights  of  the  convolutional  blocks  are used as it is for the particular classification task of interest), or weights of the pretrained network can be fine-tuned for the specific  dataset/problem. The  selection  of  a  specific  transfer

- Number of filters, K
- Filter size, F x F
- Amount of zero padding, P
- Stride, S

For an input volume of size W 1x H 1x D 1, the convolutional layer results in an output volume of the size W 2x H 2x D 2 that can be calculated as

$$\begin{matrix} \dots \dots - \dots \dots \dots \dots \dots \dots \\ W 2 = \frac { W 1 - F + 2 P } { S } + 1 \\ H 2 = \frac { H 1 - F + 2 P } { S } + 1 \\ D 2 = K \\ \dots \dots \end{matrix} \begin{matrix} \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \\ \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \ dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \\ \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \\ \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \\ \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots  \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \Dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots  \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \\ \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots & \end{matrix}$$

- (ii) Pooling  Layer: This  layer  downsamples  the  input image's  spatial  dimension  and  is  placed  in-between two convolutional layers. Pooling minimises the computational complexity by reducing the learnable network parameters. MaxPooling and AveragePooling are  two  prominently  used  Pooling  techniques,  as depicted in Fig. 3.
- (iii) Fully-Connected Layer: This layer performs the actual  prediction  (classification  or  regression)  job. It  consists  of  fully  connected  layers  as  a  regular artificial neural network followed by a Softmax layer (final  output  layer)  that  provides  the  class  scores. It  consists  of  input  layer,  output  layer  and  number of hidden layers as shown in Fig. 4. The number of hidden layers and number of nodes in each layer are Hyperparameters.

Figure 3.  Schematic representation of pooling operation in a convolutional layer, maxpooling (top), and AveragePooling (bottom).

<!-- image -->

Figure 4. Schematic  representation  of  neural  network  with  fully connected layers.

<!-- image -->

Figure 5. Schematic representation of the Transfer Learning approach.

<!-- image -->

learning approach depends upon several factors 8 that includes size and similarity of the new dataset compared to the original dataset  and  is  tabulated  in  Table  1.  Due  to  the  inadequacy of  a  sufficiently  large  dataset  and  GPU  availability,  transfer learning has been used in the present study considering Case 3 for implementation.

blocks  can  be  used  as  it  is  for  the  particular classification task. Therefore, to make the VGG16 model more relevant and specific to the classification of ship's images, the last convolution block of the VGG16 network has been re-trained. However,  a  relatively  small  number  of  images in  the  training  of  the  model  leads  to  overfitting of  the  model  that  has  been  verified  empirically also.  To  mitigate  this  issue,  we  have  used  two important process improvement techniques; the  first  is  'batchNormalisation'. The  second  is 'Dropout.' It is pertinent to mention that process improvement techniques, batchNormalisation and Dropout, were not implemented in the standard VGG16 model.

Process improvement techniques cannot be incorporated directly into the pretrained convolutional  blocks  of  the  standard  VGG16

There are several freely accessible top-performing models,  like  VGG16 9 ,  ResNet50 10 ,  Inception 11 ,  Xception 12 , InceptionResNet 13 ,  and  DenseNet 14 ,  which  can  be  readily integrated  into  a  new  image  recognition  task.  In  the  present study,  the  transfer  learning  approach  based  on  the  standard VGG16 model  has  been  used  as  a  baseline  model  for  ship image  classification.  Originally,  VGG16  was  trained  using ImageNet  that  consists  of  millions  of  images  with  1000 categories. In our study, the marine vessel images are smaller in  numbers and dissimilar in relation to the original dataset. Therefore, the standard VGG16 could not be used as a fixed feature extractor, where pretrained weights of the convolutional model.  Therefore,  an  additional  convolutional  block  similar to  the  convolutional  blocks  of  standard  VGG16  has  been appended  in  the  proposed  model  to  incorporate  process improvement techniques. An extra convolutional block would also lead to learning more specific features of the input training  dataset  as  the  convolutional  layer  goes  into  deeper stages. Process improvement  techniques have also been incorporated into the classification block consisting of FullyConnected layers. The VGG16 model has been further built upon  by  data  augmentation  and  fine-tuning  of  the  network Hyperparameters. Proposed model has been evaluated against the baseline model.

## 2.    RELATED WORK

In  the  recent  past,  several  efforts  have  been  made  to classify  maritime  vessels'  optical  images  using  CNN  based deep  learning  algorithms.  In  reference 15 ,  CNN  trained  on

Table 1. Choice of transfer learning approach depending upon the similarity and size of the images in the new problem of a statement as compared to the original dataset

| Case   | Factors ( of new dataset compared to the original dataset)   | Factors ( of new dataset compared to the original dataset)   | Choice of transfer learning approach                                                                                  | Explanation                                                                                                                                                                                                                                                     |
|--------|--------------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Case   | Similarity                                                   | Size                                                         | Choice of transfer learning approach                                                                                  | Explanation                                                                                                                                                                                                                                                     |
| (1)    | Similar                                                      | Smaller                                                      | Train only a classifier layer using pretrained weights                                                                | Fine-tuning of the pretrained weights would lead to the overfitting problem due to the small dataset. As images in the new problem of interest have similarities with the original dataset, features learned by the pretrained weights would still be relevant. |
| (2)    | Similar                                                      | larger                                                       | The model can be fine-tuned through the full network.                                                                 | Since the new dataset is sufficiently large, re-training would not suffer overfitting issues.                                                                                                                                                                   |
| (3)    | Dissimilar                                                   | Smaller                                                      | Few convolutional layers, including the classifier layer, can be fine-tuned.                                          | Since the new dataset is small, only the classifier layer to be re- trained. However, due to the new dataset's difference compared to the original dataset, few convolutional layers need to be re- trained to learn the features specific to the new dataset.  |
| (4)    | Dissimilar                                                   | larger                                                       | Amodel can be developed from scratch, or transfer learning can be utilised by fine-tuning through the entire network. | Amodel can be trained from scratch due to the availability of a large data set. Alternatively, transfer learning can be utilised by fine-tuning through the entire network.                                                                                     |

AlexNet, Inception, and ResNet50 has been developed using  the  MARVEl  dataset 16 ,  a  large-scale  image  dataset for  maritime  vessels.  MARVEl  dataset  is  a  huge  collection of  marine  vessels  consisting  of  2  million  images  from  ship photos and ship tracker website 17 .  Ship  classification 18 using AlexNet  model  for  ten  categories  of  vessels  using  images from the same website has been developed. More studies 19-20 have  been  undertaken  using  images  from  the  same  website. Although  these  studies  have  used  transfer  learning  based architectures  and  have  used  images  from  the  same  website, one-to-one  performance  comparison  with  the  present  study cannot be undertaken due to lack of uniformity in the datasets.

## 3.    EXPERIMENTAL DESIGN

## 3.1  Dataset

The first challenge in training and validating the proposed model  was  the  availability  of  authentic  and  labelled  images of  ships  for  classification  purposes.  To  ensure  this,  for  our experiment purpose, we obtained the dataset by downloading ship images from the aforementioned website.

The website consists of a large number of vessel images for each category. To reduce our model's processing complexity; we have compiled a class balanced dataset comprised of 2400 images  of  four  classes:  aircraft  carrier,  Crude  oil  tankers, Cruise ships &amp; liners, and Destroyers. A few images from the training dataset for each category are demonstrated in Fig. 6. The complete dataset has been distributed in a proportion of 80:20 for training and testing of the proposed model. Twenty percent  of  the  training  dataset  has  been  further  utilised  for validation purposes. Description of a number of the training and  testing images  from  the dataset are  enumerated  in Table 2. All the images were saved by keeping the pixel size of 224 of the image's largest dimension without affecting the pixel qualities as the standard VGG16 model was developed using an input image size of 224x224.

Aircraft Carriers            Crude Oil Tankers            Cruise Ships and liners                Destroyers

<!-- image -->

Figure 6. Sample images of four classes of ships from the dataset.

Table 2. Description of a number of training and test images from the dataset.

| Class                   |   Train |   Test |
|-------------------------|---------|--------|
| Aircraft carriers       |     480 |    120 |
| Crude oil tankers       |     480 |    120 |
| Cruise ships and liners |     480 |    120 |
| Destroyers              |     480 |    120 |
| Total                   |    1920 |    480 |

## 3.2  Neural  Network Architecture

In this part, the  baseline  model based  on VGG16 architecture and various techniques that we incorporated in the proposed model to improve the classification performance has been described.

## 3.2.1 Baseline  Model  (VGG16)

VGG16  model  has  been  used  as  a  baseline  model developed by the Visual Graphics Group (VGG) at Oxford. It comprises of series of Convolutional layers and MaxPooling layers as its primary element connected in a pattern, as shown in  Fig.  7.  Convolution  blocks  1  and  2,  each  comprise  two convolutional layers and one MaxPooling layer in succession, as a feature extractor. Similarly, Convolution blocks 3, 4, and 5, each include three convolutional layers and one MaxPooling layer  in  sequence.  The  final  block  6  consists  of  three  fully connected  layers  and  a  Softmax  layer  in  succession  as  a classifier.  It  is  important  to  note  that  the  Dropout  and  batch Normalisation  steps  were  not  implemented  in  the  standard VGG16 model. Since data augmentation does not form part of the actual VGG16 model, it has been also incorporated into the baseline model.

## 3.2.2 Data  Augmentation  and  Fine-tuning  of  VGG16 Model

The performance of the VGG16 model has been further improved upon by incorporating various process improvement techniques, as discussed below:

(a)  Data  Augmentation: Data  augmentation  has  been employed  to  achieve  diverse  feature  learning  by  adding individual variations in the images so that the same kinds of images are not fed in each epoch during the learning process. Data augmentation has been applied very carefully to generate a  new  set  of  images  to  augment  the  training  dataset  while preserving  the  basic  features.  The  various  kinds  of  random variations incorporated into the training dataset include zooming, rotation, shift, and horizontal/vertical flips.

(b)  Re-train  the  weights  of  the  VGG16  model: VGG16 was designed to extract fine-grained features of objects from 1,000 categories. As the higher-order features learned by the model corresponds to the ImageNet dataset that may not be directly relevant to the classification of optical images of the ships, we have re-trained some convolution blocks of VGG16 to fine-tune the weights for our classification task.

(c) Fine-Tuning the Model: Following Hyperparameters have  been  fine-tuned  to  improve  the  performance  of  the baseline model:

- (i) Number  of Layers: Classification accuracy may  be improved by increasing the number of hidden layers and

Figure 7. Process flow architecture of standard VGG16 model.

<!-- image -->

numbers of nodes in each layer as it enhances the model capacity. However, it has been observed empirically that a deeper network lead to the overfitting of the model, higher complexity, and more training time due to the increased number of learnable parameters. Therefore, the impact of the number of hidden layers and nodes in each layer is evaluated empirically, and optimal numbers were chosen accordingly.

- (ii) Learning Rates: learning rate is one of the vital Hyperparameter  that  needs  careful  selection.  Through experiments, it has been observed that a small learning rate  causes  the  trapping  and  slow  down  of  the learning  process;  whereas,  a  large  value  of  learning rate leads to quick and non-optimal convergence. An optimal  value  of  the  learning  rate  has  been  chosen empirically.
- (c) batch Normalisation and Dropout have been embedded into the block-6 consisting of fully connected layers.

The process flow architecture of the proposed model is presented in Fig. 8.

## 3.2.4 Experimental  Parameters

The proposed model has been trained and evaluated on the  Google  Colab  cloud  server.  The  Hyperparameter  values have been tuned optimally in multiple iterations while training our  model.  Details  of  the  final  experimental  parameters  are tabulated in Table 3.

- (iii) Batchsize: batchsize is the number of training samples fed to the gradient descent algorithm in determining the  error  gradient.  It  is  a  vital  Hyperparameter  that influences the learning algorithm's dynamics.
- (iv) BatchNormalisation: batchNormalisation performs the normalisation (shifting and scaling) of the output from a convolutional layer before feeding it to the next layer that reduces the covariance shift of the network 21 . It speeds up the learning process of an artificial neural network with enhanced stability.
- (v) Regularisation: A major challenge in the development of  any  deep  learning  model  is  to  overcome  the overfitting  problem  so  that  it  may  generalize  well on  the new  dataset.  To  mitigate  this issue,  two prominent  regularisation  techniques,  Dropout 22 and Early Stopping, have been used in this paper. These techniques  not  only  reduce  overfitting  but  can  also lead  to  the  faster  optimisation  and  better  overall performance.

## 3.2.3 Proposed  Model

During  the  performance  analysis  of  the  baseline model,  several  significant  observations  have  been  made. During  the  training  process,  cross-entropy  loss  was  first decreasing;  however,  it  started  increasing  after  a  certain number of epochs. It is  also  observed  that  there  exists  a substantial gap between the graph of training and validation accuracy. The model achieved very high training accuracy but  performed poorly on the test dataset. This behaviour clearly  indicates  the  overfitting  of  the  model.  To  further improve  the  performance,  the  following  modifications were incorporated in the proposed model:

- (a) Weights of the convolution block-5 are re-trained so that the model will be more suitable and efficient for the ship classification task.
- (b) An additional  convolution  block  consisting  of  three consecutive  convolutional  layers  and  a  MaxPooling layer  has  been  inserted  before  the  block  of  fully connected layers. This block has been primarily used to incorporate batchNormalisation and Dropout to avoid overfitting in the model and assist in learning of higherorder features.

Figure 8. Process flow architecture of the proposed model.

<!-- image -->

## 4.      ANALYSIS AND  RESULTS

both  baseline  and  proposed  models  have  been  trained on  Google  Colab  using  Hyperparameters  as  listed  in  Table 3  for  the  same  input  dataset.  The  details  of  classification performance  measures  for  both  the  models  are  tabulated  in Table 4. It is to be noted that the Early Stopping criterion takes almost twice the number of epochs to exit the training process in  the  proposed model. A gap of 4.3% between training and validation accuracy in the baseline model was further reduced to  2.4% in the proposed model, showing reduced overfitting and better convergence. Evaluation of the test dataset shows a

Table 3. Hyperparameters selected for the training  of  the baseline and the proposed model.

| Experimental parameters   | Values                  |
|---------------------------|-------------------------|
| learning rate             | 0.0001                  |
| Momentum                  | 0.99                    |
| batchsize                 | 32                      |
| Number of epochs          | 500 with early stopping |
| Dropout                   | 0.2-0.5                 |
| Optimizer                 | Adam                    |

performance improvement of 8.54% in terms of  classification  accuracy  compared  to  the baseline model.

Table 4. Classification  performance measures for the Baseline and the Proposed Model.

The performance has been also evaluated by analysing the graphs of classification accuracy and cross-entropy loss during  the  training  process.  In  the  baseline model,  through  the  analysis  of  graphs  of classification accuracy and cross-entropy loss, as shown in Fig. 9, it has been observed that there exists a considerable gap between

|          | Performance Measures   | Performance Measures   | Performance Measures   | Performance Measures   | Performance Measures   | Performance Measures   |                  |
|----------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------|
| Model    | Training               | Training               | Validation             | Validation             | Testing                | Testing                | Epochs (max 500) |
|          | Loss                   | Accuracy (%)           | Loss                   | Accuracy (%)           | Loss                   | Accuracy (%)           |                  |
| baseline | 0.0609                 | 97.53                  | 0.2321                 | 93.23                  | 0.4856                 | 88.54                  | 161              |
| Proposed | 0.0068                 | 99.80                  | 0.1728                 | 97.40                  | 0.1347                 | 97.08                  | 327              |

the training and validation, which confirms overfitting in the model. Regularisation (through Dropout) and Early Stopping has been included into the baseline model to reduce the impact of  overfitting.  The  corresponding  graph  for  the  proposed model showing improved performance and better convergence between training and validation is shown in Fig. 10.

The confusion matrix 23 , which provides the matrix of true labels vs. predicted labels, is shown in Figs. 11 and 12 for the baseline  and  the  proposed  models.  It  represents  the  number of  true  classifications  in  each  category  through  the  diagonal elements.  It  has  been  observed  that  significant  confusion occurs between the aircraft carrier and destroyer category of images, and the same has been predicted due to the similarity of  features  between  the  two  categories.  Six  aircraft  carrier images have been incorrectly predicted  to  destroyer-class  in

<!-- image -->

<!-- image -->

Cross Entropy Loss

Figure 9. Graphs of classification accuracy and cross-entropy loss for the baseline model.

<!-- image -->

Figure 10. Graphs of classification accuracy and cross-entropy loss for the proposed model.

Figure 11. Confusion matrix for the baseline model.

<!-- image -->

Figure 12. Confusion matrix for the proposed model.

<!-- image -->

the proposed model. Another significant observation is that out of  120  test  images  for  the  destroyer-class,  118  images  have been correctly classified using the baseline model, while the number is 117 for the proposed model. However, the overall classification accuracy has improved  significantly in the proposed model.

## 5.    CONCLUSION

In the present study, ship classification has been addressed using  VGG16  based  transfer  learning  architecture.  Further, the addition of several performance improvement techniques and fine-tuning of neural network Hyperparameters have been carried  out  to  improve  the  baseline  model.  Evaluation  and analysis of the proposed model have been carried out for four categories of ship images using a limited dataset. CNN based proposed model shows promising results with a classification accuracy of 97.08%, making it suitable for maritime security applications.

In all the experiments, it has been assumed that the input images belong only to one of the four categories. However, if the input image does not belong to any of the four categories, it  can  be  classified  as  a  member  of  the  'unknown'  class  by assigning  a  suitable  threshold  value  (say,  0.5)  to  the  class scores.  Class  scores  is  the  values  of  associated  probabilities at the output of Softmax layer that is the final output layer in artificial neural network. If the value of the highest class score is  lower than the threshold value, then that particular output can be marked as 'unknown' class.

As a future work, this model can be further fine-tuned for use with satellite-based or SAR based ship images to create a robust system for ship classification. The study can be further extended to the case of multiple ships or objects in each input image.

## REFERENCES

1. http://www.imo.org/en/OurWork/safety/navigation/ pages/ais.asp x [Accessed on 28 Jun 2020].
2. bentes, C.; Frost, A.; Velotto, D. &amp; Tings, b. Ship-iceberg discrimination with convolutional neural networks in high resolution SAR images. In Proceedings of EUSAR 2016:
3. 11th European Conference on Synthetic Aperture Radar, Hamburg, Germany, 2016, 1-4.
3. Rainey, katie; Reeder, John D. &amp; Corelli, Alexander G. Convolution  neural  networks  for  ship  type  recognition. In Proc.  SPIE  9844,  12  May  2016,  Automatic  Target Recognition XXVI, 984409.
5. doi:10.1117/12.2229366
4. Shi, Qiaoqiao; li, Wei; Tao, Ran; Sun, Xu &amp; Gao, lianru. Ship classification based on multifeature ensemble with convolutional  neural  network. Remote  Sens. ,  2019, 11, 419.
7. doi: 10.3390/rs11040419.
5. lecun, Y.;  Haffner,  P.;  bottou,  l.  &amp;  bengio, Y .  Object recognition with gradient-based learning. In Shape, Contour  and  Grouping  in  Computer  Vision,  SpringerVerlag, Heidelberg, berlin, 1999, 319-345.
9. doi: 10.1007/3-540-46805-6\_19
6. Gonalez, R.C. Deep convolutional neural networks [lecture Notes]. IEEE Signal Processing Magazine, Nov 2018, 35 (2), 79-87.
11. doi: 10.1109/MSP.2018.2842646
7. https://cs231n.github.io/convolutional-networks/ [Accessed on 14 Nov 2020].
8. https://cs231n.github.io/transfer-learning/  [Accessed  on 14 Nov 2020].
9. Simonyan,  karen  &amp;  Zisserman,  Andrew.  Very  deep convolutional networks for large-scale image recognition. 2015.
15. arXiv 1409.1556v6
10.   He, k.; Zhang, X; Ren S. &amp; Sun J. Deep residual learning for image recognition. In Proc. IEEE Conf. Comput. Vis. Pattern  Recognit.  (CVPR),  las  Vegas,  NV,  USA,  Jun 2016, 770-778.
17. doi: 10.1109/CVPR.2016.90
11.   Szegedy, C.; Vanhoucke, V.; Ioffe, S.; Shlens, J. &amp; Wojna, Z.  Rethinking  the  inception  architecture  for  computer vision. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), las Vegas, NV, USA, Jun 2016, 2818-2826. doi: 10.1109/CVPR.2016.308
12.   Chollet,  F.  Xception:  Deep  learning  with  depthwise separable  convolutions. In Proc.  IEEE  Conf.  Comput. Vis. Pattern Recognit. (CVPR), Honolulu, HI, USA, Jul 2017, 1800-1807.
20. doi:10.1109/CVPR.2017.195
13.   Szegedy,  C.;  Ioffe,  S.;  Vanhoucke,  V.  &amp;  Alemi,  A. Inception- v4, inception-resnet and the impact of residual connections  on  learning. In Proc.  AAAI  Conf.  Artif. Intell.,  San  Francisco,  CA,  USA:  AAAI  Press,  2017, 4278-4284.
22. doi: 10.5555/3298023. 3298188
14.   Huang, G.; liu, Z.; Van Der Maaten, l. &amp; Weinberger, k.Q. Densely connected convolutional networks. In Proc. IEEE  Conf.  Comput.  Vis.  Pattern  Recognit.  (CVPR), Honolulu, HI, USA, Jul 2017, 2261-2269. doi: 10.1109/CVPR.2017.243
15. leclerc,  M.;  Tharmarasa,  R.;  Florea,  M.  C.;  bourybrisset, A.; kirubarajan, T. &amp; Duclos-Hindié, N. Ship  classification  using  deep  learning  techniques  for

maritime  target  tracking.  21st  International  Conference on Information Fusion (FUSION), Cambridge, 2018, pp. 737-744.

doi: 10.23919/ICIF.2018.8455679

16. Solmaz,  berkan;  Gundogdu,  Erhan;  Yucesoy,  Veysel &amp;  koc,  Aykut. Generic and attribute-specific deep representations  for  maritime  vessels.  IPSJ  Transactions on Computer Vision and Applications, 2017. doi: 10.1186/s41074-017-0033-4
17. Ship  photos  and  ship  tracker.  http://www.shipspotting. com[Accessed on 22 May 2020].
18.   bartan, burak. Ship classification using an image dataset. 2017. Corpus ID: 29004678
19. Milicevic, Mario; Zubrinic, krunoslav; Obradovic, Ines &amp; Sjekavica, Tomo. Data augmentation and transfer learning for limited dataset ship classification. WSEAS Trans. Syst. Control , 2018, 13 , 460-465.
20. Dao, Cuong; Xiaohui, Hua; Morère, Olivier. Maritime vessel images classification using deep convolutional neural networks. In proceedings of the Sixth International Symposium on Information and Communication Technology, 2015, 276-281. doi:10.1145/2833258.2833266
21. Ioffe, Sergey &amp; Szegedy, Christian. batch normalisation: accelerating  deep  network  training  by  reducing  internal covariate shift. 2015. arXiv:1502.03167
22.  Srivastava,  Nitish;  Hinton,  Geoffrey;  krizhevsky, Alex; Sutskever,  Ilya  &amp;  Salakhutdinov,  Ruslan.  Dropout:  A simple way to prevent neural networks from overfitting. J. Mach. Learning Res., 2014, 15 (56), 1929-1958.
23.  https://en.wikipedia.org/wiki/Confusion\_matri x [Accessed on 28 Jun 2020].

## CONTRIBUTORS

Mr Narendra Kumar Mishra received his MTech in Communication Engineering from IIT Delhi, in 2018. He is presently working as  Scientist  'D'  at  DRDO  - Weapons  and  Electronics  Systems Engineering  Establishment,  New  Delhi.  His  field  of  research includes  Embedded  Systems,  Systems  Integration,  Signal Processing  and  Machine  learning.  He  has  contributed  in  the design  and  development  of  interface  solutions  for  various ships  and  submarines.

In the current study, he conceived and designed the experiment, optimised the deep learning techniques used in the experiment, performed  software-coding,  results  analysis  and  prepared  the manuscript.

Mr  Ashok Kumar received  his  MTech  in  Radio  Frequency Design and Technology from IIT Delhi, in 2017. He is presently working  as  Scientist  'D'  at  DRDO-Weapons  and  Electronics Systems  Engineering  Establishment,  New  Delhi.  His  field  of research  includes  Embedded  Systems,  Data  Communication and  Systems  Integration.  He  has  contributed  in  the  design and  development  of  interface  solutions  for  various  ships  and submarines.

In  the  current  study,  he  performed  data  preparation,  helped in  formulation  of  the  experiment  and  preparation  of  the manuscript.

Mr  Kishor  Choudhury received  his  MTech  in  Computer Technology from IIT Delhi, in 2012. He is presently  working as  Scientist  'F'  at  DRDO  -  Weapons  and  Electronics  Systems Engineering  Establishment,  New  Delhi.  He  has  received  CNS Commendation  in  2006  and  DRDO  Technology  Group Award in  2008.  His  field  of  research  includes  algorithms,  embedded systems and computer vision. He has contributed in the design and  development  of  interface  units  for  various  ships  and submarines.

In  the  current  study,  he  provided  overall  guidance  in conceptualisation &amp; realisation of the experiment and finalisation of  the  manuscript.