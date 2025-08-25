## 2.7 Evaluation metrics

In our study, we chose accuracy, precision, recall, F1 score, and the confusion matrix as the primary metrics for assessing our CNN models due to their crucial role in evaluating classification performance [14, 50]. These metrics provide a comprehensive understanding of model

<!-- image -->

a cat

<!-- image -->

<!-- image -->

(b) airplane

<!-- image -->

apple

<!-- image -->

bird

<!-- image -->

bicycle

<!-- image -->

(0) COVID-19

<!-- image -->

<!-- image -->

<!-- image -->

deer

<!-- image -->

boy

<!-- image -->

COVID-19

<!-- image -->

<!-- image -->

dog

<!-- image -->

chair

<!-- image -->

<!-- image -->

frog

<!-- image -->

rose

normal

<!-- image -->

(u) COVID-19

<!-- image -->

COVID-19

(w) normal

normal

Fig. 2 Sample images from CIFAR-10, CIFAR-100, X-ray, and CT scan datasets

Table 2 Details of medical images dataset used for the development of COVID-19 detection model using the proposed MaxMinPooling method

| Category    | X-ray dataset   | X-ray dataset   | X-ray dataset   | CT scan dataset   | CT scan dataset   | CT scan dataset   |
|-------------|-----------------|-----------------|-----------------|-------------------|-------------------|-------------------|
|             | Train           | Test            | Validation      | Train             | Test              | Validation        |
| Normal      | 6,849           | 2,140           | 1,712           | 4,412             | 1,378             | 1,103             |
| COVID-19    | 7,649           | 2,395           | 1,912           | 4,860             | 1,518             | 1,215             |
| Pneumonia   | 7,208           | 2,253           | 1,802           | 1,676             | 523               | 419               |
| Total       | 21,706          | 67,88           | 5,426           | 10,948            | 3,419             | 2,737             |
| Grand Total | 33,920          |                 |                 | 17,104            |                   |                   |

accuracy and effectiveness, particularly in handling false positives and negatives-vital for applications in medical diagnostics. Additionally, we employed performance learning curves to visually track the models' progress over training epochs, aiding in the identification of potential overfitting or underfitting issues. This approach allows us to fine-tune our models effectively, ensuring robust validation and optimization of the architectures for reliable real-world applications. This methodical evaluation not only demonstrates the models' capabilities, but also enhances their applicability across diverse scenarios, particularly in critical tasks such as disease detection. The computation of performance measures adhered to the following mathematical expressions:

(m) sea

<!-- image -->

pneumonia

<!-- image -->

pneumonia

<!-- image -->

(g horse

<!-- image -->

(n) girl

pneumonia

<!-- image -->

(z) pneumonia

<!-- image -->

<!-- image -->

Fig. 5 A block diagram of the custom CNN model for classifying images from (a) the CIFAR-10 dataset and (b) the CIFAR-100 dataset. Conv2D refers to convolutional layer 2D followed by activation function, and BN stands for batch normalization

<!-- image -->

<!-- image -->

$$\ A c c u r a c y = T P + T N T P + T N + F P + F N , \\ e c i s i o n = T P T P + F P , \\ T P$$
