## 4 Results analysis and discussion

This section details the classification accuracies obtained by the three distinct models-each employing MaxPooling, MinPooling, and our novel MaxMinPooling layers-across four different datasets, namely CIFAR-10, CIFAR-100, X-ray, and CT scan images. Performance metrics for each model, including accuracy, precision, recall, and F1 scores, are systematically presented in Tables 4 and 5. The tables provide a clear and concise comparison of the performance of each pooling strategy across the datasets, supporting the ablation study. Additionally, these comparisons are visually represented in a bar chart (Fig. 7), enhancing the interpretability of the data and providing clear insights into the strengths and weaknesses of each pooling method under different conditions.

Notably, enhancements from the probabilistic average method over traditional MaxPooling show an increment of 1.36% for CIFAR-10, whereas a more substantial improvement of 5.65% is observed for CIFAR-100. These results underscore the effectiveness of combining MaxMinPooling with probabilistic averaging to boost the performance of MaxPooling-based neural networks. The method's efficacy was further tested on medical images such as X-rays and CT scans, specifically for distinguishing COVID-19 cases from healthy and pneumonia instances. While the proposed pooling method showed significant improvements in MaxMinPooling over MaxPooling, the gains were less pronounced when utilizing probabilistic fusion methods in medical image datasets. This could have been because probabilistic averaging may introduce noise or dilute critical features, which are essential for the precision required in medical image classification. However, it

Fig. 7 Grad-CAM analysis of the MaxMinPooling model on six CT scan images. Each row contains two pairs of images: the original CT scan (left) and the corresponding heatmap (right), highlighting the decision-making process of the model for effective pattern interpretation

Original Image COVID

<!-- image -->

Original Image Normal

Original Image Pneumonia

<!-- image -->

Heatmap

(a) COVID

<!-- image -->

<!-- image -->

Heatmap

(b) Normal

<!-- image -->

<!-- image -->

Heatmap

Original Image Pneumonia

Heatmap

<!-- image -->

(c Pneumonia

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 8 Comparison of model accuracies for CIFAR-10, CIFAR-100, X-ray and CT scan datasets

<!-- image -->

is important to emphasize that the probabilistic fusion approach consistently showed improved results in nonmedical datasets such as CIFAR-10 and CIFAR-100, where the fusion method performed better compared to the individual pooling techniques. The model's classification accuracy for each category is also detailed through confusion matrices, as illustrated in Fig. 9 and 10.

Original Image COVID

Original Image Normal

<!-- image -->

Epochs

Accuracy on CIFAR-10

<!-- image -->

Accuracy on CIFAR-100

<!-- image -->

Accuracy on X-ray

<!-- image -->

<!-- image -->

Epochs

(b) Loss on CIFAR-10

<!-- image -->

(d Loss on CIFAR-100

Loss on X-ray

<!-- image -->

(g) Accuracy on CT scan

<!-- image -->

(h) Loss on CT scan

Fig. 9 Performance learning curves of the proposed MaxMinPooling method for CIFAR-10, CIFAR-100, X-ray and CT scan datasets

An important observation during our study was the impact of integrating both MaxPooling and MinPooling into the MaxMinPooling strategy, which effectively doubles the size of the feature maps. This increase in feature map size elevates the computational complexity of the model. However, in many AI applications, such as COVID-19 image classification, the precision of detection outweighs the need for real-time processing. Such enhancements are crucial as they lead to improved generalization of the AI solutions in scenarios where accuracy is more critical than computational speed. Through experimental study, it has been observed that the change of order of MaxPooling and MinPooling, while concatenating the outputs in MaxMinPooling, yields almost similar classification performance. The detailed results are included in the ablation study of Tables 4 and 5 under ''MinMaxPooling,'' which further validate that the sequence of pooling operations has a negligible effect on the overall model accuracy.

Fig. 11 Confusion matrices for detection of COVID-19 using the proposed MaxMinPooling method

<!-- image -->

<!-- image -->

Fig. 10 Confusion matrix for CIFAR-10 image classification using the proposed MaxMinPooling method

<!-- image -->

Our extensive performance analysis in various datasets illustrates the benefits of integrating MaxPooling, MinPooling, and MaxMinPooling strategies. This integration enables the model to capture a broader and more nuanced spectrum of features from the input data. The study details how the model leverages high activation features typically accentuated by MaxPooling along with the subtler features often highlighted by MinPooling. This synergistic approach in the MaxMinPooling2D model enhances the overall feature representation and improves classification performance. These findings underscore the potential of the model in applications that require high levels of accuracy, providing valuable insight into its utility to improve automated AI solutions.

that strongly influence the predictions of the model. In Fig. 11, we present six CT scan images processed by the MaxMinPooling model, with two examples from each category. The heatmaps reveal the critical areas that contribute to the classification, where warmer colors correspond to higher activations. Upon closer examination, distinct heatmap patterns emerge for each category, providing valuable information for medical practitioners to further analyze the reasoning behind the classification. This method enhances the interpretability of AI models, offering a transparent visualization of the model's reasoning, which is crucial for building trust in AI-driven medical diagnostics.

We employed Grad-CAM (gradient-weighted class activation mapping) to gain insights into the decisionmaking process of our CNN model when classifying CT scan images. Grad-CAM highlights the regions of an image

In our study, we have drawn comparisons with several notable advancements in CNN pooling techniques from the literature, particularly focusing on their impact on classification accuracy. For instance, Ozdemir achieved

significant improvements, enhancing CIFAR-10 and CIFAR-100 accuracies by 6.28% and 7.76%, respectively, compared to traditional MaxPooling [21]. Similarly, Yu et al. reported an improvement in the CIFAR-10 test accuracy to 89.20% using a mixed pooling method, outperforming the 88.64% accuracy from MaxPooling [22]. These studies underscore the potential of alternative pooling methods in enhancing classification performance.

Our research integrates MaxPooling, MinPooling, and our novel MaxMinPooling strategies within a unified model framework. This integrated approach not only leverages the strengths of each pooling method, but also addresses the gaps noted in the single-strategy pooling models. For example, Williams et al. and Zhao et al. also reported marginal improvements using new pooling methods over MaxPooling, with Williams et al. reaching an accuracy of 81.15% on CIFAR-10 [25], and Zhao et al. enhancing CIFAR-100 and CIFAR-10 accuracies by 4.11% and 4.32% [18]. In contrast, our integrated model consistently demonstrates superior performance, showcasing more robust and versatile capabilities across different datasets, including medical imaging datasets critical for COVID-19 detection.

These comparative analyses highlight the efficacy of our novel integrated pooling approach, indicating an improvement over traditional methods and setting a new benchmark in the utilization of CNNs for complex image classification tasks. This juxtaposition with previous studies establishes the significant strides our methodology offers beyond the current state-of-the-art, particularly in applications demanding high accuracy and reliability.
