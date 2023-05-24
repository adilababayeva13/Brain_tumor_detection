
# <a name="_63oncnnpdeqp"></a>Brain Tumour Detection Report 
+ Dataset : <a href="https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection" >Link!</a>
+ Code : <a href="https://www.kaggle.com/code/adilababayeva13/brain-tumor-detection-project/edit" >Link!</a>
+ Report : <a href="https://docs.google.com/document/d/1DPurX1XjjltDE7EmauZFmjTFEzOOny3927Mud2RsEgE/edit#">Link!</a>

- # <a name="_uvw2xu5t0m8f"></a>   Abstract 
Detecting brain tumours in their early stages is crucial. Brain tumours are classified by biopsy, which can only be performed through definitive brain surgery. 

A dataset containing 3000 Magnetic Resonance Imaging (MRI) brain images comprising images of glioma, meningioma, pituitary gland tumours, and healthy brains were used in this study. First, preprocessing and augmentation algorithms were applied to MRI brain images. 

In this report, we present the results of our brain tumour detection project using various machine learning models. The dataset consists of 1500 tumour images and 1500 non-tumor images, making it a balanced dataset:



We have evaluated the performance of several models, including Logistic Regression, SVC, k-Nearest Neighbors (kNN), Naive Bayes, Neural Networks, Random Forest, and K-means clustering. We analyse each model's accuracy, precision, recall, F1-score, true positive and negative, false positive and negative, and area under the curve (AUC) to compare their performance.

Logistic Regression(96%), Neural Networks(95%), and Random Forest (96%)showcase the highest performance in terms of accuracy, precision, recall, and F1-score. These models effectively distinguish between tumour and no tumour images, providing reliable results.

- # <a name="_7128m0eldrpc"></a>Introduction
Tumours are abnormal growths that can be either malignant or benign. There are over 200 different types of tumours that can affect humans. Brain tumours, specifically, are a serious condition where irregular growth in brain tissue impairs brain function. The number of deaths caused by brain tumours has increased by 300% in the last three decades. If left untreated, brain tumours can be fatal. Diagnosing and treating brain tumours is challenging due to their complexity. Early detection and treatment are crucial for improving survival rates. Brain tumour biopsy requires surgery, so there is a need for non-invasive diagnostic methods. Magnetic Resonance Imaging (MRI) is commonly used for diagnosing brain tumours. They can cause a range of symptoms, such as:

- ` `Headaches,
- ` `Seizures,
- ` `Vision problems,
- ` `Muscle twitching and shaking in the body,
- ` `Drowsiness, nausea, and vomiting.

Recent advancements in machine learning, particularly deep learning, have enabled the identification and classification of medical imaging patterns. Machine learning techniques have shown success in various medical applications, including disease prognosis, diagnosis, image classification, and tissue segmentation.

Due to the variation in pathology and potential limitations of human specialists, computer-assisted interventions and computational intelligence techniques can assist in tumour identification and classification. Machine learning, especially deep learning, plays a vital role in analysing, segmenting, and classifying cancer images, particularly brain tumours. These methods enable accurate and reliable tumour identification, distinguishing them from other diseases. In this study, we propose models that consider previous suggestions and limitations. We compare seven modelling methods to determine any significant differences in performance.
- # <a name="_27tbco7conlq"></a>Literature Review
`         `Over the years, researchers have explored various techniques and approaches to improve the accuracy and efficiency of brain tumour detection. In this literature review, we will summarise some key studies and their contributions to the field of brain tumour detection.

One of the fundamental challenges in brain tumour detection is the presence of inhomogeneous intensities and unclear boundaries within tumour images. Researchers have addressed this challenge by applying intensity normalisation or bias field correction techniques to balance the effect of magnetic field inhomogeneity. Additionally, features such as intensities, neighbourhood information, and texture have been widely utilised in different studies.

Several segmentation techniques have been proposed for brain tumour detection. Here, we highlight a few notable ones:

1. Spatial Clustering: It is important to differentiate between image segmentation and image clustering. In image segmentation, grouping is performed in the spatial domain, while image clustering is conducted in the measurement space.
1. Split and Merge Segmentation: This technique involves initially considering the entire image and splitting it into quarters. This process is repeated until a homogeneity criterion is satisfied. In the merge method, adjacent segments of the same object are merged or joined.
1. Region Growing: In the region growing method, neighbouring points are connected to each other to expand the region. The success of this method is often dependent on the selection of an appropriate threshold value.

It is worth noting that these are just a few examples of segmentation techniques used in brain tumour detection. Many other methods and algorithms have been developed and evaluated in the literature.

Additionally, researchers have explored the application of machine learning techniques in brain tumour detection. Various machine learning algorithms, such as logistic regression, support vector machines (SVM), k-nearest neighbours (KNN), decision trees, random forests, and neural networks, have been utilised for classification tasks. These algorithms leverage features extracted from brain images to distinguish between tumour and non-tumor regions.

Benchmark datasets, such as the MICCAI BraTS dataset, have played a crucial role in evaluating and comparing different algorithms. These datasets provide standardised data for researchers to optimise and compare their algorithms, leading to advancements in brain tumour detection.

- # <a name="_3mmy8bf03srz"></a>Methodology
***Data preparation :*** 

First, preprocessing and augmentation algorithms were applied to MRI brain images. 
### <a name="_hvwalikfvp4t"></a>**Image Augmentation includes**
1. **Geometric transformations**: randomly flip, crop, rotate, stretch, and zoom images. You need to be careful about applying multiple transformations on the same images, as this can reduce model performance. 
1. **Colour space transformations**: randomly change RGB colour channels, contrast, and brightness.

***Models:***

1. **Logistic Regression:**
   1. Logistic Regression is a linear model that predicts the probability of an input belonging to a particular class. It fits a logistic function to the input features and performs classification based on the calculated probabilities.
1. **SVC:**
   1. Support Vector Classifier (SVC) is a popular classification algorithm that separates data points by finding the optimal hyperplane in a high-dimensional feature space. It aims to maximise the margin between different classes while considering support vectors.
1. **kNN:**
   1. k-Nearest Neighbors (kNN) is a non-parametric algorithm that classifies an input based on its nearest neighbours in the feature space. It assigns a class label to the input based on the majority vote of its k nearest neighbours.
1. **Naive Bayes:**
   1. Naive Bayes is a probabilistic classifier based on Bayes' theorem. It assumes that features are conditionally independent given the class label. Despite its simplistic assumptions, Naive Bayes has been successful in many classification tasks.
1. **Neural Networks:**
   1. Neural Networks, or Artificial Neural Networks, are composed of interconnected nodes (neurons) arranged in layers. They are capable of learning complex patterns and relationships in the data through a process called training, which involves adjusting the weights between neurons.
1. **Random Forest:**
   1. Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. Each decision tree is trained on a subset of the data, and the final prediction is determined by averaging or voting among the individual trees.
1. **K-means clustering:**
   1. K-means clustering is an unsupervised learning algorithm that partitions the data into k clusters based on their similarity. It iteratively assigns data points to clusters and updates the cluster centroids until convergence (Although we know that this model is not used for classification, we only used it to test how such an approach would work).
#
- # <a name="_wbqynm2tfovv"></a><a name="_h91et6ulx51r"></a>**Experiments**

- ## <a name="_47y493wzpbel"></a>    **Logistic Regression:**


|*** |**precision**|**recall**|**f1-score**|**support**|
| :-: | :-: | :-: | :-: | :-: |
|***0***|**0.95**|**0.97**|**0.96**|**306**|
|***1***|**0.96**|**0.94**|**0.95**|**288**|
|***accuracy***|** |** |**0.96**|**594**|
|***macro avg***|**0.96**|**0.96**|**0.96**|**594**|
|***weighted avg***|**0.96**|**0.96**|**0.96**|**594**|

Logistic Regression performs exceptionally well in the brain tumour detection task with an accuracy of 96%. It demonstrates a precision of 0.96, indicating a high percentage of correct predictions for brain tumour cases.

The recall score of 0.97 highlights the model's ability to effectively identify actual brain tumour images.With an F1-score of 0.96, Logistic Regression achieves a balanced combination of precision and recall.The AUC value of 0.96 indicates the model's strong discriminatory power in distinguishing between tumour and no tumour images.
##
- ## <a name="_mlycvy38jc8p"></a><a name="_tgk7e580djp1"></a>SVC :  


|* |**precision**|**recall**|**f1-score**|**support**|
| -: | :-: | :-: | :-: | :-: |
|***0***|**0.94**|**0.94**|**0.94**|**306**|
|***1***|**0.94**|**0.94**|**0.94**|**288**|
|*accuracy*|** |** |**0.94**|**594**|
|*macro avg*|**0.94**|**0.94**|**0.94**|**594**|
|*weighted avg*|**0.94**|**0.94**|**0.94**|**594**|

`   	`SVC (Support Vector Classifier) also demonstrates strong performance in the brain tumour detection task with an accuracy of 94%. It achieves a precision of 0.94, indicating a high percentage of correct predictions for brain tumour cases. 

The recall score of 0.94 highlights the model's ability to effectively identify actual brain tumour images. 

With an F1-score of 0.94, SVC achieves a balanced combination of precision and recall. 

The AUC value of 0.94 indicates the model's good discriminatory power in distinguishing between tumour and no tumour images.




- ## <a name="_51awcj7day14"></a> **kNN:**


|* |**precision**|**recall**|**f1-score**|**support**|
| -: | :-: | :-: | :-: | :-: |
|***0***|**0.85**|**0.97**|**0.91**|**306**|
|***1***|**0.97**|**0.81**|**0.88**|**288**|
|*accuracy*|** |** |**0.90**|**594**|
|*macro avg*|**0.91**|**0.89**|**0.89**|**594**|
|*weighted avg*|**0.90**|**0.90**|**0.89**|**594**|


kNN (k-Nearest Neighbors) demonstrates satisfactory performance in the brain tumour detection task with an accuracy of 90%. It achieves a precision of 0.85, indicating a reasonably high percentage of correct predictions for brain tumour cases. The recall score of 0.97 highlights the model's ability to effectively identify actual brain tumour images. With an F1-score of 0.91, kNN achieves a relatively balanced combination of precision and recall. **kNN exhibits a relatively high number of false negatives**, indicating that it incorrectly classifies a significant number of tumour images as no tumour images.

`  `The AUC value of 0.89 suggests that the kNN model has reasonable discriminatory power in distinguishing between tumour and no tumour images.
- ## <a name="_dsniwa1hjxxn"></a>Naive Bayes:

|* |**precision**|**recall**|**f1-score**|**support**|
| -: | :-: | :-: | :-: | :-: |
|***0***|**0.63**|**0.70**|**0.66**|**306**|
|***1***|**0.64**|**0.57**|**0.60**|**288**|
|*accuracy*|** |** |**0.64**|**594**|
|*macro avg*|**0.64**|**0.63**|**0.63**|**594**|
|*weighted avg*|**0.64**|**0.64**|**0.63**|**594**|
** 
**


Naive Bayes demonstrates moderate performance in the brain tumour detection task, with an accuracy of 64%. The model achieves a precision of 0.63, indicating that it correctly predicts brain tumour cases 63% of the time. The recall score of 0.70 suggests that the model can identify actual brain tumour cases 70% of the time.

However, **Naive Bayes falls short in achieving a balanced combination of precision and recall,** as reflected in the F1-score of 0.66. The model struggles to accurately capture important features and patterns in the input images, resulting in less accurate predictions.

**The model shows limitations in correctly identifying tumour and no tumour images**, with a significant number of false positives and false negatives. It misclassifies a substantial number of no tumour images as tumour images and also fails to correctly classify several tumour images as no tumour images.

The AUC value of 0.63 indicates that **the Naive Bayes model has limited discriminatory power in distinguishing between tumour and no tumour** images.
##
##
- ## <a name="_mjzbktlij0uj"></a><a name="_qrgkr9vd1bbw"></a><a name="_icfz1nemsp8c"></a> **Neural Networks:**

|* |**precision**|**recall**|**f1-score**|**support**|
| -: | :-: | :-: | :-: | :-: |
|***0***|**0.94**|**0.97**|**0.96**|**306**|
|***1***|**0.97**|**0.94**|**0.95**|**288**|
|*accuracy*|** |** |**0.95**|**594**|
|*macro avg*|**0.96**|**0.95**|**0.95**|**594**|
|*weighted avg*|**0.95**|**0.95**|**0.95**|**594**|
** 

Neural Networks exhibit strong performance in brain tumour detection, achieving an accuracy of 95%. The model demonstrates a high precision of 0.94, indicating that it correctly predicts brain tumour cases with a 94% accuracy. With a recall score of 0.97, the model effectively identifies actual brain tumour cases.

The F1-score of 0.96 reflects a well-balanced combination of precision and recall, indicating the model's strong overall performance. Neural Networks excel at capturing complex patterns and features in the input images, leading to accurate predictions.

The AUC value of 0.95 signifies the Neural Networks' excellent discriminatory power in distinguishing between tumour and no tumour images.
- ## <a name="_6bhfnrj8sng"></a>Random Forest:

|* |**precision**|**recall**|**f1-score**|**support**|
| -: | :-: | :-: | :-: | :-: |
|***0***|**0.98**|**0.94**|**0.96**|**306**|
|***1***|**0.97**|**0.98**|**0.96**|**288**|
|*accuracy*|** |** |**0.96**|**594**|
|*macro avg*|**0.96**|**0.96**|**0.96**|**594**|
|*weighted avg*|**0.96**|**0.96**|**0.96**|**594**|
** 

Random Forest demonstrates excellent performance in the brain tumour detection task. With an accuracy of 96%, the model achieves a high level of correctness in classifying brain tumour images. The precision score of 0.98 indicates that when the model predicts a brain tumour, it is correct 98% of the time. The recall score of 0.94 highlights the model's ability to effectively identify actual brain tumour cases.The F1-score of 0.96 reflects a balanced combination of precision and recall, indicating a strong overall performance of the Random Forest model compared to other models. These metrics suggest that Random Forest can effectively capture complex patterns and features in the input images, leading to accurate predictions. It also **misclassified 7 tumour images as no tumour images (false negatives, which is the least value among models).**The AUC (Area Under the Curve) value of 0.96 indicates that the Random Forest model has strong discriminatory power in distinguishing between tumour and no tumour images.


- ## <a name="_f8ubenclik07"></a>**K-means clustering:**

|* |**precision**|**recall**|**f1-score**|**support**|
| -: | :-: | :-: | :-: | :-: |
|***0***|**0.64**|**0.76**|**0.69**|**1482**|
|***1***|**0.70**|**0.57**|**0.63**|**1488**|
|*accuracy*|** |** |**0.66**|**2970**|
|*macro avg*|**0.67**|**0.66**|**0.66**|**2970**|
|*weighted avg*|**0.67**|**0.66**|**0.66**|**2970**|
** 

K-means clustering demonstrates **moderate** performance in the brain tumour detection task. With an accuracy of 66%, the model achieves a moderate level of correctness in clustering brain tumour images. The precision score of 0.64 indicates that the model correctly identifies 64% of the brain tumour cases among the clustered data. The recall score of 0.76 suggests that the model captures 76% of the actual brain tumour cases within the clusters.

The F1-score of 0.69 reflects a relatively balanced combination of precision and recall. These metrics indicate that while K-means clustering can group similar images together, **it may not accurately capture all the nuances and variations within the brain tumour dataset**.

It **misclassified 641 brain tumour images as no tumour images (false negatives) and this is so high.**

The AUC (Area Under the Curve) value of 0.66 suggests that the K-means clustering model has **limited discriminatory power in distinguishing between tumour and no tumour images.**
# <a name="_a1hjfqhryljt"></a>**Conclusion:**
After carefully evaluating the performance of multiple machine-learning models on the brain tumour detection dataset, we can draw the following conclusions:

1. Logistic Regression, Neural Networks, and Random Forest showcase the highest performance in terms of accuracy, precision, recall, and F1-score. These models effectively distinguish between tumour and no tumour images, providing reliable results.
1. SVC and kNN models also demonstrate respectable performance, maintaining a good balance between precision and recall. They achieve accuracy levels above 90%, making them viable alternatives for brain tumour detection.
1. Naive Bayes, while achieving lower accuracy and performance metrics compared to other models, may not be as effective in this particular task. It exhibits lower precision, recall, and F1-score, suggesting its limitations in accurately classifying brain tumour images.
1. K-means clustering, though not a traditional classification model, achieves lower accuracy and performance metrics compared to other models. This indicates that it may not be suitable for binary classification in brain tumour detection.

In conclusion, considering the overall performance metrics and their respective strengths, the Random Forest model appears to be the best-performing model. Random Forest achieved an accuracy of 96%, precision of 0.98, recall of 0.94, and an F1-score of 0.96. It correctly identified 288 tumour images with only 18 false positives and 7 false negatives. The model demonstrated strong discriminatory power with an AUC value of 0.96.

The worst-performing model for brain tumour detection appears to be the Naive Bayes model. Naive Bayes achieved an accuracy of 64%, precision of 0.63, recall of 0.70, and an F1-score of 0.66. It correctly identified 213 tumour images as true positives but produced a high number of false positives (93). The model also misclassified 123 tumour images as false negatives. It correctly identified 165 no tumour images as true negatives.
# <a name="_1vdv4sjjcfdo"></a>**References :** 
- <https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02114-6#availability-of-data-and-materials>
- <https://www.datacamp.com/tutorial/complete-guide-data-augmentation>
- <https://link.springer.com/article/10.1007/s10044-017-0597-8#Sec2>
- <https://www.academia.edu/36404652/Brain_Tumor_Detection_using_MRI_A_Review_of_Literature>
- <https://www.kaggle.com/code/adilababayeva13/brain-tumor-detection-project/edit>





