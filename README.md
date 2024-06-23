# Computer-Vision-for-Smoker-Detection-Enhancing-Public-Health-Surveillance-with-Advanced-Algorithms

This project endeavours to make significant contributions to the field of smoker detection and public health surveillance. The outcomes of this research have the potential to inform policy decisions, guide smoking cessation interventions, and contribute to the creation of healthier environments worldwide. 

The dataset is sourced from Kaggle which contains the images of individuals smoking and non-smoking. The dataset contains 1120 images in total, that is divided equally in two classes, in which 560 images belongs to smokers and remaining 560 images belongs to non-smokers. All the images in the dataset are resized to a resolution of 250*250. 
### Dataset Link: https://www.kaggle.com/datasets/sujaykapadnis/smoking

## Data Augmentation
It is a technique used to increase the diversity of training data without collecting new data. This is particularly useful in image classification tasks to improve the model’s robustness and generalizability by enlarging the training dataset. Augmentation techniques involve applying various transformations to the original images, such as rotations, transitions, flips, and changes in brightness and contrast. These transformations help the model become invariant to these changes, making it more reliable when encountering variations in real-world data. 


## CNN Architectures & Vision Transformer

Multiple CNN architectures (Transfer Learning) has been applied on the dataset and test the effectiveness of each architecture. Following architectures have been used for smoker detection:

1. VGG16: This is a popular Convolution Neural Network (CNN) architecture that has been widely used for image classification tasks. Developed by Visual Geometry Group (VGG) at the University of Oxford, VGG16 is known for its simplicity and depth, consisting of 16 layers with learnable weights.
   
2. ResNet-50: This is a widely used Convolutional Neural Network (CNN) architecture, is a part of the Residual Networks family, which introduced the concept of residual learning. ResNet-50 is a specifically designed to address the vanishing gradient problem that hampers deep neural networks, allowing them to train effectively even with very deep architectures.
   
3. Efficient NetV2: This is a state-of-the-art Convolutional Neural Network (CNN) architecture designed for efficient scaling and improved performance in image classification tasks. Developed by Google Research, Efficient NetV2 incorporates advancements in both the architecture design and training techniques, making it powerful tool for the task like smoker detection. •	Efficient NetV2 uses compound scaling to balance network depth, width and resolution. This approach ensures that the model scales efficiently without disproportionately increasing computational costs.
   
4. Vision Transformer (ViT): Vision Transformer (ViT) is a novel deep learning architecture that leverages the principles of transformers, which have revolutionized natural language processing, for image classification tasks. Unlike traditional Convolutional Neural Networks (CNNs), ViTs treat images as sequence of patches and apply transformer mechanisms to capture spatial dependencies and global context more effectively.


## Comparative Analysis & Key Finding

<img width="654" alt="tab1" src="https://github.com/shivamsingh96/Computer-Vision-for-Smoker-Detection-Enhancing-Public-Health-Surveillance-with-Advanced-Algorithms/assets/123630632/7aebd144-f9a0-47bb-8647-b50cefd9d36c">


The above result indicates that Efficient NetV2 is the most effective architecture for the task of smoker detection. Its superior performance can be attributed to its advanced design, which balance model depth, width, and resolution more effectively than ResNet-50 and ViT.



### Key Findings:

1. Efficient NetV2 achieved the highest accuracy, indicating its robustness in classifying both smokers and non-smokers accurately.
2. The balance between precision and recall in Efficient NetV2 suggests it minimizes the both false positives and false negatives more effectively.
3. A higher F1-score in Efficient NetV2 confirms it overall better performance in handling classification task.
4. The Vision Transformer showed promising results, demonstrating the potential of transformer based models in image classification tasks traditionally dominated by CNN. However, it slightly lagged behind Efficient NetV2 in most metrics. 



In summary, this project has demonstrated the effectiveness of advance CNN architectures for smoker detection, with Efficient NetV2 standing out as the most capable model. By addressing the identified areas for further research and development, the field can continue to evolve, leading to more accurate, efficient, and ethical applications of deep learning in public health and beyond.  



