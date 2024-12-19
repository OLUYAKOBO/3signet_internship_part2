# **3SIGNET LIMITED CIFAR-100 Image Classification Project**

## **Project Overview**
This project focuses on building a deep learning model to classify images from the CIFAR-100 dataset into 100 fine-grained categories. The model has been deployed as a web application using Streamlit, where users can upload images for real-time classification.

---

## **Dataset**
- **Source:** The CIFAR-100 dataset, developed by the Canadian Institute for Advanced Research.
- **Description:** The dataset consists of 60,000 32x32 color images in 100 classes. Each class contains 600 images.
  - **Training Set:** 50,000 images.
  - **Test Set:** 10,000 images.
- **Classes:** Images are categorized into 100 fine-grained classes (e.g., apple, lion, airplane) grouped into 20 coarse categories (e.g., fruits, mammals, vehicles).
- **Preprocessing Steps:**
  - Data augmentation techniques such as random cropping, horizontal flipping, and rotation were applied to improve generalization.

---

## **Model Architecture**
- **Base Model:** 
  - A convolutional neural network (CNN) was designed to classify the CIFAR-100 dataset.
- **Key Layers:**
  - Multiple convolutional layers for feature extraction.
  - Max-pooling layers for down-sampling.
  - Dense layers for classification.
  - Dropout layers for regularization to reduce overfitting.
- **Advanced Techniques:**
  - Used transfer learning with a pre-trained ResNet50 model to improve performance.
  - Fine-tuned the model on the CIFAR-100 dataset.

---

## **Training Process**
- **Optimizer:** Adam optimizer with a learning rate scheduler for dynamic learning rate adjustment.
- **Loss Function:** Categorical crossentropy, suitable for multi-class classification problems.
- **Evaluation Metrics:** 
  - Accuracy
  - Precision, Recall, and F1-score for individual class evaluation.

---

## **Results**
- **Test Accuracy:** Achieved a test accuracy of **38%**. Which means that the model predicted 3or 4 out of 10 images correctly.
- **Performance Observations:**
  - The model performed well on distinct classes but struggled with visually similar categories.
- **Confusion Matrix:** Highlighted the misclassifications across challenging classes.

---

## **Deployment**
- **Platform:** The model has been deployed as a web application using **Streamlit**.
- **Functionality:** 
  - Users can upload their own images to the application.
  - The app processes the image and outputs the predicted class label with the corresponding confidence score.

---

Try out the app here: https://cifarimageclassificationapp.streamlit.app/
