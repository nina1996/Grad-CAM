# Grad-CAM
This example code is created by Nikolina Tomic on April 2021.

Gradient-weighted Class Activation Mapping (GradCAM) is a method proposed in 2017 by R Selvaraju et al [1]. The aim is to unlock the explainaiblity of Convolutional Neural Networks (CNN). In other words, to explain where the decision of CNN comes from.  GradCAM does so by propagating the gradient back from output of the CNN to the last convolution layer. The output of this method is thus heatmap that describes which parts of the input image contributed the most to the specific decision of CNN.

In this example, we use CNN trained in Tensorflow to predict 3 variables from 2 input images. This is sample code to showcase how GradCAM is used in real-world problem. Data and the codes used to generate inputs are not available due to privacy reasons.
