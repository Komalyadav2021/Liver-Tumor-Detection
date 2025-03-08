# Liver-Tumor-Detection
 Segmenting the liver and tumors from computed tomography (CT) scans is crucial for medical studies utilizing 
machine and deep learning techniques. Semantic segmentation, a critical step in this process, is accomplished 
effectively using fully convolutional neural networks (CNNs). Most Popular networks like UNet and ResUNet 
leverage diverse resolution features through meticulous planning of convolutional layers and skip connections. 
This study introduces an automated system employing different convolutional layers that automatically extract 
features and preserve the spatial information of each feature. In this study, we employed both UNet and a 
modified Residual UNet on the 3Dircadb (3D Image Reconstruction for computer Assisted Diagnosis database) 
dataset to segment the liver and tumor. The ResUNet model achieved remarkable results with a Dice Similarity 
Coefficient of 91.44% for liver segmentation and 75.84% for tumor segmentation on 128 × 128 pixel images. 
These findings validate the effectiveness of the developed models. Notably both models exhibited excellent 
performance in tumor segmentation. The primary goal of this paper is to utilize deep learning algorithms for liver 
and tumor segmentation, assessing the model using metrics such as the Dice Similarity Coefficient, accuracy, and 
precision

Download Dataset from here: https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/
