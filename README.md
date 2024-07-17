# An Experimental Evaluation of Siamese Neural Networks for Robot Localization Using Omnidirectional Imaging in Indoor Environments

**Authors:** Juan José Cabrera, Vicente Román, Arturo Gil, Oscar Reinoso, Luis Payá  
**Journal:** Artificial Intelligence Review (2024) - Vol. 57, num. 198  
**Publisher:** Springer  
**ISSN:** 1573-7462  
**DOI:** [10.1007/s10462-024-10840-0](https://link.springer.com/article/10.1007/s10462-024-10840-0)

## Introduction

The objective of this paper is to address the localization problem using omnidirectional images captured by a catadioptric vision system mounted on the robot. For this purpose, we explore the potential of Siamese Neural Networks for modeling indoor environments using panoramic images as the unique source of information. Siamese Neural Networks are characterized by their ability to generate a similarity function between two input data, in this case, between two panoramic images. In this study, Siamese Neural Networks composed of two Convolutional Neural Networks (CNNs) are used. The output of each CNN is a descriptor which is used to characterize each image. The dissimilarity of the images is computed by measuring the distance between these descriptors. This fact makes Siamese Neural Networks particularly suitable to perform image retrieval tasks. First, we evaluate an initial task strongly related to localization that consists in detecting whether two images have been captured in the same or in different rooms. Next, we assess Siamese Neural Networks in the context of a global localization problem. The results outperform previous techniques for solving the localization task using the COLD-Freiburg dataset, in a variety of lighting conditions, specially when using images captured in cloudy and night conditions.

## Citation
If you find this work useful, please consider citing:

@article{Cabrera2024SNNLocalization,
  title={An experimental evaluation of Siamese Neural Networks for robot localization using omnidirectional imaging in indoor environments},
  author={Juan José Cabrera and Vicente Román and Arturo Gil and Oscar Reinoso and Luis Payá},
  journal={Artificial Intelligence Review},
  volume={57},
  number={198},
  year={2024},
  publisher={Springer Nature},
  issn={1573-7462},
  doi={10.1007/s10462-024-10840-0}
}
