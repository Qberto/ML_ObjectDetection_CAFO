# Using a TensorFlow CNN to Detect Concentrated Animal Feeding Operations (CAFO) in Satellite Imagery
Repository containing model development, training, and execution for the CAFO satellite imagery detection process.

Summary: The Environmental Protection Agency is interested in building a dataset of Concentrated Animal Feeding Operation (CAFO) locations. This process is an exploration of how we can use a Convolutional Neural Network (CNN) to scan satellite imagery from the [National Agricultural Imagery Program (NAIP)](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) and detect CAFO facilities, writing the detected objects as features in a GIS dataset. 

The process is encapsulated in the following steps:

Step 1: Data Preparation - Image "Chip" Extraction

Step 2: Annotation and Labeling of the Images

Step 3: CNN Model Training

Step 4: CNN Model visual verification

Step 5: Output feature transfer to GIS

Please consider this project and all aspects of this repository as beta, with extensive changes and further testing still needed to finalize.

## Step 1: Data Preparation - Image "Chip" Extraction

Goal: The goal of this step is to extract images containing satellite imagery over a known CAFO location. These images will be manually labeled for the PASCAL_VOC format that a TensorFlow model can interpret for CNN development. 

Description: Using an ArcGIS Pro project, we will leverage two data sources to build our model inputs: a [NAIP imagery service](https://naip.arcgis.com/arcgis/services/NAIP/ImageServer) and a feature class containing identified CAFO sites in Kentucky (gis_inputs/ky_afo_lonx_nonzero) used in a image classification workshop. 

The 1_image_exporter.py script iterates on each record of the Kentucky feature class containing CAFO sites, loading the NAIP imagery at the location at three different specified scales: 1:1000; 1:2000; and 1:3000, and exporting each as a .JPEG image in a designated directory. A total of 250 locations are used, resulting in 750 input features (3 for each location) for training and testing.

## Step 2: Annotation and Labeling of the Images

Goal: We need to generate TFRecords: the inputs to our CNN. To do this, we must use our exported images and manually label where the Kentucky CAFO site resides in each image. This is used by the CNN to 

## Step 3: CNN Model Training

## Step 4: CNN Model visual verification

## Step 5: Output feature transfer to GIS



### Appendix A: CNN Model Pseudocode

The CNN Model pseudocode can be categorized by two general steps: Feedforward and backpropagation. In other terms, the movement of the data through the neural network, and the learning or training of the model given results in an iteration. A single feedforward and backpropagation loop is known as an "epoch".

#### Feedforward: 
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer

#### Backpropagation:
compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer, Stochastic Gradient Descent, AdaGrad, etc.)

#### Epoch:
feedforward + backpropagation
