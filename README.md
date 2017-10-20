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

Goal: We need to generate TFRecords: the inputs to our CNN. To do this, we must use our exported images and manually label where the Kentucky CAFO site resides in each image. The outputs will be provided to TensorFlow as a PASCAL_VOC format that can be easily interpreted by an existing CNN or for development of a new CNN. 

Description: Using [LabelImg](https://github.com/tzutalin/labelImg), we can create bounding boxes on images and generate xmls with data that can be used to generate TFRecords for our CNN. 

Once installed (you can get it with git clone https://github.com/tzutalin/labelImg), labelImg is pointed at the directory of our exported images from step 1. A lengthy process ensues: we must iterate on each image and using our own Neural Network (our eyes and noggins!) we must annotate where the CAFO sites are in each image.

Example Extracted CAFO Image Labeled for Test/Train Split: 

![Labeled for Test/Train Split](https://github.com/Qberto/ML_ObjectDetection_CAFO/blob/master/doc/img/img_106_2000_labeled.jpg)

This is a simple but repetitive process. It takes some time but the more images you have and the more detailed the labeling, the better the model can be trained for detection of these sites in any location. Sip on some coffee, put on a [nice Machine Learning podcast](https://twimlai.com/), and start labeling away!

Alternatively, if you would like to use our labeled images, you can use the images and xmls found in the images/test and images/train folders.

## Step 3: Split labeled images into train/test samples

Goal: The Convolutional Neural Network will use a subset of our labeled images to train a model, and a separate subset to check how well it's doing at detecting CAFO sites. We need to split our labeled images into these subsets before we start training the model.

Description: Once all your labeled images are ready, you should see xml files corresponding to each labeled image. We need to split our entire folder of images and corresponding xmls into training and testing subsets. For this model, I used a roughly 70% Train / 30% Test split (503 images for training, and 208 images for testing). 

For now, this is done manually. In an upcoming version of this repo, the test/split will be configured and done automatically. 

## Step 4: Generate TF Records from the train/test splits

Goal: We need to create TFRecord files that we need to train an object detection model in TensorFlow. 

Description: A few format changes must occur: First, we convert the XML files from all the images in the train and test folders into singular CSV files. Second, we convert the singular CSV files into TFRecord files. We'll use a few scripts to perform the conversions:

1. [XML to CSV](https://github.com/Qberto/ML_ObjectDetection_CAFO/2_xml_to_csv.py)

2. [Generate TFRecord](https://github.com/Qberto/ML_ObjectDetection_CAFO/3_generate_tfrecord.py)

Each script is currently provided as is, so please review each file and make the appropriate changes to reference your own directories and workspace. A future version of this repo will contain a more appropriate process to handle relative paths and more seamless transition of labeled images to TFRecords.

The first script should generate two CSV files: "test_labels.csv" and "train_labels.csv". Please take a second to confirm that each of these files contain data; the data should correspond to geometry for bounding boxes from step 3.

The 3_generate_tfrecord.py script then reads these and generates the TFRecord files. Please note, at this stage you should have TensorFlow installed on your system and the following repository available (https://github.com/tensorflow/models/tree/master/research). To execute this script for the test and train subsets, create a data folder in your workspace and run the following commands from your prompt:

python 3_generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
python 3_generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record

You should now see the train.record and test.record in your data folder. 

For reference, you may take a look at the data folder in this repository to see what your singular CSVs and TFRecord files should look like. 

## Step 5: Set up a configuration file containing CNN hyperparameters

## Step 6: Train

## Step 7: Export a computation graph from the new trained model

## Step 8: Detect CAFO sites in real time!

## Step 9: ...

## Step 10: Party.



### Appendix A: CNN Model Pseudocode

The CNN Model pseudocode can be categorized by two general steps: Feedforward and backpropagation. In other terms, the movement of the data through the neural network, and the learning or training of the model given results in an iteration. A single feedforward and backpropagation loop is known as an "epoch".

#### Feedforward: 
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer

#### Backpropagation:
compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer, Stochastic Gradient Descent, AdaGrad, etc.)

#### Epoch:
feedforward + backpropagation
