# Using a TensorFlow CNN to Detect Concentrated Animal Feeding Operations (CAFO) in Satellite Imagery
Repository containing model development, training, and execution for the CAFO satellite imagery detection process.

Summary: The Environmental Protection Agency is interested in building a dataset of Concentrated Animal Feeding Operation (CAFO) locations. This process is an exploration of how we can use a Convolutional Neural Network (CNN) to scan satellite imagery from the [National Agricultural Imagery Program (NAIP)](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) and detect CAFO facilities, writing the detected objects as features in a GIS dataset. 

The process is encapsulated in the following steps:

Step 1: Data Preparation - Image "Chip" Extraction

Step 2: Annotation and Labeling of the Images

Step 3: Split labeled images into train/test samples

Step 4: Generate TF Records from the train/test splits

Step 4: CNN Model visual verification

Step 5: Set up a configuration file containing CNN hyperparameters and a label file containing your object classes

Step 6: Train

Step 7: Export an inference graph from the new trained model

Step 8: Detect CAFO sites!

Step 9: Convert Information to GIS



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

The 3_generate_tfrecord.py script then reads these and generates the TFRecord files. Please note, at this stage you should have TensorFlow installed on your system and the following repository available in your workspace (https://github.com/tensorflow/models/tree/master/research). To execute this script for the test and train subsets, create a data folder in your workspace and run the following commands from your prompt:

'''
python 3_generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
python 3_generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
'''

You should now see the train.record and test.record in your data folder. 

For reference, you may take a look at the data folder in this repository to see what your singular CSVs and TFRecord files should look like. 

## Step 5: Set up a configuration file containing CNN hyperparameters and a label file containing your object classes

Goal: To start training, we need the images, matching TFRecords for training and testing data, and we need to set up a configuration file that will hold most of the hyperparameters for the CNN. We will also use a label file that tells the model which classes we expect to detect. 

Description: The configuration file's variables contain references to hyperparameters for the CNN: These determine most of the architecture of the neural network, including the number of output classes, matching thresholds, number of layers in the network, convolutional box parameters, which activation function will be used, and several other parameters. For help with setting up your own configuration file, refer to the [TensorFlow documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md). Alternatively, you may use the configuration file we used and is [available in this repo](https://github.com/Qberto/ML_ObjectDetection_CAFO/training/ssd_mobilenet_v1_cafo.config). If you use the config file attached, be sure to alter the PATH_TO_BE_CONFIGURED variables to reference your workspace. You may also want to change your batch size, depending on your GPU's VRAM. The default of 24 should work for fairly modern systems with powerful graphics cards, but if you experience a memory error, you may want to test with a lower batch size.

We also have the option of using an existing model and using transfer learning to teach the model how to detect a new object. In this case, we'll use the [SSD with MobileNet V1](https://github.com/tensorflow/models/tree/master/research/object_detection/models) as a checkpoint and teach it to detect a new object: CAFO sites in satellite imagery. 

The label file is a much simpler process... in fact here's the entire contents of the one we used:

item {
	id: 1
	name: 'cafo'
}

This one can be found at training/object-detection.pbtxt.

## Step 6: Train

Goal: We can finally train the model! Let's teach this puppy how to do a new trick: Find CAFO sites in satellite imagery.

Description: We will use a script from the TensorFlow models repo to execute the training epochs (feedforward + backpropagation) for the CNN. You can find the script in this repo at 4_train.py but I recommend that you clone the [TensorFlow models repo](https://github.com/tensorflow/models/tree/master/research/object_detection) and execute train.py from that directory, referencing your workspace and configuration file as needed. 

Neural Network Pattern Detection (https://github.com/Qberto/ML_ObjectDetection_CAFO/blob/master/doc/img/neuralnetwork.jpg)

To execute training (finally!) run the following command:

'''
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
'''

You should now start seeing the training epochs, with a loss value provided after each iteration. As the model continues training, the loss should decrease. The goal in this case is to approach less than 2. In my Laptop with an NVIDIA Quattro, it took roughly a full day (22 hours), but in other systems or better GPUs it should be faily quick. I also did not configure it to correctly use my GPU, so more pending on that item...

Every so many steps, the training script will export a checkpoint of the model in its current state. Once we reach a consistently low number (~ 2 in our case), we can stop the training process. Before doing so though, take a look at your designated training directory and confirm that you can see "model.ckpt-<stepcount>.data-00000-of-00001", "model.ckpt-<stepcount>.index", and "model.ckpt<stepcount>.meta" files. These are the output of our training!

Be easy on your model as it learns a whole new way of seeing the world for you! 

## Step 7: Export an inference graph from the new trained model

Goal: We need to export an inference graph from our model to test how it performs in real time. 

Description: The models/object_detection directory has a script that does this for us: export_inference_graph.py. It has also been [included in this repo](https://github.com/Qberto/ML_ObjectDetection_CAFO/5_export_inference_graph.py).

We run the script by passing it our checkpoint file and the configuration file from the earlier steps. Here's a sample of a call to the script:

'''
python 3_export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_cafo.config\
    --trained_checkpoint_prefix training/model.ckpt-9160\
    --output_directory cafo_inference_graph
'''

In this case, the graph is exported to the [cafo_inference_graph](https://github.com/Qberto/ML_ObjectDetection_CAFO/cafo_inference_graph) directory. We're ready to test in real time!

## Step 8: Detect CAFO sites!

Goal: Run our inference graph from Jupyter Notebook and detect images in a specified folder or in a video capture. 

Description: A [Jupyter Notebook](https://github.com/Qberto/ML_ObjectDetection_CAFO/object_detection_CAFO_staticimages.ipynb) is provided to execute the inference graph and point it at a folder of images. Another [Jupyter Notebook](https://github.com/Qberto/ML_ObjectDetection_CAFO/object_detection_CAFO_screencap.ipynb) is provided to execute the inference graph, but this notebook uses Python's PIL library to capture a quadrant of our current screen to pass to the model. The output should be images with bounding boxes and the confidence level of the objects detected in each image. 

## Step 9: Convert Information to GIS

Goal: We need to transfer the bounding box information from detected objects to GIS features.

Description: Work and details pending... stay tuned!

## Step 9: ...

## Step 10: Party.





## Appendix A: CNN Model General

The CNN Model pseudocode can be categorized by two general steps: Feedforward and backpropagation. In other terms, the movement of the data through the neural network, and the learning or training of the model given results in an iteration. A single feedforward and backpropagation loop is known as an "epoch".

#### Feedforward: 
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer

#### Backpropagation:
compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer, Stochastic Gradient Descent, AdaGrad, etc.)

#### Epoch:
feedforward + backpropagation
