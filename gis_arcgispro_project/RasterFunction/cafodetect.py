import numpy as np
import tensorflow as tf

class Detect():

    def __init__(self):
        self.name = "Detect Object Function"
        self.description = ""

        self.modelPath = None
        self.clsLblPath = None
        self.threshold = 0.8
        # self.num_classes = 1
        #self.batch = 128

    def getParameterInfo(self):
        return [
            {
                'name': 'input',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': ""
            },
            {
                'name': 'model',
                'dataType': 'string',
                'value': None,
                'required': True,
                'displayName': "Trained Model",
                'description': ""
            },
            {
                'name': 'clslbl',
                'dataType': 'string',
                'value': None,
                'required': True,
                'displayName': "Class Label",
                'description': ""
            },
            {
                'name': 'thres',
                'dataType': 'numeric',
                'value': 0.5,
                'required': True,
                'displayName': "Score Threshold",
                'description': ""
            }
        ]

    def getConfiguration(self, **scalars):
        return {
            'inheritProperties': 4,
            'invalidateProperties': 2 | 4 | 8,
            #'inputMask': False                         # Need input raster mask in .updatePixels()
        }

    def updateRasterInfo(self, **kwargs):
        self.modelPath = kwargs.get('model', None)
        self.clsLblPath = kwargs.get('clslbl', None)
        self.threshold = kwargs.get('thres')
        
        # Load a frozen TF model into memory
        if not self.detection_graph:
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.modelPath, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')


                 # Definite input and output Tensors for detection_graph
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


        kwargs['output_info']['bandCount'] = 1
        kwargs['output_info']['statistics'] = ()    # we know nothing about the stats of the outgoing raster. 
        kwargs['output_info']['histogram'] = ()     # we know nothing about the histogram of the outgoing raster.
        kwargs['output_info']['pixelType'] = 'u1'   
        kwargs['output_info']['resampling'] = True

        return kwargs

    def updatePixels(self, tlc, size, props, **pixelBlocks):
        tile_data = np.array(pixelBlocks['input_pixels'], copy = False) 
        b, h, w = tile_data.shape
        
        boxes = None
        scores = None 

        with tf.Session(graph=self.detection_graph) as sess:
            # Expand dimensions since the model expects images to have shape: [1, h, w, 3]
            image_np = np.rollaxis(tile_data,0,3) # [h,w,b]
            image_np_expanded = np.expand_dims(image_np, axis=0) # [1,h,w,b]
            (boxes, scores) = sess.run([self.detection_boxes, self.detection_scores],
                feed_dict={self.image_tensor: image_np_expanded})

        # output
        boxes=boxes[0] # (100,4)
        scores=scores[0] # (100)

        output_data = np.zeros((1, h, w), 'u1') 
        thres=self.threshold
        for box,score in zip(boxes,scores):
            if score>thres:
                # [0,0] is upper left corner
                up=int(box[0]*h)
                down=int(box[2]*h)
                left=int(box[1]*w)
                right=int(box[3]*w)

                #output_data[0][up:down,left:right]=1 # should be class value when available
                #output_data[0][min(up+3,down):max(down-3,up),min(left+3,right):max(right-3,left)]=0 # cut out middle part, make a ring

                output_data[0][up:min(up+3,down),left:right]=1 # up edge
                output_data[0][max(down-3,up):down,left:right]=1 # down edge
                output_data[0][up:down,left:min(left+3,right)]=1 # left edge
                output_data[0][up:down,max(right-3,left):right]=1 # right edge

        #np.save('E:\\temp\\out.npy',output_data)
        pixelBlocks['output_pixels'] = output_data
        return pixelBlocks

