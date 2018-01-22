import numpy as np
import tensorflow as tf

class Detect():

    def __init__(self):
        self.name = "Detect Object Function"
        self.description = ""

        self.modelPath = None
        self.clsLblPath = None
        self.threshold = 0.5
        
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

        kwargs['output_info']['bandCount'] = 1
        kwargs['output_info']['statistics'] = ()    # we know nothing about the stats of the outgoing raster. 
        kwargs['output_info']['histogram'] = ()     # we know nothing about the histogram of the outgoing raster.
        kwargs['output_info']['pixelType'] = 'u1'   
        kwargs['output_info']['resampling'] = True

        return kwargs

    def updatePixels(self, tlc, size, props, **pixelBlocks):
        tile_data = np.array(pixelBlocks['input_pixels'], copy = False) # [b,h,w]
        b, h, w = tile_data.shape
        
        # output 
        boxes = None # (100,4)
        scores = None # (100)

        detGra = tf.Graph()
        with detGra.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.modelPath, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
            # define IO tensor of the session run
            imgTs = detGra.get_tensor_by_name('image_tensor:0') # [1,h,w,b]
            boxTs = detGra.get_tensor_by_name('detection_boxes:0')
            scoreTs = detGra.get_tensor_by_name('detection_scores:0')

            with tf.Session(graph=detGra) as sess:
                # Expand dimensions since the model expects images to have shape: [1, h, w, 3]
                image_np = np.rollaxis(tile_data,0,3) # [h,w,b]
                image_np_expanded = np.expand_dims(image_np, axis=0) # [1,h,w,b]
                (boxes, scores) = sess.run([boxTs, scoreTs],feed_dict={imgTs: image_np_expanded})

        # output 
        boxes = boxes[0] # (100,4)
        scores = scores[0] # (100)

        output_data = np.zeros((1, h, w), 'u1') 
        thres=self.threshold
        for box,score in zip(boxes,scores):
            if score>thres:
                # [0,0] is upper left corner
                up=int(box[0]*h)
                down=int(box[2]*h)
                left=int(box[1]*w)
                right=int(box[3]*w)

                output_data[0][up:min(up+3,down),left:right]=1 # up edge
                output_data[0][max(down-3,up):down,left:right]=1 # down edge
                output_data[0][up:down,left:min(left+3,right)]=1 # left edge
                output_data[0][up:down,max(right-3,left):right]=1 # right edge

        pixelBlocks['output_pixels'] = output_data
        return pixelBlocks

