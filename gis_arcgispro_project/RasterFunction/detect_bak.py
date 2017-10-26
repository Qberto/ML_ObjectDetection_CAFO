import numpy as np

class Detect():

    def __init__(self):
        self.name = "Detect Object Function"
        self.description = ""

        self.modelPath = None
        self.clsLblPath = None
        self.threshold = 0.5
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
        #self.modelPath = kwargs.get('model', None)
        #_, _, self.padding = lel.get_train_info(self.model)
        self.batch = kwargs.get('batch', 256)
        self.threshold = kwargs.get('thres')

        kwargs['output_info']['bandCount'] = 1
        kwargs['output_info']['statistics'] = ()    # we know nothing about the stats of the outgoing raster. 
        kwargs['output_info']['histogram'] = ()     # we know nothing about the histogram of the outgoing raster.
        kwargs['output_info']['pixelType'] = 'u1'   
        kwargs['output_info']['resampling'] = True

        return kwargs

    def updatePixels(self, tlc, size, props, **pixelBlocks):
        tile_data = np.array(pixelBlocks['input_pixels'], copy = False) 
        b, h, w = tile_data.shape
        output_data = np.zeros((1, h, w), 'u1') 

        boxes=np.load("e:\\data\\objdet\\boxes.npy") # shape (1,100,4)
        boxes=boxes[0] # (100,4)
        scores=np.load("e:\\data\\objdet\\scores.npy") # shape (1,100)
        scores=scores[0] # (100)

        thres=self.threshold
        for box,score in zip(boxes,scores):
            if score>thres:
                # [0,0] is upper left corner
                left=int(box[0]*w)
                right=int(box[2]*w)
                up=int(box[1]*h)
                down=int(box[3]*h)

                #output_data[0][up:down,left:right]=1 # should be class value when available
                #output_data[0][min(up+3,down):max(down-3,up),min(left+3,right):max(right-3,left)]=0 # cut out middle part, make a ring

                output_data[0][up:min(up+3,down),left:right]=1 # up edge
                output_data[0][max(down-3,up):down,left:right]=1 # down edge
                output_data[0][up:down,left:min(left+3,right)]=1 # left edge
                output_data[0][up:down,max(right-3,left):right]=1 # right edge

        #np.save('E:\\temp\\out.npy',output_data)
        pixelBlocks['output_pixels'] = output_data
        return pixelBlocks

