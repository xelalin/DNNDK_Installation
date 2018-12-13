import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import tensorflow as tf
import cv2
import caffe
import numpy as np

print "tensorflow version ", tf.__version__
print "openCV     version ", cv2.__version__
print "caffe      version ", caffe.__version__
print "numpy      version ", np.__version__
