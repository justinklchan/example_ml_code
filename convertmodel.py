import keras
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Sequential
from keras.models import Model,Input
from keras.models import load_model
import coremltools
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras import backend as K
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299,299,3))
class_labels=['Hemangioma','Pyogenic Granuloma','Venous Malformation','Capillary Malformation','Spider Angioma','Lymphatic Malformation','Atopic Dermatitis','Milia','Nevus']

i = base_model.output
a = Flatten(name='a1')(i)
a = Dense(256,activation='relu',name='a2')(a)
a = Dropout(.6,name='a3')(a)
o = Dense(9, activation='softmax',name='a4')(a)
model = Model(inputs=base_model.input, outputs=o)

model.load_weights('traintop-4-1-80.h5',by_name=True)

def ios():
    coreml_model = coremltools.converters.keras.convert(
        model,
        input_names="image",
        image_input_names="image",
        image_scale=1/127.5,
        red_bias=-1.0,
        green_bias=-1.0,
        blue_bias=-1.0,
        class_labels=class_labels,
        )
    coreml_model.save('VA.mlmodel')
    
def android(model_name, input_node_name, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        model_name + '_graph.pbtxt')

    tf.train.Saver().save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, \
        False, 'out/' + model_name + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + model_name + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_name], [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/tensorflow_lite_' + model_name + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

# ios()
# android('xor_nn', "input_1", "a4")
        
        
        
        
        