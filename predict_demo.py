from __future__ import print_function
<<<<<<< HEAD
=======
from utils import format_input, lookup_table, get_lookup_table

>>>>>>> 0bfc29a9f8c9c9bb8978a6238540b7906f1680bf
import argparse
import json
import os
import pickle
import sys
<<<<<<< HEAD
# import sagemaker_containers
import pandas as pd
import numpy as np
import boto3
# import mxnet as mx

import joblib

# from utils import format_input, lookup_table, get_lookup_table
=======

#import sagemaker_containers
import pandas as pd
import numpy as np
import boto3
#import mxnet as mx


import joblib


>>>>>>> 0bfc29a9f8c9c9bb8978a6238540b7906f1680bf

## Import any additional libraries you need to define a model
from sklearn.cluster import KMeans

bucket_name = 'sagemaker-eu-west-2-363162872357'
model_loc = 's3://sagemaker-eu-west-2-363162872357/model/'
 
# Provided model load function
def model_fn(model_dir):
    """Load the Sklearn model from the `model_dir` directory."""
    print("Loading model.")

#     # First, load the parameters used to create the model.
#     s3_dir = 'sagemaker-eu-west-2-363162872357'
#     model_key = "model/" + s3_dir + "/model.tar.gz"
    
    # Ioana: 25.08.2020
    # load using joblib
    # You'll note that this is significantly shorter to write out. 
    # We will probably have to upgrade this to be able to take from an s3 bucket
    # but you've already done so below
<<<<<<< HEAD
#     model = joblib.load(os.path.join(model_dir, "model.joblib"))
#     print("Done loading model.")
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, 'model/model.tar.gz', 'model/model.tar.gz')
    os.system('tar -zxvf model.tgz')
=======
#     model = joblib.load(os.path.join(model_dir, "kmeans_cluster.joblib"))
#     print("Done loading model.")
    bucket_name = 'sagemaker-us-east-1-068949824886'
    sagemaker_model = boto3.resource('s3').Bucket(bucket_name).download_file(model_dir, 'model.tar.gz')
#    sagemaker_model = os.system('tar -zxvf model.tar.gz')
    
>>>>>>> 0bfc29a9f8c9c9bb8978a6238540b7906f1680bf
#     sagemaker_model = MXNetModel(model_data='s3://'+ model_key,
#                              role='arn:aws:iam::accid:sagemaker-role',
#                              entry_point='utils.py')
    
    
    print("Done loading model.")
    return 


# def input_fn(serialized_input_data, content_type):
#     print('Deserializing the input data.')
#     if content_type == 'text/plain':
#         data = serialized_input_data.decode('utf-8')
#         return data
#     raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

# def output_fn(prediction_output, accept):
#     print('Serializing the generated output.')
#     return str(prediction_output)

# def predict_fn(input_data, model):
#     print('Determining nearest cluster.')

<<<<<<< HEAD
    #calling lookuptable
=======
#     #calling lookuptable
>>>>>>> 0bfc29a9f8c9c9bb8978a6238540b7906f1680bf
#     lookup_table = get_lookup_table()
    
#     # process input data and turn to numpy array using lookup table
#     formatted_input_data = format_input(input_data)
<<<<<<< HEAD
#     vectorised_input = lookup_table(lookup_table, formatted_input_data)
    
    output = model.predict(vectorised_input)
    
    return output
=======
#     vectorised_input = lookup_table(search_table = lookup_table, formatted_input_data)
    
#     output = sagemaker_model.predict(vectorised_input)
    
#     return result
>>>>>>> 0bfc29a9f8c9c9bb8978a6238540b7906f1680bf
