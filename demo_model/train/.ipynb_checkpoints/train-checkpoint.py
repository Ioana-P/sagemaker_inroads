from __future__ import print_function

import argparse
import os
import pandas as pd


import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn.cluster import KMeans


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "kmeans_cluster.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--n-clusters', type=int, default=3)

    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "iris_data_test.csv"), header=None, names=None)
        

    ## TODO: Define a model 
    model = KMeans(
            n_estimators=args.n_clusters,
            )
    
    
    ## TODO: Train the model
    model.fit(train_data)
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))