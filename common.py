from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import keras.applications
import os
from sklearn.metrics import confusion_matrix
import bcolz
import numpy as np
import pandas as pd
from tqdm import trange
import random

def setup():
    #path = 'sample/'
    path = ''

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(random.randint(0, 15))

    training_path = path + 'train/'
    validation_path = path + 'valid/'
    test_path = path + 'test/'
    results_path = path + 'results/'

    weights_file = results_path + "head.h5"
    model_file = results_path + "head_model.h5"

    #sample_submission_file = results_path + "sample_submission_stg1.csv"
    sample_submission_file = results_path + "sample_submission.csv"
    output_submission_file = results_path + "submission.csv"
    batch_size = 64
    input_shape = (300, 300, 3)

    return training_path, validation_path, test_path, results_path, batch_size, input_shape, weights_file, model_file, sample_submission_file, output_submission_file

def get_headless_base_model(input_shape):
    #print("Started with base model of type ResNet50 with input shape:" + str(input_shape))
    return keras.applications.ResNet50(include_top=False, input_shape=input_shape)

def get_base_predictions(base_model, results_path, directory_to_search, class_mode='categorical'):
    print("get_base_predictions for directory: " + directory_to_search)

    data = {}
    cache_file = results_path + 'get_base_predictions_' + directory_to_search.replace("/", "-") + ".bc"
    try:
        data = load_array(cache_file)
    except:
        gen = ImageDataGenerator() #rescale=1./255
        input_shape = base_model.input_shape[1:3] # todo: needed?

        batches = gen.flow_from_directory(directory_to_search, input_shape, shuffle=False, class_mode=class_mode)
        print('*** Predict on base model')
        bottleneck = base_model.predict_generator(batches, batches.nb_sample)
        categories = to_categorical(batches.classes)

        base_out = bottleneck
        expected_out = categories
        number_of_classes = batches.nb_class
        filenames = batches.filenames

        data = {
            "base_out": base_out,
            "expected_out": expected_out,
            "number_of_classes": number_of_classes,
            "filenames": filenames,
        }
        save_array(cache_file, data)

    return data['base_out'], data['expected_out'], data['number_of_classes'], data.get('filenames',[])

def get_new_head_model(input_shape, output_shape):
    model = Sequential([
        GlobalAveragePooling2D(input_shape=input_shape),
        Dropout(0.3),
        # Flatten(),
        # Dense(128, activation='relu'),
        # Dropout(0.25),
        Dense(output_shape, activation="softmax")
    ])
    return model

def evaluate(valid_head_expected_out, valid_head_predicted_out):
    expected_categories = [np.argmax(x) for x in valid_head_expected_out]
    predicted_categories = [np.argmax(x) for x in valid_head_predicted_out]
    cm = confusion_matrix(expected_categories, predicted_categories)
    print(cm)

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[0]



def create_submission_file_for_state_farm(test_head_predicted_out, test_filenames, sample_submission_file, output_submission_file):
    print('*** Create submission file')
    test_ids = [x.split(os.path.sep)[1] for x in test_filenames]
    #test_ids = [int(x.split(os.path.sep)[1].split(".")[0]) for x in filenames]

    number_of_rows = len(test_head_predicted_out)
    number_of_columns = len(test_head_predicted_out[0])

    submission = pd.read_csv(sample_submission_file)
    image_label = submission.loc[1].keys()[0]
    column_labels = submission.loc[1].keys()[1:]
    for row_id in trange(number_of_rows):
        image_id = test_ids[row_id]
        #print(str(row_id) + " / " + str(number_of_rows))
        for column_id in range(0, number_of_columns):
            column_label = column_labels[column_id]
            submission.loc[submission[image_label] == image_id, column_label] = test_head_predicted_out[row_id][column_id]

    submission.to_csv(output_submission_file, index=False)
    return submission



def create_submission_file(test_head_predicted_out, filenames, sample_submission_file, output_submission_file):
    print('*** Create submission file')
    #test_ids = [x.split(os.path.sep)[1] for x in filenames]
    test_ids = [int(x.split(os.path.sep)[1].split(".")[0]) for x in filenames]

    number_of_rows = len(test_head_predicted_out)
    number_of_columns = len(test_head_predicted_out[0])

    submission = pd.read_csv(sample_submission_file)
    # for row_id in range(0, number_of_rows):
    #     for column_id in range(1, number_of_columns):
    #         submission.loc[submission.image == test_ids[0]][column_id] = test_head_predicted_out[row_id][column_id - 1]

    for row_id in range(number_of_rows):
        submission.loc[submission.id == test_ids[row_id], "label"] = test_head_predicted_out[:, 1][row_id]
        # submission.loc[submission.id == test_head_predicted_out[row_id], "label"] = test_ids[row_id]
        # subm.loc[subm.id == ids[i], "label"] = y_test[:, 1][i]

    submission.to_csv(output_submission_file, index=False)
    return submission


def delete_weights_and_model_files(weights_file, model_file):
    try:
        os.remove(weights_file)
    except OSError:
        pass

    try:
        os.remove(model_file)
    except OSError:
        pass


