from common import *
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import argparse
from PIL import Image
import requests
from StringIO import StringIO
import re

def start(args):
    image_path = args.image_path
    is_dog = predict_image(image_path)
    print ("DOG" if is_dog else "CAT") + ' (' + image_path + ')'

def predict_image(image_path):
    training_path, validation_path, test_path, results_path, batch_size, input_shape, weights_file, model_file, sample_submission_file, output_submission_file = setup()

    number_of_classes = 2
    base_model = get_headless_base_model(input_shape=input_shape)
    new_head_model = get_new_head_model(input_shape=(1, 1, 2048), output_shape=number_of_classes)

    image_as_array = image_path_to_array(args.image_path)

    base_model_predictions = base_model.predict(image_as_array)
    head_predictions = new_head_model.predict(base_model_predictions)

    is_dog = head_predictions[0][0] > 0.5

    return is_dog


def image_path_to_array(image_path):
    if is_valid_url(image_path):
        response = requests.get(image_path)
        image_as_pil = Image.open(StringIO(response.content))
        size = (300, 300)
        image_as_pil.thumbnail(size, Image.ANTIALIAS)
        resized_image = Image.new('RGB', size, (255, 255, 255))  # with alpha
        resized_image.paste(image_as_pil, ((size[0] - image_as_pil.size[0]) / 2, (size[1] - image_as_pil.size[1]) / 2))
        image_as_pil = resized_image
    else:
        image_as_pil = image.load_img(image_path, target_size=(300, 300))

    image_as_array = image.img_to_array(image_as_pil)
    image_as_array = np.expand_dims(image_as_array, axis=0)
    image_as_array = preprocess_input(image_as_array)
    return image_as_array

def is_valid_url(url):
    regex = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return not not (url is not None and regex.search(url))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    start(args)


