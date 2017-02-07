from common import *

def start():
    train()
    predict()

def train():
    print('*** Train')
    training_path, validation_path, test_path, results_path, batch_size, input_shape, weights_file, model_file, sample_submission_file, output_submission_file = setup()

    base_model = get_headless_base_model(input_shape=input_shape)

    train_headless_out, train_head_expected_out, number_of_classes, train_filenames = get_base_predictions(base_model=base_model, results_path=results_path, directory_to_search=training_path)
    valid_headless_out, valid_head_expected_out, number_of_classes, valid_filenames = get_base_predictions(base_model=base_model, results_path=results_path, directory_to_search=validation_path)

    base_model_output_shape = train_headless_out.shape[1:]

    new_head_model = get_new_head_model(input_shape=base_model_output_shape, output_shape=number_of_classes)

    train_and_save(model=new_head_model,
                   train_in=train_headless_out, train_out=train_head_expected_out,
                   valid_in=valid_headless_out, valid_out=valid_head_expected_out,
                   weights_file=weights_file, model_file=model_file)

def train_and_save(model, train_in, train_out, valid_in=None, valid_out=None, weights_file=None, model_file=None):
    print('*** Train and save')

    delete_weights_and_model_files(weights_file, model_file)

    cb = [ModelCheckpoint(filepath=weights_file, save_best_only=True, save_weights_only=True)]

    model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_in, train_out, callbacks=cb, validation_data=(valid_in, valid_out), nb_epoch=3)
    model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_in, train_out, callbacks=cb, validation_data=(valid_in, valid_out), nb_epoch=3)

    model.load_weights(weights_file)
    model.save(model_file)


def predict():
    print('*** Predict')
    training_path, validation_path, test_path, results_path, batch_size, input_shape, weights_file, model_file, sample_submission_file, output_submission_file = setup()

    base_model = get_headless_base_model(input_shape=input_shape)

    valid_headless_out, valid_head_expected_out, number_of_classes, valid_filenames = get_base_predictions(
        base_model=base_model, results_path=results_path, directory_to_search=validation_path)
    test_headless_out, test_head_expected_out, number_of_test_classes, test_filenames = get_base_predictions(
        base_model=base_model, results_path=results_path, directory_to_search=test_path)

    new_head_model = load_model(results_path + "head_model.h5")
    new_head_model.load_weights(weights_file)

    valid_head_predicted_out = new_head_model.predict(valid_headless_out)


    evaluate(valid_head_expected_out, valid_head_predicted_out)

    test_head_predicted_out = new_head_model.predict(test_headless_out)

    submission = create_submission_file(test_head_predicted_out, test_filenames, sample_submission_file,
                                        output_submission_file)
    print(submission.head())


if __name__ == "__main__":
    start()
