## Welcome to the Deep Learning toolbelt

Here be dragons, and image classifications

### Tested on

- [Kaggle's Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
- [Kaggle's State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

### Main features
1. Use ResNet-50 pretrained on imagenet with a new head layer
2. Cache of heavy computation to disk for fast re-runs
3. Confusion matrix generation
4. Kaggle sample file generation

### Prerequisits:

Create directories, and populate them with your data:
- train
	- c0
    - c1
    ...
- valid
	- c0
    - c1
    ...
- test
	- unknown
- results

### Usage

To train, run 

    python train.py
    
To predict an image, run 

    ./predict.sh image-url-or-file-path   2> /dev/null



## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Keras](https://keras.io/getting-started/faq/#how-should-i-cite-keras)

## License

- All code in this repository is under the MIT license as specified by the LICENSE file.
- The pretrained ResNet50 imagenet weights are ported from the ones [released by Kaiming He](https://github.com/KaimingHe/deep-residual-networks) under the [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE).
