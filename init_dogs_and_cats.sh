#!/bin/bash

export LC_ALL=C

sudo apt-get install unzip
unzip train.zip
unzip test.zip

mkdir results
mkdir valid
mkdir valid/cats
mkdir valid/dogs
mkdir test/unknown
mkdir train/cats
mkdir train/dogs

mkdir sample
mkdir sample/results
mkdir sample/valid
mkdir sample/valid/cats
mkdir sample/valid/dogs
mkdir sample/test
mkdir sample/test/unknown
mkdir sample/train
mkdir sample/train/dogs
mkdir sample/train/cats

cp sample_submission.csv results
cp sample_submission.csv sample/results

sudo mv test/*.jpg test/unknown/ -f
sudo mv train/cat.* train/cats -f
sudo mv train/dog.* train/dogs -f

sudo mv train/dogs/dog.1????.* valid/dogs -f
sudo mv train/cats/cat.1????.* valid/cats -f

sudo cp train/cats/cat.??.jpg sample/train/cats
sudo cp train/dogs/dog.??.jpg sample/train/dogs
sudo cp valid/cats/cat.121??.jpg sample/valid/cats
sudo cp valid/dogs/dog.121??.jpg sample/valid/dogs
sudo cp test/unknown/??.jpg sample/test/unknown

sudo pip install -r requirements.txt
