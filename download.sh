#!/bin/sh

wget https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2?tf-hub-format=compressed -O model/compressed_image_stylization.tar.gz

mkdir model/image_stylization
tar xvzf model/compressed_image_stylization.tar.gz --directory=model/image_stylization

