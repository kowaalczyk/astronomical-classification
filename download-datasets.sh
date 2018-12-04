#!/bin/bash

echo "If something fails, it's probably due to bad config of kaggle api"
echo "see https://github.com/Kaggle/kaggle-api#kaggle-api"

kaggle competitions download -c PLAsTiCC-2018 -p data/raw

echo "Unzipping..."

cd data/raw
pwd
unzip "*.zip" && rm -rf *.zip
cd -
