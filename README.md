# Quantifying Building Damage Using Satellite Imagery

This project's goal is to use satellite imagery to understand building damage before and after disasters using the xBD Dataset.

## Dataset

We are using xBD, which is a large-scale dataset for the ad-
vancement of change detection and building damage assess-
ment. The dataset includes post-disaster imagery with trans-
posed polygons from before the disaster over the buildings,
as well as damage classification labels. The dataset provides
four damage categories - No Damage, Minor Damage, Major
Damage, and Destroyed for each building. Over 45,000KM2
of polygon labeled pre and post-disaster imagery is included
in the dataset. The xBD dataset is provided in the train, test
and holdout splits in a 80/10/10% ratio, respectively.

## Steps Involved

1. Data Prep-processing -  where we preprocess the satellite images so that they are model ready, by generating a BW mask. 
2. Image Segmentation - here we extract patches of buildings present in a satellite image    
* FCDenseNet - https://github.com/GeorgeSeif/Semantic-Segmentation-Suite

* SegFormer - https://huggingface.co/blog/fine-tune-segformer

3. Damage Classification - where we have built a model to assess the extent of damage in post-disaster imagery
* Siamese Network - https://gitlab.com/awadailab/sage-project/-/tree/main


