## Implementation based on the paper Li et al., “A Convolutional Neural Network Cascade for Face Detection, ” 2015 CVPR

### A few modifications to the paper:<br>
1. Multi-resolution is not used for simplicity, you can add them in the .prototxt files under **CNN_face_detection_models** to do so.<br>
2. 12-net is turned into fully convolutional neural network to reduce computation.
3. I took out the normalization layers out of the **deploy.prototxt** files in 48-net and 48-calibration-net, because of convenience for me implementing them in hardware, you can just simply at them back as in the corresponding **train_val.prototxt** files.

### In order to test CNN Cascade: 
Detection scripts are stored under **CNN_face_detection/face_detection** directory, 
and models can be found in **CNN_face_detection_models** repository.

For testing single image, use script **face_cascade_fullconv_single_crop_single_image.py**<br>
For benchmarking on FDDB, use script **face_cascade_fullconv_fddb.py**

If you're not familiar with caffe's flow yet, dennis-chen's reply [here](https://github.com/BVLC/caffe/issues/550) gives a great picture.

### In order to train CNN Cascade: 
1. You should first download all faces from the AFLW dataset, and at least 3000 images without any faces (negative images).
2. Create negative patches by running **face_preprocess_10kUS/create_negative.py** with data_base_dir modified to the folder containing the negative images.
3. Create positive patches by running **face_preprocess_10kUS/aflw.py**
4. Run **face_preprocess_10kUS/shuffle_write_positives.py** and **face_preprocess_10kUS/shuffle_write_negatives.py** to shuffle and write position and labels of images to file.
5. Run **face_preprocess_10kUS/write_train_val.py** to create train.txt, val.txt and move images to corresponding folders as caffe requires.
6. Use scripts in **CNN_face_detection_models/create_lmdb_scripts/** to create lmdb files as caffe requires.
7. Start training by using such commands in terminal. <br>
`./build/tools/caffe train --solver=models/face_12c/solver.prototxt`

24 net and 48 net can be created in a similar way, however negative images shoud be created by running **face_preprocess_10kUS/create_negative_24c.py** and **face_preprocess_10kUS/create_negative_48c.py**

Calibration nets are also trained similarly, scripts can be found in **face_calibration/**

