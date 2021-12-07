### Age Prediction with Facial Landmarks

This repository is based off the "IMDB-WIKI – 500k+ face images with age and gender labels" paper [1]. The purpose of this repo was to see the effect of using facial landmarking masks in age prediction (in particular if facial landmarks/bounding boxes helps in age prediction). 


#### to run experiments:
A data directory (data_dir) is needed for all three commands. Essentially a spot where all the experiment outputs will be kept (likely hard drive or somewhere with at least 20 GB of space). Data will be downloaded (in directory called data) and log and model directories (to store logs of experiments and model checkpoints) will be present as subdirectories of data_dir. 
Note: if data_dir = "/home/name" then this directory will have three subdirectories called "data", "models", and "logs".

1. to download the data, run this command: (fill in data_dir with your data_dir). Note: may need to run in tmux session because will take a while.
    `bash setup.sh [data_dir]`
2. to create the csvs needed for dataloading, run:
    `generate_data_csvs.py --data_dir [data_dir]`
3. to run experiments: (this is an example run which has experiment name 'ignore' so you can ignore it in the final results. 
    ` python main.py --model_type="AlteredVIT" --mask_info="1CHANNEL" --optimizer_type="adam" --scheduler_type="cosine" --loss_type="NLL" --train_data="train" --bs=16 --num_workers=4 --lr=0.0001 --num_epochs=50 --output_type="classification" --pre_mask="0" --exp_name="ignore"  --data_source="wiki" --dropout=0.20 --data_dir=[data_dir]`

Note: "incorrect.txt" in notebook_prototyping contains list of examples which do not have a bounding box when the detector was used. Might want to think about implications of not having faces without a bounding box while training etc.

References:
1. Rothe, R., Timofte, R., and Gool, L. V. Deep expectation ofreal and apparent age from a single image without facial landmarks. International Journal of Computer Vision,126(2-4):144–157, 2018.
2. Facial landmarks with dlib, OpenCV, and Python by Adrian Rosebrock (ppublished april 3, 2017 on pyimagesearch)
3. Detect eyes, nose, lips, and jaw with dlib, OpenCV, and Python by Adrian Rosebrock (published april 10, 2017 on pyimagesearch)