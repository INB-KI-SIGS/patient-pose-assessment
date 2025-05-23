# Patient Pose Assessment
This is the source code and the dataset for the paper `Patient Pose Assessment in Radiography using Time-of-Flight Cameras` presented at SPIE Medical Imaging 2024.

**Update: May 20, 2025**:

Our synthetic dataset described in the paper `Synthetic Data Generated from CT Scans for Patient Pose Assessment` (to be presented at **MIDL 2025**) is now included in the [latest release](https://github.com/INB-KI-SIGS/patient-pose-assessment/releases/download/v0.0.2/synthetic_dataset.tar.7z).  
The [Dataset section](#dataset) has been updated so that both datasets are now listed and explained.

## Usage
### Setup
There are different ways to set up an environment for this package.

The easiest one is to create a new conda environment from the `environment.yaml` file.
Therefore, run the following command in the cloned repository folder.
```shell
conda env create -f environment.yaml
```

If you don't use conda as a package manager make sure the packages listed in `requirements.txt` are installed in your environment. 

### Get the Data
The data described in the paper is uploaded as an [attachment to the first release](https://github.com/INB-KI-SIGS/patient-pose-assessment/releases/download/v0.0.1/specimen_dataset.zip) of this repository.
Just download it and unzip the file. In the [Dataset section](#dataset) the structure of the dataset is described.

### Install as Package
This step is optional and only needed if you want to use the provided code from your own code.
If you only want to train the models as described in the paper, e.g. for reproduction, you can skip this. 

To install the provided code as a package set up the environment as described, then run `pip install .` to install the package.

Then you can import the package in your code.
For example to load the dataset, get the first data element, consisting of an image, its quality and some metadata, 
and print its quality you would use:

```python
from pathlib import Path

from patient_pose_assessment.data import DataElement, DepthImageDataset

dataset = DepthImageDataset(
    base_folder_path=Path("/path/to/your/specimen_dataset/data"),  # Change path to correct data path
    test_patients=["patient_1"],
    dataset_type="train",
    seed=1234,
)

data_element: DataElement = dataset.data[0]
print(data_element.quality)
```

### Run Training and Evaluation

#### Training
```shell
mllooper \
    -g git@github.com:INB-KI-SIGS/patient-pose-assessment.git \
    -i @patient-pose-assessment:src/patient_pose_assessment \
    -c @patient-pose-assessment:config/architecture/single-network.yaml \
    -c @patient-pose-assessment:config/base.yaml \
    -c @patient-pose-assessment:config/runner.yaml \
    -y '!DepthImageDataset { test_patients: [patient_1], base_folder_path: /path/to/your/specimen_dataset/data, device: cuda:0 }' \
    -y '!FileLogBase { log_dir: ./training_runs }' \
    -y '!Model { device: cuda:0 }' \
    run @patient-pose-assessment:config/train.yaml
```
The `-g` option clones the repository from GitHub, `-i` imports the package from the cloned repository, 
the `-c` options load several configuration files, which define how the networks are trained, 
the `-y` options are inline configurations and the `run` argument starts the training.

To run the training you should edit the `base_folder_path` according to your dataset path and 
update both occurrences of  `device` to `cpu` if you don't have a GPU.
To edit where the saved models and the metrics are logged to, change the `log_dir`.

If you do not want the git to be cloned automatically, clone it yourself, change to the repository folder, 
and leave out the `@patient-pose-assessment` parts in the command. For example:
```shell
git clone git@github.com:INB-KI-SIGS/patient-pose-assessment.git
cd patient-pose-assessment

mllooper \
    -i src/patient_pose_assessment \
    -c config/architecture/single-network.yaml \
    -c config/base.yaml \
    -c config/runner.yaml \
    -y '!DepthImageDataset { test_patients: [patient_1], base_folder_path: /path/to/your/specimen_dataset/data, device: cuda:0 }' \
    -y '!FileLogBase { log_dir: ./training_runs }' \
    -y '!Model { device: cuda:0 }' \
    run config/train.yaml
```

To train the different architectures just change the line `-c config/architecture/single-network.yaml` 
to a different architecture from the [config/architecture](config%2Farchitecture) folder, 
e.g. `-c config/architecture/late-fusion.yaml`.

#### Metrics and Evaluation
The metrics from the training and the models are saved in the folder you specified as `log_dir`.
Start a tensorboard in this folder to view the metrics.

To evaluate a trained model rerun the train command from above but add the line
`-y '!Model { module_load_file: path/to/the/trained/model.pth }'` and change the last line to 
`run @patient-pose-assessment:config/eval.yaml`. For example:

```shell
mllooper \
    -g git@github.com:INB-KI-SIGS/patient-pose-assessment.git \
    -i @patient-pose-assessment:src/patient_pose_assessment \
    -c @patient-pose-assessment:config/architecture/single-network.yaml \
    -c @patient-pose-assessment:config/base.yaml \
    -c @patient-pose-assessment:config/runner.yaml \
    -y '!DepthImageDataset { test_patients: [patient_1], base_folder_path: /path/to/your/specimen_dataset/data, device: cuda:0 }' \
    -y '!FileLogBase { log_dir: ./training_runs }' \
    -y '!Model { device: cuda:0 }' \
    -y '!Model { module_load_file: training_runs/2024-01-01-12:00:00/EfficientNet.pth }' \
    run @patient-pose-assessment:config/eval.yaml
```

### Edit and Run Training Configurations
To change the training parameters look in the config files to see what can be changed. You can then either
- directly change the values in the yaml config files if you have cloned the repository yourself,
- copy a config file from git, edit it and include a line like `-i path/to/your/own/config.yaml`,
- or add an inline configuration. For example to update the batch size to 64 add `-y '!DataLoaderArgs { batch_size: 64 }'`

## Datasets

### Specimen Dataset
The dataset contains one folder for each foot. These folder names start with the prefix `patient_` 
followed by a consecutive increasing number and a `l` or `r` delimited by an underscore.
The `l` or `r` describe if the foot is a left or a right one.

Each patient folder contains folders for each flexion. Possible flexion names are `max`, `medium` and `min`.
These flexion folder contain a `data` and in the `data` folder a `dataset` subfolder.

In the `dataset` folder there are the folders `depth`, `xray` and `label`. 

### Synthetic Dataset

The synthetic dataset follows a similar structure to the specimen dataset, but **flexions are replaced by augmentations**.

Each patient folder of the 10 patients (e.g., `patient_0`, `patient_1`, ...) contains:
- `depth`
- `labels`



### Depth images
#### Specimen Dataset
The `depth` folder contains all the depth images as 16 bit single-channel PNGs,
where the value of each pixel is a distance in mm. The depth images a named in the format
`depth_cam_{left,right}_orig_{image_number}_{timestamp}.png`
where `{image_number}` is the number of the pose for this foot, 
`{timestamp}` is an over time increasing timestamp and 
`{left,right}` is the string `left` or `right` depending on the side of the camera.
Note that the timestamp is an arbitrary number and can not be used for time measuring. 

#### Synthetic Dataset
Depth images are located at:  

`patient_X/depth/{l,r}/{orig,big,small}/`

Each of these contains depth images for different augmentation levels:
- `orig`: original point cloud
- `big`: point cloud where the points have been shifted in a positive direction along the normal 
- `small`: point cloud where the points have been shifted in a negative direction along the normal 

Files are named using the format:

`depth_cam_{left,right}_{angle}.png`
- `{angle}` is the rotation angle (ranging from`0.0` to `90.0` in 0.5 steps)

All depth images are 16-bit single-channel PNGs, where pixel values indicate distance in millimeters.
### X-ray images
#### Specimen Dataset
The `xray` folder contains all X-ray images as 16 bit single-channel PNGs.
The X-ray images a named in the format
`xray_{image_number}.png`, where `{image_number}` is the number of the pose for this foot.

#### Synthetic Dataset
There are no X-ray images in this dataset.

### Labels
#### Specimen Dataset
The `label` folder contains the folder `quailty` and `roi`. In the `quality` folder there is a file `labels.json`,
which contains all quality labels for this foot.
Its contents is a mapping of the file names of the X-ray images to a mapping with the keys `label`, `std` and `image_number`.
The `std` contains the standard deviation of the quality annotations from the four radiologists.


For example:
```json
"xray_0.png": {
    "label": 1.5,
    "std": 0.3535533905932738,
    "image_number": 0
}
```

In the `roi` folder there is a file `roi_depth.json`, which contains ROIs for each depth image.

#### Synthetic Dataset
Labels are stored in:

`patient_X/labels/{l,r}/`

Each folder contains:
- `labels.json`: label information (e.g., quality)
- `roi_depth.json`: region of interest annotations for the depth image

Although the folder and file structure matches the specimen dataset, **the mapping between label entries and depth images is based on pose steps**.

Each label entry uses a step-based format like this:
```json
"xray_image_180.png": {
     "label": 3.0,
      "std": 0.0,
      "step": 180 
}
```
To map the step to the corresponding depth image:

Use the formula:

```angle = step / 2```

For example:

    step = 180 → angle = 90.0 → depth_cam_{left,right}_90.0.png

    step = 91 → angle = 45.5 → depth_cam_{left,right}_45.5.png