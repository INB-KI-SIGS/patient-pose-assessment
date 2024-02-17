# Patient Pose Assessment

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
The data described in the paper is uploaded as an attachment to the first release of this repository.
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

## Dataset

The dataset contains one folder for each foot. These folder names start with the prefix `patient_` 
followed by a consecutive increasing number and a `l` or `r` delimited by an underscore.
The `l` or `r` describe if the foot is a left or a right one.

Each patient folder contains folders for each flexion. Possible flexion names are `max`, `medium` and `min`.
These flexion folder contain a `data` and in the `data` folder a `dataset` subfolder.

In the `dataset` folder there are the folders `depth`, `xray` and `label`. 

### Depth images
The `depth` folder contains all the depth images as 16 bit single-channel PNGs,
where the value of each pixel is a distance in mm. The depth images a named in the format
`depth_cam_{left,right}_orig_{image_number}_{timestamp}.png`
where `{image_number}` is the number of the pose for this foot, 
`{timestamp}` is an over time increasing timestamp and 
`{left,right}` is the string `left` or `right` depending on the side of the camera.
Note that the timestamp is an arbitrary number and can not be used for time measuring. 

### X-ray images
The `xray` folder contains all X-ray images as 16 bit single-channel PNGs.
The X-ray images a named in the format
`xray_{image_number}.png`, where `{image_number}` is the number of the pose for this foot.


### Labels
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
