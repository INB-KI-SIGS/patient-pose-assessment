import json
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Literal, Tuple, Any, Union

import albumentations
import cv2
import numpy as np
import pydantic
import torch
import yaloader
from mllooper import State
from mllooper.data import IterableDataset
from mllooper.data.dataset import Dataset, DatasetConfig
from mllooper.logging.messages import ImageLogMessage
from torchvision.utils import make_grid

from patient_pose_assessment.data.image_transforms import AddInvalidPoints
from patient_pose_assessment.data.types import (
    FLEXION_TYPE,
    FLEXIONS,
    SIDE_TYPE,
    CAMERA_TYPE,
    CAMERAS,
)


class DataElement(pydantic.BaseModel):
    patient_name: str
    image_number: int
    side: SIDE_TYPE
    flexion: FLEXION_TYPE

    quality: float

    camera: Optional[CAMERA_TYPE]

    cache: bool = False

    timestamp: int

    image_size: int
    image_path: Path = None

    roi_top_left_x: Optional[int] = None
    roi_top_left_y: Optional[int] = None
    roi_size: Optional[int] = None

    _image: Optional[np.ndarray] = pydantic.PrivateAttr(default=None)

    depth_threshold: float

    @property
    def image(self):
        if self._image is not None:
            image = self._image
            return (
                np.clip(image, a_min=0, a_max=self.depth_threshold, dtype=np.float32) / self.depth_threshold
            )

        image = cv2.imread((str(self.image_path)), -1)
        image = image[
            self.roi_top_left_y : self.roi_top_left_y + self.roi_size,
            self.roi_top_left_x : self.roi_top_left_x + self.roi_size,
        ]
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST_EXACT)

        if self.cache:
            self._image = image
        return np.clip(image, a_min=0, a_max=self.depth_threshold, dtype=np.float32) / self.depth_threshold


class DepthImageDataset(Dataset, ABC):
    def __init__(
        self,
        base_folder_path: Path,
        test_patients: Optional[List[str]] = None,
        cameras: Optional[List[CAMERA_TYPE]] = None,
        cache_data: bool = False,
        images_per_xray: int = 1,
        image_size: int = 336,
        depth_threshold: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_folder_path = base_folder_path
        self.test_folder_prefixes = [] if test_patients is None else test_patients
        self.cameras = cameras if cameras is not None else CAMERAS.copy()
        self.cache_data = cache_data

        self.images_per_xray = images_per_xray
        self.image_size = image_size

        self.data: List[DataElement]
        self.depth_threshold = depth_threshold

        self.data: List[Union[DataElement, Tuple[DataElement, ...]]] = []
        self.load_patient_folders()

        self.transform = albumentations.Compose(self._augmentations)

    @property
    def _augmentations(self) -> List[albumentations.BasicTransform]:
        if self.train:
            return [
                albumentations.HorizontalFlip(),
                albumentations.Rotate(limit=15, border_mode=cv2.BORDER_REPLICATE),
                albumentations.GaussNoise(var_limit=0.001),
                albumentations.Blur(blur_limit=(3, 7), p=0.7),
                AddInvalidPoints(
                    top_gradient_percentage=0.7,
                    min_selected_indices=0.1,
                    max_selected_indices=0.5,
                    p=0.7,
                ),
                albumentations.RandomSizedCrop(
                    min_max_height=(self.image_size, int(self.image_size * 1.2)),
                    height=self.image_size,
                    width=self.image_size,
                    interpolation=cv2.INTER_NEAREST_EXACT,
                ),
            ]
        else:
            return [
                albumentations.Resize(
                    height=self.image_size,
                    width=self.image_size,
                    interpolation=cv2.INTER_NEAREST_EXACT,
                )
            ]

    def load_patient_folders(self):
        for patient_folder in self.base_folder_path.glob("patient_*/"):
            # noinspection PyTypeChecker
            side: SIDE_TYPE = patient_folder.name.split("_")[-1]

            for flexion in FLEXIONS:
                folder_prefix = f"{patient_folder.name}/{flexion}"
                is_test_folder = any(
                    map(
                        lambda test_folder_prefix: folder_prefix.startswith(test_folder_prefix),
                        self.test_folder_prefixes,
                    )
                )
                if is_test_folder == (self.type == "test"):
                    self.data.extend(self.load_folder(patient_folder, side, flexion))

    def load_folder(
        self, patient_folder: Path, side: SIDE_TYPE, flexion: FLEXION_TYPE
    ) -> List[Union[DataElement, Tuple[DataElement, ...]]]:
        patient_name = patient_folder.stem
        dataset_folder = patient_folder.joinpath(flexion, "data", "dataset")
        quality_labels = self.load_quality_labels(dataset_folder)

        data = []
        for camera in self.cameras:
            roi_labels = self.load_roi_labels(dataset_folder, camera)
            for image_number, quality_label in quality_labels.items():
                roi_label = roi_labels[image_number]
                data.extend(
                    self.load_modalities(
                        dataset_folder,
                        patient_name,
                        side,
                        flexion,
                        camera,
                        image_number,
                        quality_label,
                        roi_label,
                    )
                )

        return data

    def load_modalities(
        self,
        dataset_folder: Path,
        patient_name: str,
        side: SIDE_TYPE,
        flexion: FLEXION_TYPE,
        camera: CAMERA_TYPE,
        image_number: int,
        quality_label: float,
        roi_label: Dict,
    ) -> List[DataElement]:
        depth_images_folder = dataset_folder.joinpath("depth")
        depth_image_paths = list(depth_images_folder.glob(f"depth_{camera}_orig_{image_number}_*.png"))
        # Sort based on timestamp
        depth_image_paths.sort(key=lambda path: int(path.stem.split("_")[-1]))
        depth_image_paths = depth_image_paths[: self.images_per_xray]

        data = {}
        for depth_image_path in depth_image_paths:
            timestamp = int(depth_image_path.stem.split("_")[-1])
            data[(patient_name, side, flexion, image_number, camera, timestamp)] = DataElement(
                patient_name=patient_name,
                image_number=image_number,
                side=side,
                flexion=flexion,
                quality=quality_label,
                camera=camera,
                cache=self.cache_data,
                timestamp=timestamp,
                image_size=int(self.image_size * 1.2),
                image_path=depth_image_path,
                roi_top_left_x=roi_label["x_start"],
                roi_top_left_y=roi_label["y_start"],
                roi_size=roi_label["size"],
                depth_threshold=self.depth_threshold,
            )
        return list(data.values())

    @staticmethod
    def load_quality_labels(dataset_folder: Path) -> Dict[int, float]:
        label_path = dataset_folder.joinpath("label", "quality", "labels.json")

        if not label_path.exists():
            raise FileNotFoundError(f"Label file does not exist: {label_path}")

        labels = json.loads(label_path.read_text())
        labels = {v["image_number"]: v["label"] for v in labels.values()}
        return labels

    @staticmethod
    def load_roi_labels(dataset_folder: Path, camera: CAMERA_TYPE) -> Dict[int, Dict]:
        rois_path = dataset_folder.joinpath("label", "roi", "roi_depth.json")

        if not rois_path.exists():
            raise FileNotFoundError(f"ROI file does not exist: {rois_path}")

        rois = json.loads(rois_path.read_text())
        rois = {
            roi["image_number"]: roi for roi in filter(lambda roi: roi["camera"] == camera, rois.values())
        }
        return rois

    def get_data_element(self, data_element: DataElement):
        transformed_images = self.transform(image=data_element.image)
        depth_image = transformed_images["image"] - 0.5

        data = {
            "input": np.expand_dims(depth_image, 0).astype(np.float32),
            "target": np.array(data_element.quality).astype(np.float32),
            "patient_name": data_element.patient_name,
            "side": data_element.side,
            "flexion": data_element.flexion,
            "image_number": data_element.image_number,
            "camera": data_element.camera,
            "timestamp": data_element.timestamp,
        }

        return data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data_element = self.data[index]
        return self.get_data_element(data_element)

    def __len__(self):
        return len(self.data)

    def _log(self, state: State) -> None:
        dataset_name = state.dataset_state.name
        data = state.dataset_state.data
        if data is None or not "input" in data:
            return

        images = data["input"][:4].detach().cpu()

        # Undo the normalization
        images = (images + 0.5) * 255.0

        # Order in a grid
        grid_image = make_grid(images, 2, normalize=False, scale_each=False)

        # Get a [0, 255] uin8 image/label
        log_image = (grid_image.numpy()).astype(np.uint8)

        self.logger.info(
            ImageLogMessage(tag=f"{dataset_name}", step=state.looper_state.step_iteration, image=log_image)
        )


class DepthImageDatasetIter(DepthImageDataset, IterableDataset, ABC):
    def __init__(self, base_folder_path: Path, nr_buckets: int = 1, **kwargs):
        super().__init__(base_folder_path, **kwargs)
        self.nr_buckets = nr_buckets
        self._buckets: Dict[Any, List[Union[DataElement, Tuple[DataElement, ...]]]] = {}
        self._bucket_intervals = np.linspace(1, 3, self.nr_buckets + 1)
        self._not_empty_bucket_keys = []

        self.sort_data_into_buckets()

    def sort_data_into_buckets(self):
        self._buckets: Dict[Any, List[Union[DataElement, Tuple[DataElement, ...]]]] = {}

        for data_element in self.data:
            bucket_key = self._get_sampling_bucket(data_element)
            try:
                self._buckets[bucket_key].append(data_element)
            except KeyError:
                self._buckets[bucket_key] = [data_element]
        self._not_empty_bucket_keys = list(
            filter(lambda key: len(self._buckets[key]) > 0, self._buckets.keys())
        )
        if len(self._not_empty_bucket_keys) == 0:
            raise RuntimeError

    def __next__(self):
        bucket_key = self.random.choice(self._not_empty_bucket_keys)
        data_element: Union[DataElement, Tuple[DataElement, ...]] = self.random.choice(
            self._buckets[bucket_key]
        )
        return self.get_data_element(data_element)

    def _get_sampling_bucket(
        self,
        data_element: DataElement | Tuple[DataElement, ...] | Tuple[Tuple[DataElement, ...], ...],
    ) -> Any:
        if isinstance(data_element, DataElement):
            quality = data_element.quality
        elif isinstance(data_element, tuple) and isinstance(data_element[0], DataElement):
            quality = data_element[0].quality
        elif (
            isinstance(data_element, tuple)
            and isinstance(data_element[0], tuple)
            and isinstance(data_element[0][0], DataElement)
        ):
            quality = data_element[0][0].quality
        else:
            raise RuntimeError
        bucket = np.max(np.where(quality >= self._bucket_intervals))
        return np.clip(bucket, 0, self.nr_buckets - 1)


class DepthImageCoupleDataset(DepthImageDataset):
    def __init__(self, cameras: Optional[List[CAMERA_TYPE]] = None, **kwargs):
        self.data: List[Tuple[DataElement, DataElement]]

        if cameras is not None and set(cameras) != set(CAMERAS):
            raise ValueError(f"{self.name} requires all two cameras.")

        super().__init__(cameras=cameras, **kwargs)
        self.transform = albumentations.Compose(
            self._augmentations,
            additional_targets={camera_name: "image" for camera_name in self.cameras[1:]},
        )

    def load_folder(
        self, patient_folder: Path, side: SIDE_TYPE, flexion: FLEXION_TYPE
    ) -> List[Union[DataElement, Tuple[DataElement, ...]]]:
        patient_name = patient_folder.stem
        dataset_folder = patient_folder.joinpath(flexion, "data", "dataset")
        quality_labels = self.load_quality_labels(dataset_folder)

        camera_data = defaultdict(dict)
        for camera in self.cameras:
            roi_labels = self.load_roi_labels(dataset_folder, camera)
            for image_number, quality_label in quality_labels.items():
                roi_label = roi_labels[image_number]

                modality_data = self.load_modalities(
                    dataset_folder,
                    patient_name,
                    side,
                    flexion,
                    camera,
                    image_number,
                    quality_label,
                    roi_label,
                )
                for element in modality_data:
                    if isinstance(element, DataElement):
                        timestamp = element.timestamp
                    elif isinstance(element, Tuple) and all(
                        map(lambda e: isinstance(e, DataElement), element)
                    ):
                        timestamps = set(map(lambda e: e.timestamp, element))
                        assert len(timestamps) == 1
                        timestamp = timestamps.pop()
                    else:
                        raise RuntimeError

                    camera_data[camera][
                        (patient_name, side, flexion, "orig", image_number, timestamp)
                    ] = element

        data = []
        camera_keys = set(map(lambda d: tuple(set(d.keys())), camera_data.values()))
        # Keys are identical over all cameras
        assert len(camera_keys) == 1

        # Iter over keys, don't use the set since its order is arbitrary
        for key in next(iter(camera_data.values())).keys():
            data.append(tuple(map(lambda camera: camera_data[camera][key], self.cameras)))
        return data

    def get_data_element(self, data_element: Tuple[DataElement, DataElement]):
        first_cam_image = data_element[0]
        additional_targets = {element.camera: element.image for element in data_element[1:]}
        transformed_images = self.transform(image=data_element[0].image, **additional_targets)

        data: Dict[str, Any] = {
            f"input_{camera}": np.expand_dims(transformed_images[camera] - 0.5, 0).astype(np.float32)
            for camera in self.cameras[1:]
        }
        data[f"input_{self.cameras[0]}"] = np.expand_dims(transformed_images["image"] - 0.5, 0).astype(
            np.float32
        )

        data.update(
            {
                "target": np.array(first_cam_image.quality).astype(np.float32),
                "patient_name": first_cam_image.patient_name,
                "image_number": first_cam_image.image_number,
                "side": first_cam_image.side,
                "flexion": first_cam_image.flexion,
                "camera": "all",
                "timestamp": first_cam_image.timestamp,
            }
        )

        return data

    def _log(self, state: State) -> None:
        dataset_name = state.dataset_state.name
        data = state.dataset_state.data
        if data is None or not any("input" in key for key in data):
            return

        images_cam_right = data["input_cam_right"][:2].detach().cpu()
        images_cam_left = data["input_cam_left"][:2].detach().cpu()

        images = torch.cat((images_cam_right, images_cam_left), 0)

        # Undo the normalization
        images = (images + 0.5) * 255.0

        # Order in a grid
        grid_image = make_grid(images, 2, normalize=False, scale_each=False)

        # Get a [0, 255] uin8 image/label
        log_image = (grid_image.numpy()).astype(np.uint8)

        self.logger.info(
            ImageLogMessage(tag=f"{dataset_name}", step=state.looper_state.step_iteration, image=log_image)
        )


class DepthImageCoupleDatasetIter(DepthImageCoupleDataset, DepthImageDatasetIter):
    pass


@yaloader.loads(DepthImageDataset)
class DepthImageDatasetConfig(DatasetConfig):
    name: str = "Depth Image Dataset"
    base_folder_path: Path
    dataset_type: Literal["train", "test", "prediction"] = "train"
    test_patients: Optional[List[str]] = None
    cameras: Optional[List[CAMERA_TYPE]] = None
    cache_data: bool = False

    images_per_xray: int = 1
    image_size: int = 336

    depth_threshold: int = 1000


@yaloader.loads(DepthImageDatasetIter)
class DepthImageDatasetIterConfig(DepthImageDatasetConfig):
    nr_buckets: int = 1


@yaloader.loads(DepthImageCoupleDataset)
class DepthImageCoupleDatasetConfig(DepthImageDatasetConfig):
    name: str = "Depth Image Couple Dataset"


@yaloader.loads(DepthImageCoupleDatasetIter)
class DepthImageCoupleDatasetIterConfig(DepthImageCoupleDatasetConfig, DepthImageDatasetIterConfig):
    pass
