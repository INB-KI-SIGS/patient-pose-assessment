import albumentations
import cv2
import numpy as np


class AddInvalidPoints(albumentations.ImageOnlyTransform):
    def __init__(
        self,
        top_gradient_percentage: float = 0.7,
        min_selected_indices: float = 0.1,
        max_selected_indices: float = 0.5,
        always_apply=False,
        p=0.5,
    ):
        super(AddInvalidPoints, self).__init__(always_apply, p)
        self.top_gradient_percentage = top_gradient_percentage
        self.min_selected_indices = min_selected_indices
        self.max_selected_indices = max_selected_indices

    def apply(self, image, **params):
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.maximum(np.abs(grad_x), np.abs(grad_y))

        top_gradient_indices = np.argwhere(grad > np.quantile(grad, 0.998))
        min_indices = int(top_gradient_indices.shape[0] * self.min_selected_indices)
        max_indices = int(top_gradient_indices.shape[0] * self.max_selected_indices)

        # min and max are both
        if min_indices == max_indices:
            min_indices = 0
            max_indices = min_indices + 1

        selected_indices = np.random.choice(
            np.arange(top_gradient_indices.shape[0]),
            size=np.random.randint(min_indices, max_indices),
        )
        selected_gradient_indices = top_gradient_indices[selected_indices]

        image[selected_gradient_indices[:, 0], selected_gradient_indices[:, 1]] = 0

        return image

    def get_transform_init_args_names(self):
        return "top_gradient_percentage", "min_selected_indices"
