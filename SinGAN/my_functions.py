"""
New code for replace_background feature
"""

import cv2
import numpy as np
from skimage import filters
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.restoration import inpaint
from sklearn.cluster import KMeans


def create_mask(img: "np.ndarray") -> "np.ndarray":
    grayscale_img = rgb2gray(img)
    val = filters.threshold_otsu(grayscale_img)
    mask = grayscale_img < val
    return mask


def create_background(img: "np.ndarray") -> "np.ndarray":
    mask = create_mask(img)

    kernel = np.ones((8, 8), np.uint8)
    mask_dilation = cv2.dilate(img_as_ubyte(mask), kernel, iterations=2)

    # Defect image over the same region in each color channel
    image_defect = img.copy()
    for layer in range(image_defect.shape[-1]):
        image_defect[np.where(mask_dilation)] = 0

    image_result = inpaint.inpaint_biharmonic(image_defect, mask_dilation, multichannel=True)
    return image_result


def create_img_over_background(img: "np.ndarray", background_img: "np.ndarray") -> "np.ndarray":
    # Now create a mask and create its inverse mask also
    mask = create_mask(img)
    mask_inv = np.bitwise_not(mask)

    mask_3channels = np.repeat(mask[:, :, np.newaxis], img.shape[-1], axis=2)
    mask_inv_3channels = np.repeat(mask_inv[:, :, np.newaxis], background_img.shape[-1], axis=2)

    # Now black-out the area of img in the background
    bg = np.bitwise_and(img_as_ubyte(background_img), img_as_ubyte(mask_inv_3channels))

    # Remove everything but the foreground in the image
    img_fg = np.bitwise_and(img_as_ubyte(img), img_as_ubyte(mask_3channels))

    # Combine the images
    return np.bitwise_or(bg, img_fg)


def create_clustered_image(img: "np.ndarray", n_clusters: int) -> "np.ndarray":
    image_2d = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    # Use KMeans clustering algorithm from sklearn.cluster to cluster pixels in image
    # tweak the cluster size and see what happens to the Output
    kmeans = KMeans(n_clusters).fit(image_2d)
    clustered_img_2d = kmeans.cluster_centers_[kmeans.labels_]
    # Reshape back the image from 2D to 3D image
    clustered_img_3d = clustered_img_2d.reshape(img.shape[0], img.shape[1], img.shape[2])
    return clustered_img_3d
