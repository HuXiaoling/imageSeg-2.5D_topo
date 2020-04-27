import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from random import randint


def flip(image, label):
    """
    Args:
        image : numpy array of image
        option_value = random integer between 0 to 3
    Return :
        image : numpy array of flipped image
    """
    for i, (img, lab) in enumerate(zip(image, label)):
        flip_num = randint(0, 3)
        if flip_num == 0:
            # vertical
            image[i] = np.flip(img, flip_num)
            label[i] = np.flip(lab, flip_num)
        elif flip_num == 1:
            # horizontal
            image[i] = np.flip(img, flip_num)
            label[i] = np.flip(lab, flip_num)
        elif flip_num == 2:
            # horizontally and vertically flip
            image[i] = np.flip(img, 0)
            image[i] = np.flip(img, 1)
            label[i] = np.flip(lab, 0)
            label[i] = np.flip(lab, 1)
        else:
            image[i] = img
            label[i] = lab
            # no effect
    return image, label


def add_gaussian_noise(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    for i, img in enumerate(image):
        gaus_noise = np.random.normal(mean, std, img.shape)
        img = img.astype("int16")
        image[i] = img + gaus_noise
        # image[i] = ceil_floor_image(image)
    return image


def add_uniform_noise(image, low=-10, high=10):
    """
    Args:
        image : numpy array of image
        low : lower boundary of output interval
        high : upper boundary of output interval
    Return :
        image : numpy array of image with uniform noise added
    """
    for i, img in enumerate(image):
        uni_noise = np.random.uniform(low, high, img.shape)
        img = img.astype("int16")
        image[i] = img + uni_noise
        # image = ceil_floor_image(image)
    return image


def change_brightness(image, value):
    for i, img in enumerate(image):
        img = img.astype("int16")
        img = img + value
        image[i] = ceil_floor_image(img)
    return image


def add_elastic_transform(image, alpha, sigma, pad_size=30, seed=None):
    image_size = int(image.shape[0])
    image = np.pad(image, pad_size, mode="symmetric")
    if seed is None:
        seed = randint(1, 100)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(seed)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return cropping(map_coordinates(image, indices, order=1).reshape(shape), 512, pad_size, pad_size), seed


def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image


def cropping(image, crop_size, dim1, dim2):
    cropped_img = image[dim1:dim1 + crop_size, dim2:dim2 + crop_size]
    return cropped_img


def multi_cropping(image, crop_size, crop_num1, crop_num2):
    img_height, img_width = image.shape[0], image.shape[1]
    assert crop_size * crop_num1 >= img_width and crop_size * \
           crop_num2 >= img_height, "Whole image cannot be sufficiently expressed"
    assert crop_num1 <= img_width - crop_size + 1 and crop_num2 <= img_height - \
           crop_size + 1, "Too many number of crops"

    cropped_imgs = []
    # int((img_height - crop_size)/(crop_num1 - 1))
    dim1_stride = stride_size(img_height, crop_num1, crop_size)
    # int((img_width - crop_size)/(crop_num2 - 1))
    dim2_stride = stride_size(img_width, crop_num2, crop_size)
    for i in range(crop_num1):
        for j in range(crop_num2):
            cropped_imgs.append(cropping(image, crop_size,
                                         dim1_stride * i, dim2_stride * j))
    return np.asarray(cropped_imgs)


def approximate_image(image):
    image[image > 127.5] = 255
    image[image < 127.5] = 0
    image = image.astype("uint8")
    return image


def normalization(image, mean, std):
    image = image / 255
    image = (image - mean) / std
    return image


def normalization2(image, max, min):
    for i, img in enumerate(image):
        image[i] = (img - np.min(img)) * (max - min) / (np.max(img) - np.min(img)) + min
    return image


def stride_size(image_len, crop_num, crop_size):
    """return stride size
    Args :
        image_len(int) : length of one size of image (width or height)
        crop_num(int) : number of crop in certain direction
        crop_size(int) : size of crop
    Return :
        stride_size(int) : stride size
    """
    return int((image_len - crop_size) / (crop_num - 1))
