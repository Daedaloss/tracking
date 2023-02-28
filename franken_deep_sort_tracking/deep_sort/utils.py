import numpy as np 
from skimage.transform import rescale
from scipy.ndimage import center_of_mass
import skvideo
import skimage
skvideo.setFFmpegPath("/usr/local/bin/")
import skvideo.io
import random
import matplotlib as mpl
import tqdm
from colormap import hex2rgb
from distutils.version import LooseVersion
from joblib import Parallel, delayed
import cv2



def rescale_img(mask, frame, mask_size=224):
    rectsize = [mask[3] - mask[1], mask[2] - mask[0]]

    rectsize = np.asarray(rectsize)
    scale = mask_size / rectsize.max()

    cutout = frame[mask[0] : mask[0] + rectsize[1], mask[1] : mask[1] + rectsize[0], :]

    img_help = rescale(cutout, scale, multichannel=True)
    padded_img = np.zeros((mask_size, mask_size, 3))

    padded_img[
        int(mask_size / 2 - img_help.shape[0] / 2) : int(
            mask_size / 2 + img_help.shape[0] / 2
        ),
        int(mask_size / 2 - img_help.shape[1] / 2) : int(
            mask_size / 2 + img_help.shape[1] / 2
        ),
        :,
    ] = img_help

    return padded_img

#%%

def masks_to_coms(masks):
    # calculate center of masses
    coms = []
    for idx in range(0, masks.shape[-1]):
        mask = masks[:, :, idx]
        com = center_of_mass(mask.astype("int"))
        coms.append(com)
    coms = np.asarray(coms)
    
    return coms

#%%

def loadVideo(path, num_frames=None, greyscale=True):
    # load the video
    if not num_frames is None:
        return skvideo.io.vread(path, as_grey=greyscale, num_frames=num_frames)
    else:
        return skvideo.io.vread(path, as_grey=greyscale)
    
#%%

# adapted from maskrcnn
def resize(
    image,
    output_shape,
    order=1,
    mode="constant",
    cval=0,
    clip=True,
    preserve_range=False,
    anti_aliasing=False,
    anti_aliasing_sigma=None,
):
    """A wrapper for Scikit-Image resize().
    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image,
            output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma,
        )
    else:
        return skimage.transform.resize(
            image,
            output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
        )

#%%
def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y: y + min_dim, x: x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop

#%%
def sample_colormap(map_name, max_ids):
    
    cmap = mpl.cm.get_cmap(map_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_ids)
    rgba = cmap(norm(range(max_ids)))
    np.random.shuffle(rgba)
    
    hex_cols = [ mpl.colors.to_hex(k, keep_alpha=False) for k in rgba ]
    rgb_cols = [ hex2rgb(k) for k in hex_cols ]
    
    return rgb_cols

#%%

def mold_image(img, config=None, dimension=None, min_dimension=None, return_all=False):
    """
    Args:
        img:
        config:
        dimension:
    """
    if config:
        image, window, scale, padding, crop = resize_image(
            img[:, :, :],
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE,
        )
    elif dimension:
        if min_dimension:
            image, window, scale, padding, crop = resize_image(
                img[:, :, :], min_dim=min_dimension, max_dim=dimension, mode="pad64",
            )
        else:
            image, window, scale, padding, crop = resize_image(
                img[:,:,:], min_dim=dimension, max_dim=dimension, mode="square",
            )
    else:
        return NotImplementedError
    if return_all:
        return image, window, scale, padding, crop
    else:
        return image


def mold_video(video, dimension, n_jobs=40):
    """
    Args:
        video:
        dimension:
        n_jobs:
    """
    results = Parallel(
        n_jobs=n_jobs, max_nbytes=None, backend="multiprocessing", verbose=40
    )(delayed(mold_image)(image, dimension=dimension) for image in video)
    return results

#%%

def displayBoxes(frame, mask, color=(0, 0, 255), animal_id=None, mask_id=None):
    #mask_color_labeled = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1

    print(mask)
    print(color)
    cv2.rectangle(frame, (mask[1], mask[0]), (mask[3], mask[2]), color, 3)

    if animal_id:
        cv2.putText(
            frame,
            str(animal_id),
            (mask[1], mask[0]),
            font,
            0.7,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

    return frame



#%%
def serialize_tracks(results, max_ids=10):
    
    single_boxes   = {key: [] for key in range(max_ids)}
    single_indices = {key: [] for key in range(max_ids)}

    for idx, result in tqdm(enumerate(results)):
        
      ids = results[idx]['track_ids'] 
      boxes = results[idx]['boxes']
      overlaps = results[idx]['overlaps']

      for i in range(len(boxes)): #go through box one by one
        if i >= max_ids:
          print('OUTSIDE')
          continue
        try:
          mask = boxes[i]
        except IndexError:
          print('OUTSIDE3!')
          continue

        if i in ids: # this give some problems
          try:
            overl=[]
            for el in results[idx-3 : idx+3]: #this checks whether id is present for about 6 frames (can change this)
              overl.append(el['overlaps'][i])
          except IndexError:
            pass

          condition = np.array(overl) > 0
          condition = condition.all()

          if condition:
            try:
              # get rid of black frames (not sure whether this behavior is intended)
                single_indices[i].append(idx)
                single_boxes[i].append(mask)

            except (TypeError, IndexError, KeyError):
              print(idx)
              continue

          else:
            print(idx)
        else:
          print(idx)

    return single_indices, single_boxes

#