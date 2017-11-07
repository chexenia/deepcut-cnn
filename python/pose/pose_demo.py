#!/usr/bin/env python
"""
Pose predictions in Python.

Caffe must be available on the Pythonpath for this to work. The methods can
be imported and used directly, or the command line interface can be used. In
the latter case, adjust the log-level to your needs. The maximum image size
for one prediction can be adjusted with the variable _MAX_SIZE so that it
still fits in GPU memory, all larger images are split in sufficiently small
parts.

Authors: Christoph Lassner, based on the MATLAB implementation by Eldar
  Insafutdinov.
"""
# pylint: disable=invalid-name
import os as _os
import logging as _logging
import numpy as _np
import scipy as _scipy
import click as _click
import caffe as _caffe

from estimate_pose import estimate_pose

_LOGGER = _logging.getLogger(__name__)


def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    ny, nx = image.shape[:2]
    y_radius_0 = radius if cy - radius >= 0 else cy
    y_radius_1 = radius if cy + radius <= ny else ny - cy
    x_radius_0 = radius if cx - radius >= 0 else cx
    x_radius_1 = radius if cx + radius <= nx else nx - cx
    y, x = _np.ogrid[-y_radius_1:y_radius_1, -x_radius_0:x_radius_1]
    index = x**2 + y**2 <= radius**2
    cy0 = cy - radius
    cy1 = cy + radius
    cx0 = cx - radius
    cx1 = cx + radius
    image[cy0:cy1, cx0:cx1][index] = (
        image[cy0:cy1, cx0:cx1][index].astype('float32') * transparency
        + _np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')


###############################################################################
# Command line interface.
###############################################################################

@_click.command()
@_click.argument('image_names',
                 type=_click.Path(exists=True, readable=True),
                 nargs=-1)
@_click.option('--out_name',
               type=_click.Path(dir_okay=True, writable=True),
               help='The result location to use. By default, use `image_name`_pose.npz.',
               default=None)
@_click.option('--scales',
               type=_click.STRING,
               help=('The scales to use, comma-separated. The most confident '
                     'will be stored. Default: 1.'),
               default='1.')
@_click.option('--visualize',
               type=_click.BOOL,
               help='Whether to create a visualization of the pose. Default: True.',
               default=True)
@_click.option('--folder_image_suffix',
               type=_click.STRING,
               help=('The ending to use for the images to read, if a folder is '
                     'specified. Default: .png.'),
               default='.png')
@_click.option('--use_cpu',
               type=_click.BOOL,
               is_flag=True,
               help='Use CPU instead of GPU for predictions.',
               default=False)
@_click.option('--gpu',
               type=_click.INT,
               help='GPU device id.',
               default=0)
def predict_pose_from(image_names,
                      out_name=None,
                      scales='1.',
                      visualize=True,
                      folder_image_suffix='.png',
                      use_cpu=False,
                      gpu=0):
    """
    Load an image file, predict the pose and write it out.

    `IMAGE_NAME` may be an image or a directory, for which all images with
    `folder_image_suffix` will be processed.
    """
    script_dir = _os.path.dirname(_os.path.realpath(__file__))
    model_def = _os.path.join(script_dir,
            '../../models/deepercut/ResNet-152.prototxt')
    model_bin = _os.path.join(script_dir,
            '../../models/deepercut/ResNet-152.caffemodel')
    scales = [float(val) for val in scales.split(',')]
    images = image_names
    process_folder = False
    if use_cpu:
        _caffe.set_mode_cpu()
    else:
        _caffe.set_mode_gpu()
        _caffe.set_device(gpu)
    out_name_provided = out_name
    if process_folder and out_name is not None and not _os.path.exists(out_name):
        _os.mkdir(out_name)
    for image_name in images:
        if out_name_provided is None:
            out_name = image_name + '_pose.npz'
        elif process_folder:
            out_name = _os.path.join(out_name_provided,
                                     _os.path.basename(image_name) + '_pose.npz')
        _LOGGER.info("Predicting the pose on `%s` (saving to `%s`) in best of "
                     "scales %s.", image_name, out_name, scales)
        image = _scipy.misc.imread(image_name)
        if image.ndim == 2:
            _LOGGER.warn("The image is grayscale! This may deteriorate performance!")
            image = _np.dstack((image, image, image))
        else:
            image = image[:, :, ::-1]
            if image.shape[2] > 3:
                image = image[:, :, :3]
        pose = estimate_pose(image, model_def, model_bin, scales)
        _np.savez_compressed(out_name, pose=pose)
        if visualize:
            visim = image[:, :, ::-1].copy()
            colors = [[255, 0, 0],[0, 255, 0],[0, 0, 255],[0,245,255],[255,131,250],[255,255,0],
                      [128, 0, 0],[0, 128, 0],[0, 0, 128],[0,122,128],[128,67,125],[128,128,0],
                      [0,0,0],[255,255,255]]
            for p_idx in range(14):
                _npcircle(visim,
                          pose[0, p_idx],
                          pose[1, p_idx],
                          8,
                          colors[p_idx],
                          0.0)
            vis_name = out_name + '_vis.png'
            _scipy.misc.imsave(vis_name, visim)


if __name__ == '__main__':
    _logging.basicConfig(level=_logging.INFO)
    # pylint: disable=no-value-for-parameter
    predict_pose_from()
