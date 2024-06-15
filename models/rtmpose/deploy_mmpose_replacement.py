import math
from collections import abc
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pseudo_collate(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each data_itement in ``data_batch``.

    The default behavior of dataloader is to merge a list of samples to form
    a mini-batch of Tensor(s). However, in MMEngine, ``pseudo_collate``
    will not stack tensors to batch tensors, and convert int, float, ndarray to
    tensors.

    This code is referenced from:
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.

    Args:
        data_batch (Sequence): Batch of data from dataloader.

    Returns:
        Any: Transversed Data in the same format as the data_itement of
        ``data_batch``.
    """
    data_item = data_batch[0]
    data_item_type = type(data_item)
    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named tuple
        return data_item_type(*(pseudo_collate(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [pseudo_collate(samples)
                    for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type(
                    [pseudo_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [pseudo_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({
            key: pseudo_collate([d[key] for d in data_batch])
            for key in data_item
        })
    else:
        return data_batch


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Union[Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            # if isinstance(transform, dict):
            #     transform = TRANSFORMS.build(transform)
            #     if not callable(transform):
            #         raise TypeError(f'transform should be a callable object, '
            #                         f'but got {type(transform)}')
            #     self.transforms.append(transform)
            if callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

class LoadImage():
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self
                #  to_float32: bool = False,
                #  color_type: str = 'color',
                #  imdecode_backend: str = 'cv2',
                #  file_client_args: Optional[dict] = None,
                #  ignore_empty: bool = False,
                #  *,
                #  backend_args: Optional[dict] = None
                ) -> None:
        pass
        # self.ignore_empty = ignore_empty
        # self.to_float32 = to_float32
        # self.color_type = color_type
        # self.imdecode_backend = imdecode_backend

        # self.file_client_args: Optional[dict] = None
        # self.backend_args: Optional[dict] = None


    def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    @staticmethod
    def get(filepath: Union[str, Path]) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.get(filepath)
            b'hello world'
        """
        with open(filepath, 'rb') as f:
            value = f.read()
        return value

    @staticmethod
    def imfrombytes(content: bytes):

        img_np = np.frombuffer(content, np.uint8)
        flag = cv2.IMREAD_COLOR
        img = cv2.imdecode(img_np, flag)
        # if flag == IMREAD_COLOR and channel_order == 'rgb':
        #     cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img


    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if "img" not in results:
            filename = results['img_path']



            img_bytes = self.get(filename)

            # img = mmcv.imfrombytes(
            #     img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            img = self.imfrombytes(img_bytes)

            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
        else:
            img = results['img']
            assert isinstance(img, np.ndarray)
            # if self.to_float32:
            #     img = img.astype(np.float32)

            if 'img_path' not in results:
                results['img_path'] = None
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
        return results


class GetBBoxCenterScale():
    """Convert bboxes from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required Keys:

        - bbox

    Added Keys:

        - bbox_center
        - bbox_scale

    Args:
        padding (float): The bbox padding scale that will be multilied to
            `bbox_scale`. Defaults to 1.25
    """

    def __init__(self, padding: float = 1.25) -> None:
        # super().__init__()

        self.padding = padding

    def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    @staticmethod
    def bbox_xyxy2cs(bbox: np.ndarray,
                    padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
        """Transform the bbox format from (x,y,w,h) into (center, scale)

        Args:
            bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
                as (left, top, right, bottom)
            padding (float): BBox padding factor that will be multilied to scale.
                Default: 1.0

        Returns:
            tuple: A tuple containing center and scale.
            - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
                (n, 2)
            - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
                (n, 2)
        """
        # convert single bbox from (4, ) to (1, 4)
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]

        scale = (bbox[..., 2:] - bbox[..., :2]) * padding
        center = (bbox[..., 2:] + bbox[..., :2]) * 0.5

        if dim == 1:
            center = center[0]
            scale = scale[0]

        return center, scale


    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GetBBoxCenterScale`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        bbox = results['bbox']
        center, scale = self.bbox_xyxy2cs(bbox, padding=self.padding)

        results['bbox_center'] = center
        results['bbox_scale'] = scale

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(padding={self.padding})'
        return repr_str

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """

    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt

def _get_3rd_point(a: np.ndarray, b: np.ndarray):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c

def get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Calculate the affine transformation matrix under the unbiased
    constraint. See `UDP (CVPR 2020)`_ for details.

    Note:

        - The bbox number: N

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (tuple): Size ([w, h]) of the output image

    Returns:
        np.ndarray: A 2x3 transformation matrix

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (-0.5 * input_size[0] * math.cos(rot_rad) +
                                0.5 * input_size[1] * math.sin(rot_rad) +
                                0.5 * scale[0])
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (-0.5 * input_size[0] * math.sin(rot_rad) -
                                0.5 * input_size[1] * math.cos(rot_rad) +
                                0.5 * scale[1])
    return warp_mat

def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0., 0.),
    inv: bool = False,
    fix_aspect_ratio: bool = True,
) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
        fix_aspect_ratio (bool): Whether to fix aspect ratio during transform.
            Defaults to True.

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    shift = np.array(shift)
    src_w, src_h = scale[:2]
    dst_w, dst_h = output_size[:2]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.]), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    if fix_aspect_ratio:
        src[2, :] = _get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    else:
        src_dir_2 = _rotate_point(np.array([0., src_h * -0.5]), rot_rad)
        dst_dir_2 = np.array([0., dst_h * -0.5])
        src[2, :] = center + src_dir_2 + scale * shift
        dst[2, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir_2

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat

class TopdownAffine():
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        # super().__init__()

        self.input_size = input_size
        self.use_udp = use_udp

    def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            if results.get('transformed_keypoints', None) is not None:
                transformed_keypoints = results['transformed_keypoints'].copy()
            else:
                transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
                results['keypoints'][..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints

        results['input_size'] = (w, h)
        results['input_center'] = center
        results['input_scale'] = scale

        return results

def image_to_tensor(img: Union[np.ndarray,
                               Sequence[np.ndarray]]) -> torch.torch.Tensor:
    """Translate image or sequence of images to tensor. Multiple image tensors
    will be stacked.

    Args:
        value (np.ndarray | Sequence[np.ndarray]): The original image or
            image sequence

    Returns:
        torch.Tensor: The output tensor.
    """

    if isinstance(img, np.ndarray):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)

        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    else:
        assert is_seq_of(img, np.ndarray)
        tensor = torch.stack([image_to_tensor(_img) for _img in img])

    return tensor

def keypoints_to_tensor(keypoints: Union[np.ndarray, Sequence[np.ndarray]]
                        ) -> torch.torch.Tensor:
    """Translate keypoints or sequence of keypoints to tensor. Multiple
    keypoints tensors will be stacked.

    Args:
        keypoints (np.ndarray | Sequence[np.ndarray]): The keypoints or
            keypoints sequence.

    Returns:
        torch.Tensor: The output tensor.
    """
    if isinstance(keypoints, np.ndarray):
        keypoints = np.ascontiguousarray(keypoints)
        tensor = torch.from_numpy(keypoints).contiguous()
    else:
        assert is_seq_of(keypoints, np.ndarray)
        tensor = torch.stack(
            [keypoints_to_tensor(_keypoints) for _keypoints in keypoints])

    return tensor

class PackPoseInputs():
    """Pack the inputs data for pose estimation.

    The ``img_meta`` item is always populated. The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default it includes:

        - ``id``: id of the data sample

        - ``img_id``: id of the image

        - ``'category_id'``: the id of the instance category

        - ``img_path``: path to the image file

        - ``crowd_index`` (optional): measure the crowding level of an image,
            defined in CrowdPose dataset

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``input_size``: the input size to the network

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

        - ``flip_indices``: the indices of each keypoint's symmetric keypoint

        - ``raw_ann_info`` (optional): raw annotation of the instance(s)

    Args:
        meta_keys (Sequence[str], optional): Meta keys which will be stored in
            :obj: `PoseDataSample` as meta info. Defaults to ``('id',
            'img_id', 'img_path', 'category_id', 'crowd_index, 'ori_shape',
            'img_shape', 'input_size', 'input_center', 'input_scale', 'flip',
            'flip_direction', 'flip_indices', 'raw_ann_info')``
    """

    # items in `instance_mapping_table` will be directly packed into
    # PoseDataSample.gt_instances without converting to Tensor
    instance_mapping_table = dict(
        bbox='bboxes',
        bbox_score='bbox_scores',
        keypoints='keypoints',
        keypoints_cam='keypoints_cam',
        keypoints_visible='keypoints_visible',
        # In CocoMetric, the area of predicted instances will be calculated
        # using gt_instances.bbox_scales. To unsure correspondence with
        # previous version, this key is preserved here.
        bbox_scale='bbox_scales',
        # `head_size` is used for computing MpiiPCKAccuracy metric,
        # namely, PCKh
        head_size='head_size',
    )

    # items in `field_mapping_table` will be packed into
    # PoseDataSample.gt_fields and converted to Tensor. These items will be
    # used for computing losses
    field_mapping_table = dict(
        heatmaps='heatmaps',
        instance_heatmaps='instance_heatmaps',
        heatmap_mask='heatmap_mask',
        heatmap_weights='heatmap_weights',
        displacements='displacements',
        displacement_weights='displacement_weights')

    # items in `label_mapping_table` will be packed into
    # PoseDataSample.gt_instance_labels and converted to Tensor. These items
    # will be used for computing losses
    label_mapping_table = dict(
        keypoint_labels='keypoint_labels',
        keypoint_weights='keypoint_weights',
        keypoints_visible_weights='keypoints_visible_weights')

    def __init__(self,
                 meta_keys=('id', 'img_id', 'img_path', 'category_id',
                            'crowd_index', 'ori_shape', 'img_shape',
                            'input_size', 'input_center', 'input_scale',
                            'flip', 'flip_direction', 'flip_indices',
                            'raw_ann_info', 'dataset_name'),
                 pack_transformed=False):
        self.meta_keys = meta_keys
        self.pack_transformed = pack_transformed

    def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_samples' (obj:`PoseDataSample`): The annotation info of the
                sample.
        """
        # Pack image(s) for 2d pose estimation
        if 'img' in results:
            img = results['img']
            inputs_tensor = image_to_tensor(img)
        # Pack keypoints for 3d pose-lifting
        elif 'lifting_target' in results and 'keypoints' in results:
            if 'keypoint_labels' in results:
                keypoints = results['keypoint_labels']
            else:
                keypoints = results['keypoints']
            inputs_tensor = keypoints_to_tensor(keypoints)

        # data_sample = PoseDataSample()
        # data_sample = DataSample(metainfo=dict())
        data_sample = {}

        # pack instance data
        # gt_instances = InstanceData()
        # _instance_mapping_table = results.get('instance_mapping_table',
        #                                       self.instance_mapping_table)
        # for key, packed_key in _instance_mapping_table.items():
        #     if key in results:
        #         gt_instances.set_field(results[key], packed_key)

        # pack `transformed_keypoints` for visualizing data transform
        # and augmentation results
        # if self.pack_transformed and 'transformed_keypoints' in results:
        #     gt_instances.set_field(results['transformed_keypoints'],
        #                            'transformed_keypoints')

        # data_sample.gt_instances = gt_instances

        # pack instance labels
        # gt_instance_labels = InstanceData()
        # _label_mapping_table = results.get('label_mapping_table',
        #                                    self.label_mapping_table)
        # for key, packed_key in _label_mapping_table.items():
        #     if key in results:
        #         if isinstance(results[key], list):
        #             # A list of labels is usually generated by combined
        #             # multiple encoders (See ``GenerateTarget`` in
        #             # mmpose/datasets/transforms/common_transforms.py)
        #             # In this case, labels in list should have the same
        #             # shape and will be stacked.
        #             _labels = np.stack(results[key])
        #             gt_instance_labels.set_field(_labels, packed_key)
        #         else:
        #             gt_instance_labels.set_field(results[key], packed_key)
        # data_sample.gt_instance_labels = gt_instance_labels.to_tensor()

        # pack fields
        # gt_fields = None
        # _field_mapping_table = results.get('field_mapping_table',
        #                                    self.field_mapping_table)
        # for key, packed_key in _field_mapping_table.items():
        #     if key in results:
        #         if isinstance(results[key], list):
        #             if gt_fields is None:
        #                 gt_fields = MultilevelPixelData()
        #             else:
        #                 assert isinstance(
        #                     gt_fields, MultilevelPixelData
        #                 ), 'Got mixed single-level and multi-level pixel data.'
        #         else:
        #             if gt_fields is None:
        #                 gt_fields = PixelData()
        #             else:
        #                 assert isinstance(
        #                     gt_fields, PixelData
        #                 ), 'Got mixed single-level and multi-level pixel data.'

        #         gt_fields.set_field(results[key], packed_key)

        # if gt_fields:
        #     data_sample.gt_fields = gt_fields.to_tensor()

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        # data_sample.set_metainfo(img_meta)
        data_sample['metainfo'] = img_meta

        packed_results = dict()
        packed_results['inputs'] = inputs_tensor
        packed_results['data_samples'] = data_sample

        return packed_results

def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type

    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(
        tensor_list,
        list), (f'Expected input type to be list, but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({
        tensor.ndim
        for tensor in tensor_list
    }) == 1, (f'Expected the dimensions of all tensors must be the same, '
              f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(
        torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)

class PoseDataPreprocessor(nn.Module):
    """Image pre-processor for pose estimation tasks.

    Comparing with the :class:`ImgDataPreprocessor`,

    1. It will additionally append batch_input_shape
    to data_samples considering the DETR-based pose estimation tasks.

    2. Support image augmentation transforms on batched data.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Apply batch augmentation transforms.

    Args:
        mean (sequence of float, optional): The pixel mean of R, G, B
            channels. Defaults to None.
        std (sequence of float, optional): The pixel standard deviation
            of R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to BGR.
            Defaults to False.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
        batch_augments: (list of dict, optional): Configs of augmentation
            transforms on batched data. Defaults to None.
    """

    def __init__(self,
                 mean: Sequence[float] = None,
                 std: Sequence[float] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None
                 ):
        super().__init__()
        self._non_blocking = non_blocking
        self._device = torch.device('cpu')
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, (
                '`mean` should have 1 or 3 values, to be compatible with '
                f'RGB or gray image, but got {len(mean)} values')
            assert len(std) == 3 or len(std) == 1, (
                '`std` should have 1 or 3 values, to be compatible with RGB '
                f'or gray image, but got {len(std)} values')
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        # if batch_augments is not None:
        #     self.batch_augments = nn.ModuleList(
        #         [MODELS.build(aug) for aug in batch_augments])
        # else:
        #     self.batch_augments = None
        self.batch_augments = None

    @property
    def device(self):
        return self._device

    def cast_data(self, data):
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, '_fields'):
            # namedtuple
            return type(data)(*(self.cast_data(sample) for sample in data))
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)
        elif isinstance(data, (torch.Tensor)): # BaseDataElement
            return data.to(self.device, non_blocking=self._non_blocking)
        else:
            return data

    def _forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Performs normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataset. If the collate
                function of DataLoader is :obj:`pseudo_collate`, data will be a
                list of dict. If collate function is :obj:`default_collate`,
                data will be a tuple with batch input tensor and list of data
                samples.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs = _batch_inputs.float()
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = self._forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        # update metainfo since the image shape might change
        # batch_input_shape = tuple(inputs[0].size()[-2:])
        # for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
        #     data_sample.set_metainfo({
        #         'batch_input_shape': batch_input_shape,
        #         'pad_shape': pad_shape
        #     })

        # apply batch augmentations
        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape

# from mmpose.codecs.utils import get_simcc_maximum
def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray,
                      apply_softmax: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        apply_softmax (bool): whether to apply softmax on the heatmap.
            Defaults to False.

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """

    assert isinstance(simcc_x, np.ndarray), ('simcc_x should be numpy.ndarray')
    assert isinstance(simcc_y, np.ndarray), ('simcc_y should be numpy.ndarray')
    assert simcc_x.ndim == 2 or simcc_x.ndim == 3, (
        f'Invalid shape {simcc_x.shape}')
    assert simcc_y.ndim == 2 or simcc_y.ndim == 3, (
        f'Invalid shape {simcc_y.shape}')
    assert simcc_x.ndim == simcc_y.ndim, (
        f'{simcc_x.shape} != {simcc_y.shape}')

    if simcc_x.ndim == 3:
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
    else:
        N = None

    if apply_softmax:
        simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
        simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
        ex, ey = np.exp(simcc_x), np.exp(simcc_y)
        simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
        simcc_y = ey / np.sum(ey, axis=1, keepdims=True)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    if N:
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

    return locs, vals
