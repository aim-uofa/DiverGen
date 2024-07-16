# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import json
import numpy as np
from torch.nn import functional as F
import logging
import os
from fvcore.common.timer import Timer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
#detectron2.data.datasets
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta
from detectron2.data.datasets.lvis_v0_5_categories import LVIS_CATEGORIES as LVIS_V0_5_CATEGORIES
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
from tqdm import tqdm
logger = logging.getLogger(__name__)

__all__ = ["load_lvis_json", "register_lvis_instances", "get_lvis_instances_meta"]
def load_class_freq(
    path='datasets/metadata/lvis_v1_train_cat_info.json', freq_weight=1.0):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight


def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared



def reset_cls_test(model, cls_path, num_classes):
    model.roi_heads.num_classes = num_classes
    if type(cls_path) == str:
        print('Resetting zs_weight', cls_path)
        zs_weight = torch.tensor(
            np.load(cls_path), 
            dtype=torch.float32).permute(1, 0).contiguous() # D x C
    else:
        zs_weight = cls_path
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))], 
        dim=1) # D x (C + 1)
    if model.roi_heads.box_predictor[0].cls_score.norm_weight:
        zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(model.device)
    for k in range(len(model.roi_heads.box_predictor)):
        del model.roi_heads.box_predictor[k].cls_score.zs_weight
        model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight

def register_lvis_instances_with_id(name, metadata, json_file, image_root):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_lvis_json_with_id(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="lvis", **metadata
    )


def load_lvis_json_with_id(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file in LVIS's annotation format.,keep the ann_id

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "bbox", "bbox_mode", "category_id",
            "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    if dataset_name is not None:
        meta = get_lvis_instances_meta(dataset_name)
        MetadataCatalog.get(dataset_name).set(**meta)

    # sort indices for reproducible results
    img_ids = sorted(lvis_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        json_file
    )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file))

    if extra_annotation_keys:
        logger.info(
            "The following extra annotation keys will be loaded: {} ".format(extra_annotation_keys)
        )
    else:
        extra_annotation_keys = []

    def get_file_name(img_root, img_dict):
        # Determine the path including the split folder ("train2017", "val2017", "test2017") from
        # the coco_url field. Example:
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
        file_name = img_dict["coco_url"][30:]
        return os.path.join(img_root, file_name)

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = get_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS,'iscrowd':anno.get('iscrowd',0),'annid': anno['id']}
            # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
            # variable will have a field with COCO-specific category mapping.
            if dataset_name is not None and "thing_dataset_id_to_contiguous_id" in meta:
                obj["category_id"] = meta["thing_dataset_id_to_contiguous_id"][anno["category_id"]]
            else:
                obj["category_id"] = anno["category_id"] - 1  # Convert 1-indexed to 0-indexed
            segm = anno["segmentation"]  # list[list[float]]
            # filter out invalid polygons (< 3 points)
            valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
            # assert len(segm) == len(
            #     valid_segm
            # ), "Annotation contains an invalid polygon with < 3 points"
            assert len(segm) > 0
            obj["segmentation"] = segm
            for extra_ann_key in extra_annotation_keys:
                obj[extra_ann_key] = anno[extra_ann_key]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def get_lvis_instances_meta(dataset_name):
    """
    Load LVIS metadata.

    Args:
        dataset_name (str): LVIS dataset name without the split name (e.g., "lvis_v0.5").

    Returns:
        dict: LVIS metadata with keys: thing_classes
    """
    if "cocofied" in dataset_name:
        return _get_coco_instances_meta()
    if "v0.5" in dataset_name:
        return _get_lvis_instances_meta_v0_5()
    elif "v1" in dataset_name:
        return _get_lvis_instances_meta_v1()
    raise ValueError("No built-in metadata for dataset {}".format(dataset_name))


def _get_lvis_instances_meta_v0_5():
    assert len(LVIS_V0_5_CATEGORIES) == 1230
    cat_ids = [k["id"] for k in LVIS_V0_5_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V0_5_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


def _get_lvis_instances_meta_v1():
    assert len(LVIS_V1_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta
