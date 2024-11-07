## 🔢 Data Generation
### 1️⃣ Instance Generation
Preliminary
1. install environment and download checkpoints refer to [DeepFloyd-IF](https://github.com/deep-floyd/IF)
2. generate data using Stable Diffusion refer to [X-Paste](https://github.com/yoctta/xpaste)

#### 1.1 use manually designed prompts
```python
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --use_env \
    generation/txt2img_diffusers_stages_from_txt.py \
    --from_file input/lvis_prompt/ \
    --outdir OUT_DIR \
    --n_samples 1024 \
    --max_batch_size 1 \
    --offset 0 \
    --seed 42 \
    --dist \
    --ckpt_dir CKPT_DIR/ \
    --stages I II
```

#### 1.2 use ChatGPT designed prompts
```python
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --use_env \
    generation/txt2img_diffusers_stages_from_txt.py \
    --from_file input/gpt_prompt/ \
    --outdir OUT_DIR \
    --n_samples 8 \
    --max_batch_size 1 \
    --offset 0 \
    --seed 42 \
    --dist \
    --ckpt_dir CKPT_DIR/ \
    --stages I II
```

#### 1.3 use ImageNet categories
```python
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --use_env \
    generation/txt2img_diffusers_stages_from_txt.py \
    --from_file input/imgnet_prompt/ \
    --outdir OUT_DIR \
    --n_samples 1024 \
    --max_batch_size 1 \
    --offset 0 \
    --seed 42 \
    --dist \
    --ckpt_dir CKPT_DIR/ \
    --stages I II
```

#### 1.4 convert folder structure
```python
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --use_env \
    generation/convert_dir_structure.py \
    --indir OUT_DIR \
    --in_lvis_json_path datasets/metadata/ImageNet2012_filtered04_lvis_v1_train_cat_info_250.json \
    --outdir OUT_DIR_RE \
    --stages II \
    --n_samples 1024 \
    --dist
```

### 2️⃣ Instance Annotation
Preliminary
1. install environment and download checkpoints refer to [SAM](https://github.com/facebookresearch/segment-anything)

```python
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --use_env \
    segmentation/get_background_sam_mask.py \
    --in_img_dir OUT_DIR_RE \
    --out_mask_dir OUT_DIR_RE_SEG \
    --in_lvis_json_path datasets/metadata/ImageNet2012_filtered04_lvis_v1_train_cat_info_250.json \
    --stages II \
    --seg_name background_m0 \
    --background_mode \
    --corner_margin 0 \
    --corner_location all \
    --n_samples 1024 \
    --dist
```

### 3️⃣ Instance Filteration
Preliminary
1. install environment and download checkpoints refer to [CLIP](https://github.com/openai/CLIP)

#### 3.1 crop LVIS instance patch
```python
python filteration/convert_lvis_to_coco_crop.py \
    --dataset_root_dir DATA_DIR/lvis \
    --dataset_json_path DATA_DIR/lvis/lvis_v1_train.json \
    --dataset_out_dir LVIS_CROP_DIR \
    --split train \
    --mode padding \
    --fill blur \
    --padding_width 40
```

#### 3.2 get image feature
```python
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --use_env \
    filteration/get_image_feature.py \
    --dataset_json_path DATA_DIR/lvis/lvis_v1_train.json \
    --process_lvis \
    --process_gen \
    --lvis_crop_in_dir LVIS_CROP_DIR/ \
    --gen_in_dir OUT_DIR_RE \
    --gen_mask_in_dir OUT_DIR_RE_SEG \
    --result_out_dir OUT_FEATURE_DIR \
    --method clip
```

#### 3.3 get image similarity from feature
```python
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --use_env \
    filteration/get_image_similarity_from_feature.py \
    --dataset_json_path DATA_DIR/lvis/lvis_v1_train.json \
    --lvis_crop_in_dir OUT_FEATURE_DIR/lvis \
    --gen_in_dir OUT_FEATURE_DIR/gen \
    --result_out_dir OUT_SIM_DIR \
    --method clip
```

#### 3.4 filter image by similarity
```python
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --use_env \
    filteration/filter_image_by_similarity.py \
    --dataset_json_path DATA_DIR/lvis/lvis_v1_train.json \
    --result_in_dir OUT_SIM_DIR \
    --result_out_dir OUT_SIM_DIR \
    --threshold 0.6 \
    --dist
```

#### 3.5 clean image pool
```python
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --use_env \
    segment_methods/clean_pool_if.py \
    --input_dir OUT_DIR_RE_SEG \
    --image_dir OUT_DIR_RE \
    --output_file OUT_DIR_RE_SEG/test_filter_0.6.json \
    --filter_image_csv_path OUT_SIM_DIR/filename_thres_0.6.csv \
    --seg_method background_m0 \
    --stages II \
    --dist \
    --backend gloo
```