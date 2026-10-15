#!/bin/bash
# Checkpoints the pipeline loads, into data/pretrain/.
set -e
mkdir -p ./data/pretrain/phmr ./data/pretrain/phmr_vid ./data/pretrain/sam2_ckpts

# PromptHMR image model
gdown --folder -O ./data/pretrain/ https://drive.google.com/drive/folders/1EQ7arZz135T-WpxkS_K1R_hjZp3prh-y?usp=share_link

# PromptHMR-Vid video head, trained on BEDLAM 1 and 2, and its config
gdown -O ./data/pretrain/phmr_vid/phmr_b1b2.ckpt 1a0IqV5IBA5L0H9BYbLRVLm3KwQDPXFO0
gdown -O ./data/pretrain/phmr_vid/prhmr_release_002.yaml 1t282bZ_VmTSmB4GnivJRyVhLIsvhyscx

# SAM 2.1, SPEC camera calibration, ViTPose, YOLO11-pose
wget -O ./data/pretrain/sam2_ckpts/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
gdown --fuzzy -O ./data/pretrain/camcalib_sa_biased_l2.ckpt https://drive.google.com/file/d/1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO/view?usp=sharing
gdown --fuzzy -O ./data/pretrain/vitpose-h-coco_25.pth https://drive.google.com/file/d/1ZprPoNXe_f9a9flr0RhS3XCJBfqhFSeE/view?usp=sharing
wget -O ./data/pretrain/yolo11x-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt
