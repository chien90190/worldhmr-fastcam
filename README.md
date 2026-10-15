# A Diagnostic Study of World-Space Human Recovery Under High Camera Dynamics

[Hao-Jen Chien](https://chien90190.github.io/)<sup>1</sup>,
[Wei-Lun Chao](https://sites.google.com/view/wei-lun-harry-chao)<sup>2</sup>,
[Yu-Lun Liu](https://yulunalexliu.github.io/)<sup>1</sup>\
<sup>1</sup>National Yang Ming Chiao Tung University, <sup>2</sup>Boston University\
[[Website](https://chien90190.github.io/worldhmr-fastcam-website/)]
[[Dataset](https://huggingface.co/datasets/chien90190/worldhmr-fastcam)]
[[arXiv](https://arxiv.org/abs/2610.19729)]



https://github.com/user-attachments/assets/df25a46a-09fd-45a5-b04f-382429578357


https://github.com/user-attachments/assets/7a94614f-c85b-49a1-883b-65c9d693c102

Recovers people in a video as a 3D SMPL-X body in world coordinates, with drift correction
for fast-moving cameras such as K-pop broadcasts. For each video the pipeline estimates the camera
with [ViPE](https://github.com/nv-tlabs/vipe), tracks people with YOLO11 and SAM 2.1, estimates
their bodies with PromptHMR-Vid (or GVHMR, TRAM), places them in a gravity-aligned world, removes
the camera drift using planted feet, and refines people and camera with a physically grounded
optimization (PGO).

## Install

Requires Linux, an NVIDIA GPU and conda. Run all commands from the repository root.

```bash
bash scripts/install.sh          # conda env worldhmr-fastcam (--env NAME for another name)
conda activate worldhmr-fastcam
bash scripts/fetch_smplx.sh      # body models, needs SMPL-X and SMPL accounts
bash scripts/fetch_data.sh       # checkpoints
```

The downloads go to:

```
data/
├── body_models/
│   ├── smplx/SMPLX_NEUTRAL.npz
│   ├── smpl/SMPL_NEUTRAL.pkl
│   ├── smplx2smpl.pkl
│   └── smplx2smpl_joints.npy
└── pretrain/
    ├── phmr/checkpoint.ckpt, config.yaml                    PromptHMR
    ├── phmr_vid/phmr_b1b2.ckpt, prhmr_release_002.yaml      PromptHMR-Vid
    ├── sam2_ckpts/sam2.1_hiera_large.pt                     SAM 2.1
    ├── camcalib_sa_biased_l2.ckpt                           SPEC
    ├── vitpose-h-coco_25.pth                                ViTPose
    └── yolo11x-pose.pt                                      YOLO11
```

**ViPE.** Install [ViPE](https://github.com/nv-tlabs/vipe) in a conda env named `vipe`, then add
our config:

```bash
cp third_party/vipe/kpop.yaml <ViPE checkout>/configs/pipeline/
```

## Run

```bash
python scripts/demo_video.py --input_video dance.mp4 --output_dir results
```

`--help` lists the options; other settings are in `pipeline/config.yaml`. `bash run.sh` takes the
same arguments and picks the GPU with the most free memory.

## Outputs

`results/<video name>/results.npz` holds the camera and every person's SMPL-X on every frame, in
the same format as the dataset annotations:

| key | shape | meaning |
|---|---|---|
| `body_model` | () | `"smplx"` |
| `camera/intrinsics` | (3,3) | pinhole intrinsics, pixels |
| `camera/extrinsics` | (T,4,4) | world to camera, OpenCV |
| `track_id` | (P,) | person ids |
| `detected` | (P,T) | False where the pose is filled in |
| `bbox_xyxy`, `keypoints_2d` | (P,T,4), (P,T,25,3) | box, OpenPose BODY_25 |
| `contact`, `contact_joints` | (P,T,6), (6,) | planted ankles, feet, wrists |
| `smplx/betas`, `global_orient`, `body_pose`, `transl` | (P,10), (P,T,3), (P,T,63), (P,T,3) | world, +y up, meters |
| `smplx_incam/global_orient`, `body_pose`, `transl` | (P,T,3), (P,T,63), (P,T,3) | camera frame |

Build the mesh with `smplx.SMPLX(use_pca=False, flat_hand_mean=True, num_betas=10)`, all other
parameters zero.

## Rendering

```bash
python scripts/render.py --results results/dance/results.npz --video dance.mp4 --output dance_render.mp4
```

Draws the meshes over the video, next to a world view with the camera path.

## Evaluation on the K-pop dataset

Download the [dataset](https://huggingface.co/datasets/chien90190/worldhmr-fastcam) and save
your predictions as `<predictions>/<clip name>/results.npz`, then:

```bash
python evaluation/evaluate.py --dataset <dataset folder> --pred <predictions> --set main  # PCK, jitter, foot skating
python evaluation/evaluate.py --dataset <dataset folder> --pred <predictions> --set pair  # WA-/W-/PA-MPJPE, pair distance, formation, scale
```

`--masks <folder>` adds Mask IoU against your own person masks. `evaluation/convert.py` converts
PromptHMR-style `results.pkl` and TRAM outputs; `evaluation/smpl_to_smplx.py` turns an SMPL
result into SMPL-X.

## Other HMR backbones

**GVHMR.** Install [GVHMR](https://github.com/zju3dv/GVHMR) with its checkpoints in a conda env named
`gvhmr`, then:

```bash
python scripts/demo_video.py --input_video dance.mp4 --output_dir results_gvhmr \
    --hmr gvhmr --gvhmr_dir <GVHMR checkout> --vipe_dir results/vipe
```

**TRAM.** Install [TRAM](https://github.com/yufu-wang/tram), apply our patch so it reads the ViPE
camera, and run it:

```bash
cd <TRAM checkout>
git apply <this repo>/third_party/tram/vipe_camera.patch
python scripts/estimate_camera.py --video <path>/dance.mp4 --output_dir <tram output> --vipe_dir <this repo>/results/vipe
python scripts/estimate_humans.py --video <path>/dance.mp4 --output_dir <tram output>
```

TRAM has no foot contact, so it takes it from a default run of the same video in `results/`:

```bash
python scripts/demo_video.py --input_video dance.mp4 --output_dir results_tram \
    --hmr tram --tram_dir <tram output> --static_conf_dir results --vipe_dir results/vipe
```

To add another odometry or HMR backbone, add its function to `ODOMETRIES` in
`pipeline/odometry.py` or `HMR_BACKBONES` in `pipeline/hmr/__init__.py`.

## Citation

```bibtex
@article{chien2026diagnostic,
  title   = {A Diagnostic Study of World-Space Human Recovery Under High Camera Dynamics},
  author  = {Chien, Hao-Jen and Chao, Wei-Lun and Liu, Yu-Lun},
  journal = {arXiv preprint arXiv:2610.19729},
  year    = {2026}
}
```

## Acknowledgements

The code builds on [PromptHMR](https://github.com/yufu-wang/PromptHMR) and uses
[ViPE](https://github.com/nv-tlabs/vipe),
[YOLO11](https://github.com/ultralytics/ultralytics),
[SAM 2](https://github.com/facebookresearch/sam2),
[ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
(through [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose)),
[SPEC](https://github.com/mkocabas/SPEC), [GVHMR](https://github.com/zju3dv/GVHMR) and
[TRAM](https://github.com/yufu-wang/tram).

## License

For non-commercial research only, under PromptHMR's terms; see [LICENSE](LICENSE). The models
above, SMPL-X and SMPL have their own licenses. The dataset is under CC BY-NC 4.0.
