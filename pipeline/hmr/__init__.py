"""
HMR backbones: every tracked person's SMPL-X in camera coordinates.

A backbone is a function (images, results, smplx, cfg, output_folder) -> results that sets,
for every person k it keeps in results['people'], results['people'][k]['smplx_cam'] with

    rotmat               (Ti,55,3,3)  SMPL-X joint rotations, root in camera coordinates
    pose                 (Ti,165)     the same as axis-angle
    shape                (Ti,10)      betas
    trans                (Ti,3)       translation in camera coordinates, meters
    static_conf_logits   (Ti,6)       foot and wrist contact logits, columns as
                                      CONTACT_JOINT_IDS in joints.py; optional,
                                      see static_conf.py

on the person's frames (Ti). A backbone that predicts SMPL maps body joints 1-21 onto SMPL-X.

To add one, write such a function, add it to HMR_BACKBONES with the name shown in the output,
and set `hmr_type: <key>` in config.yaml.
"""
import os


def run_phmr(images, results, smplx, cfg, output_folder):
    """PromptHMR-Vid, in this env."""
    from pipeline.hmr.phmr import PromptHMR_Video
    return PromptHMR_Video().run(images, results, smplx, mask_prompt=True)


def run_gvhmr(images, results, smplx, cfg, output_folder):
    """GVHMR, in its own checkout (cfg.gvhmr_dir) and conda env (cfg.gvhmr_env)."""
    from pipeline.hmr.gvhmr import GVHMR_Video
    return GVHMR_Video(cfg.gvhmr_dir, cfg.gvhmr_env).run(images, results)


def run_tram(images, results, smplx, cfg, output_folder):
    """TRAM's saved output for this video, in cfg.tram_dir/<video name>/."""
    from pipeline.hmr.tram import TRAM_Video
    video_name = os.path.basename(os.path.normpath(output_folder))
    return TRAM_Video().run(results, os.path.join(cfg.tram_dir, video_name))


# key in config.yaml -> (name shown in the output, function)
HMR_BACKBONES = {
    'phmr': ('PromptHMR-Vid', run_phmr),
    'gvhmr': ('GVHMR', run_gvhmr),
    'tram': ('TRAM', run_tram),
}
