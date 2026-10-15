import os
import sys
import time
from typing import Literal, Optional

import torch
import tyro

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.vis.viser import viser_vis_world4d
from prompt_hmr.vis.traj import get_floor_mesh
from pipeline.pipeline import Pipeline


def main(input_video: str,
         output_dir: str = 'results',
         hmr: Literal['phmr', 'gvhmr', 'tram'] = 'phmr',
         vipe_dir: Optional[str] = None,
         gvhmr_dir: Optional[str] = None,
         tram_dir: Optional[str] = None,
         static_conf_dir: Optional[str] = None,
         run_viser: bool = False,
         viser_total: int = 1500,
         viser_subsample: int = 1,
         ):
    """
    Reconstruct world-space 3D humans from a video into <output_dir>/<video name>/.

    Args:
        input_video: path to the video.
        output_dir: results go to <output_dir>/<video name>/.
        hmr: camera-frame HMR backbone.
        vipe_dir: ViPE output folder (default <output_dir>/vipe); ViPE runs only if the video has none.
        gvhmr_dir: GVHMR checkout, for --hmr gvhmr.
        tram_dir: TRAM output folder, for --hmr tram.
        static_conf_dir: output_dir of a phmr run, whose contact confidence --hmr tram uses.
        run_viser: show the result in a viser 3D viewer afterwards.
        viser_total: frames shown in the viewer.
        viser_subsample: show every n-th frame in the viewer.
    """
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    output_folder = os.path.join(output_dir, video_name)
    overrides = {'hmr_type': hmr, 'vipe_dir': vipe_dir, 'gvhmr_dir': gvhmr_dir,
                 'tram_dir': tram_dir, 'static_conf_dir': static_conf_dir}
    pipeline = Pipeline(**overrides)
    results = pipeline(input_video, output_folder)

    if run_viser and results is not None:
        show_in_viser(pipeline, viser_total, viser_subsample)


def show_in_viser(pipeline, viser_total, viser_subsample):
    smplx = SMPLX_Layer('data/body_models/smplx').cuda()
    images = pipeline.images[:viser_total][::viser_subsample]
    world4d = pipeline.create_world4d(step=viser_subsample, total=viser_total)
    world4d = {i: world4d[k] for i, k in enumerate(world4d)}

    all_verts = []
    for world3d in world4d.values():
        if len(world3d['track_id']) == 0:
            continue
        rotmat = axis_angle_to_matrix(world3d['pose'].reshape(-1, 55, 3))
        verts = smplx(global_orient=rotmat[:, :1].cuda(),
                      body_pose=rotmat[:, 1:22].cuda(),
                      betas=world3d['shape'].cuda(),
                      transl=world3d['trans'].cuda()).vertices.cpu().numpy()
        world3d['vertices'] = verts
        all_verts.append(torch.tensor(verts, dtype=torch.bfloat16))
    [gv, gf, gc] = get_floor_mesh(torch.cat(all_verts), scale=2)

    server, gui = viser_vis_world4d(images, world4d, smplx.faces, floor=[gv, gf],
                                    init_fps=pipeline.fps / viser_subsample)
    print(f'Open https://localhost:{server.get_port()} to view the result.')

    gui_playing, gui_timestep, gui_framerate, num_frames = gui
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == '__main__':
    tyro.cli(main)
