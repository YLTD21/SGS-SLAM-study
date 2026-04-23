import argparse
import os
import sys
import json
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
from queue import Queue
from enum import Enum

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation
from viz_scripts.final_recon import load_camera, load_scene_data, make_lineset, render, rgbd2pcd

# Queue for inter-thread communication
command_queue = Queue()


class Operation(Enum):
    SWITCH_MODE = 1
    APPLY_MASK = 2
    CAM_TRANS = 3
    CAM_ROTATE = 4
    OBJ_TRANS = 5  # 新增：物体平移
    OBJ_ROT = 6  # 新增：物体旋转
    OBJ_SCALE = 7  # 新增：物体缩放


class ControlPanel(tk.Frame):
    def __init__(self, master, command_queue, cfg, width, height, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.master = master
        self.command_queue = command_queue
        master.title('SGS-SLAM 语义物体编辑器')

        # Semantic IDs
        self.semantic_ids_label = tk.Label(master, text='Semantic IDs:')
        self.semantic_ids_label.pack()
        self.semantic_ids = set([0])
        self.semantic_ids_value = tk.Label(master, text=', '.join(str(num) for num in sorted(self.semantic_ids)))
        self.semantic_ids_value.pack()

        # Mode
        self.mode_label = tk.Label(master, text='渲染模式:')
        self.mode_label.pack()
        self.mode_var = tk.StringVar(value='Colors')
        self.colors_radiobutton = tk.Radiobutton(master, text='Colors', variable=self.mode_var, value='color')
        self.colors_radiobutton.pack(anchor=tk.CENTER)
        self.centers_radiobutton = tk.Radiobutton(master, text='Centers', variable=self.mode_var, value='centers')
        self.centers_radiobutton.pack(anchor=tk.CENTER)
        self.semantic_colors_radiobutton = tk.Radiobutton(master, text='Semantic Colors', variable=self.mode_var,
                                                          value='semantic_color')
        self.semantic_colors_radiobutton.pack(anchor=tk.CENTER)

        # Manipulate
        self.manipulate_label = tk.Label(master, text='物体操作:')
        self.manipulate_label.pack()
        self.semantic_id_entry_label = tk.Label(master, text='输入 Semantic ID:')
        self.semantic_id_entry_label.pack()
        self.semantic_id_entry = tk.Entry(master)
        self.semantic_id_entry.pack()
        self.keep_var = tk.BooleanVar(value=True)
        self.keep_checkbox = tk.Checkbutton(master, text='Keep (显示)', variable=self.keep_var)
        self.keep_checkbox.pack()

        # 相机控制
        self.create_buttons("相机平移", ["X", "Y", "Z"], "move", "-", "+")
        self.create_buttons("相机旋转", ["X", "Y", "Z"], "rotate", "-", "+")

        # 新增：物体控制
        self.create_buttons("物体平移", ["X", "Y", "Z"], "obj_trans", "-", "+")
        self.create_buttons("物体旋转", ["X", "Y", "Z"], "obj_rot", "-", "+")
        self.create_buttons("物体缩放", ["X", "Y", "Z"], "obj_scale", "-", "+")

        # Apply / Reset buttons
        self.apply_button = tk.Button(master, text='Apply (应用)', command=self.apply)
        self.apply_button.pack(side=tk.LEFT)
        self.reset_button = tk.Button(master, text='Reset (重置)', command=self.reset)
        self.reset_button.pack(side=tk.RIGHT)

    def apply(self):
        mode = self.mode_var.get()
        semantic_id = self.semantic_id_entry.get()
        keep = self.keep_var.get()

        cmd = {'type': Operation.SWITCH_MODE, 'payload': {'mode': mode}}
        self.command_queue.put(cmd)

        if semantic_id.isdigit():
            cmd = {
                'type': Operation.APPLY_MASK,
                'payload': {
                    'semantic_id': int(semantic_id),
                    'to_keep': keep
                }
            }
            self.command_queue.put(cmd)

    def create_buttons(self, action, directions, transform_type, minus_text, plus_text):
        action_label = tk.Label(self.master, text=f"{action}:")
        action_label.pack()
        for direction in directions:
            button_frame = tk.Frame(self.master)
            button_frame.pack()
            label = tk.Label(button_frame, text=f"{direction}:")
            label.pack(side=tk.LEFT)
            minus_button = tk.Button(button_frame, text=f"{minus_text}",
                                     command=lambda d=direction: self.perform_transform(transform_type, d, -1))
            minus_button.pack(side=tk.LEFT)
            plus_button = tk.Button(button_frame, text=f"{plus_text}",
                                    command=lambda d=direction: self.perform_transform(transform_type, d, 1))
            plus_button.pack(side=tk.LEFT)

    def perform_transform(self, transform_type, direction, factor):
        if transform_type == 'move':
            self.command_queue.put({'type': Operation.CAM_TRANS, 'payload': {'direction': direction, 'factor': factor}})
        elif transform_type == 'rotate':
            self.command_queue.put(
                {'type': Operation.CAM_ROTATE, 'payload': {'direction': direction, 'factor': factor}})
        elif transform_type == 'obj_trans':
            self.command_queue.put({'type': Operation.OBJ_TRANS, 'payload': {'direction': direction, 'factor': factor}})
        elif transform_type == 'obj_rot':
            self.command_queue.put({'type': Operation.OBJ_ROT, 'payload': {'direction': direction, 'factor': factor}})
        elif transform_type == 'obj_scale':
            self.command_queue.put({'type': Operation.OBJ_SCALE, 'payload': {'direction': direction, 'factor': factor}})

    def reset(self):
        self.mode_var.set('Colors')
        self.semantic_id_entry.delete(0, tk.END)
        self.keep_var.set(False)
        self.command_queue.put({'type': Operation.SWITCH_MODE, 'payload': {'mode': 'color'}})
        self.command_queue.put({'type': Operation.APPLY_MASK, 'payload': {'semantic_id': -2}})


def move_camera_x(current_pose, distance):
    new_pose = current_pose.copy()
    translation_matrix = np.identity(4)
    translation_matrix[0, 3] = distance
    new_pose = new_pose @ translation_matrix
    return new_pose


def move_camera_y(current_pose, distance):
    new_pose = current_pose.copy()
    translation_matrix = np.identity(4)
    translation_matrix[1, 3] = distance
    new_pose = new_pose @ translation_matrix
    return new_pose


def move_camera_z(current_pose, distance):
    new_pose = current_pose.copy()
    translation_matrix = np.identity(4)
    translation_matrix[2, 3] = distance
    new_pose = new_pose @ translation_matrix
    return new_pose


def rotate_camera_x(current_pose, theta_degrees):
    new_pose = current_pose.copy()
    theta_radians = np.radians(theta_degrees)
    rotation_matrix = np.array([[1, 0, 0, 0],
                                [0, np.cos(theta_radians), -np.sin(theta_radians), 0],
                                [0, np.sin(theta_radians), np.cos(theta_radians), 0],
                                [0, 0, 0, 1]])
    new_pose = new_pose @ rotation_matrix
    return new_pose


def rotate_camera_y(current_pose, theta_degrees):
    new_pose = current_pose.copy()
    theta_radians = np.radians(theta_degrees)
    rotation_matrix = np.array([[np.cos(theta_radians), 0, np.sin(theta_radians), 0],
                                [0, 1, 0, 0],
                                [-np.sin(theta_radians), 0, np.cos(theta_radians), 0],
                                [0, 0, 0, 1]])
    new_pose = new_pose @ rotation_matrix
    return new_pose


def rotate_camera_z(current_pose, theta_degrees):
    new_pose = current_pose.copy()
    theta_radians = np.radians(theta_degrees)
    rotation_matrix = np.array([[np.cos(theta_radians), -np.sin(theta_radians), 0, 0],
                                [np.sin(theta_radians), np.cos(theta_radians), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    new_pose = new_pose @ rotation_matrix
    return new_pose


def visualize(scene_path, cfg):
    w2c, k = load_camera(cfg, scene_path)
    if 'load_semantics' in cfg:
        load_semantics = cfg['load_semantics']
    else:
        load_semantics = False

    scene_data, scene_depth_data, scene_semantic_data, all_w2cs, semantic_ids = load_scene_data(
        scene_path, w2c, k, load_semantics=load_semantics)

    render_mask = torch.ones(scene_data['means3D'].shape[0], dtype=torch.bool).cuda()
    selected_semantic_id_global = -1  # 全局：当前选中的物体ID

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']),
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg, render_mask)
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    if cfg['visualize_cams']:
        frustum_size = 0.045
        num_t = len(all_w2cs)
        cam_centers = []
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for i_t in range(num_t):
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(np.array(cam_colormap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])

        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines) + norm_factor)[:3]))
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    focal_length_scale_factor = 0.8
    k[0, 0] *= focal_length_scale_factor
    k[1, 1] *= focal_length_scale_factor
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if cfg['offset_first_viz_cam']:
        view_w2c = w2c
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = w2c

    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False
    render_options.background_color = [0.0, 0.0, 0.0]

    render_mode = cfg['render_mode']
    delta_trans = 0.2
    delta_rotate = 2.5
    obj_delta_trans = 0.1  # 物体平移步长
    obj_delta_rot = 5.0  # 物体旋转步长（度）
    obj_delta_scale = 0.1  # 物体缩放步长
    set_camera_w2c = False

    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        while not command_queue.empty():
            msg = command_queue.get()
            print(msg)
            if msg['type'] == Operation.SWITCH_MODE:
                render_mode = msg['payload']['mode']
            elif load_semantics and msg['type'] == Operation.APPLY_MASK:
                input_semantic_id = int(msg['payload']['semantic_id'])
                if input_semantic_id == -2:
                    render_mask = torch.ones(scene_data['means3D'].shape[0], dtype=torch.bool).cuda()
                    selected_semantic_id_global = -1
                elif input_semantic_id == -1:
                    continue
                else:
                    to_keep = msg['payload']['to_keep']
                    render_mask[semantic_ids.squeeze() == input_semantic_id] = bool(to_keep)
                    selected_semantic_id_global = input_semantic_id
            elif msg['type'] == Operation.CAM_TRANS:
                set_camera_w2c = True
                if msg['payload']['direction'] == 'X':
                    w2c = move_camera_x(w2c, msg['payload']['factor'] * delta_trans)
                elif msg['payload']['direction'] == 'Y':
                    w2c = move_camera_y(w2c, msg['payload']['factor'] * delta_trans)
                elif msg['payload']['direction'] == 'Z':
                    w2c = move_camera_z(w2c, msg['payload']['factor'] * delta_trans)
            elif msg['type'] == Operation.CAM_ROTATE:
                set_camera_w2c = True
                if msg['payload']['direction'] == 'X':
                    w2c = rotate_camera_x(w2c, msg['payload']['factor'] * delta_rotate)
                elif msg['payload']['direction'] == 'Y':
                    w2c = rotate_camera_y(w2c, msg['payload']['factor'] * delta_rotate)
                elif msg['payload']['direction'] == 'Z':
                    w2c = rotate_camera_z(w2c, msg['payload']['factor'] * delta_rotate)

            # ================== 新增：物体平移 ==================
            elif msg['type'] == Operation.OBJ_TRANS:
                if selected_semantic_id_global == -1:
                    continue
                dir = msg['payload']['direction']
                factor = msg['payload']['factor']
                dist = factor * obj_delta_trans
                trans_vec = torch.zeros(3).cuda()
                if dir == 'X':
                    trans_vec[0] = dist
                elif dir == 'Y':
                    trans_vec[1] = dist
                elif dir == 'Z':
                    trans_vec[2] = dist
                selected_mask = (semantic_ids.squeeze() == selected_semantic_id_global)
                scene_data['means3D'][selected_mask] += trans_vec

            # ================== 新增：物体旋转 ==================
            elif msg['type'] == Operation.OBJ_ROT:
                if selected_semantic_id_global == -1:
                    continue
                dir = msg['payload']['direction']
                factor = msg['payload']['factor']
                theta = np.deg2rad(factor * obj_delta_rot)
                rot_mat = torch.eye(3).cuda()
                if dir == 'X':
                    rot_mat = torch.tensor([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)],
                                            [0, np.sin(theta), np.cos(theta)]]).float().cuda()
                elif dir == 'Y':
                    rot_mat = torch.tensor([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                                            [-np.sin(theta), 0, np.cos(theta)]]).float().cuda()
                elif dir == 'Z':
                    rot_mat = torch.tensor([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]]).float().cuda()
                selected_mask = (semantic_ids.squeeze() == selected_semantic_id_global)
                means = scene_data['means3D'][selected_mask]
                center = means.mean(dim=0)
                means_centered = means - center
                means_rot = (rot_mat @ means_centered.T).T
                scene_data['means3D'][selected_mask] = means_rot + center

            # ================== 新增：物体缩放 ==================
            elif msg['type'] == Operation.OBJ_SCALE:
                if selected_semantic_id_global == -1:
                    continue
                dir = msg['payload']['direction']
                factor = msg['payload']['factor']
                scale = 1.0 + factor * obj_delta_scale
                selected_mask = (semantic_ids.squeeze() == selected_semantic_id_global)
                means = scene_data['means3D'][selected_mask]
                center = means.mean(dim=0)
                means_centered = means - center
                means_scaled = means_centered * scale
                scene_data['means3D'][selected_mask] = means_scaled + center
                scene_data['log_scales'][selected_mask] += np.log(scale)

        if render_mode == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'][render_mask].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(
                scene_data['colors_precomp'][render_mask].contiguous().double().cpu().numpy())
        elif render_mode == 'semantic_color':
            seg, depth, sil = render(w2c, k, scene_semantic_data, scene_depth_data, cfg, render_mask)
            pts, cols = rgbd2pcd(seg, depth, w2c, k, cfg)
        else:
            im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg, render_mask)
            if cfg['show_sil']:
                im = (1 - sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)

        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

        if set_camera_w2c:
            cam_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)
            set_camera_w2c = False

    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()
    experiment = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()
    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(experiment.config["workdir"], experiment.config["run_name"])
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]

    print(scene_path)
    viz_cfg = experiment.config["viz"]

    thread = Thread(target=visualize, args=(scene_path, viz_cfg))
    thread.daemon = True
    thread.start()

    root = tk.Tk()
    control_panel = ControlPanel(root, command_queue, viz_cfg, width=20, height=200)
    root.mainloop()