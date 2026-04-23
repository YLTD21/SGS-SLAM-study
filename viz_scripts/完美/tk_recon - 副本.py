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
# 全局变量：同步语义ID
edit_semantic_id = -1
edit_transform = np.eye(4)
semantic_ids_global = None
control_panel_global = None

# 右键拖拽状态（由渲染线程读写，主线程只注册回调）
drag_state = {
    'active': False,          # 是否正在拖拽
    'last_x': 0,              # 上一帧鼠标屏幕X
    'last_y': 0,              # 上一帧鼠标屏幕Y
    'semantic_id': -1,        # 拖拽目标语义ID（-1=使用当前selected）
    'z_mode': False,          # True=拖拽控制Z轴（上下），False=控制XY平面
}

def _load_scannet_label_map():
    """从 preprocess/scannet/scannetv2-labels.combined.tsv 加载完整标签映射"""
    tsv_path = os.path.join(_BASE_DIR, 'preprocess', 'scannet', 'scannetv2-labels.combined.tsv')
    label_map = {0: 'background'}
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过表头
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        label_map[int(parts[0])] = parts[1].strip()
                    except ValueError:
                        pass
        print(f"[Label Map] Loaded {len(label_map)} ScanNet labels from TSV.")
    except FileNotFoundError:
        print(f"[Label Map] TSV not found at {tsv_path}, using built-in fallback.")
        label_map.update({
            1: 'wall', 2: 'chair', 3: 'floor', 4: 'table', 5: 'door',
            6: 'couch', 7: 'cabinet', 8: 'shelf', 9: 'desk', 10: 'office chair',
            11: 'bed', 13: 'pillow', 14: 'sink', 15: 'picture', 16: 'window',
            18: 'bookshelf', 19: 'monitor', 21: 'curtain', 22: 'books', 28: 'lamp',
            41: 'ceiling', 56: 'trash can', 71: 'mirror',
        })
    return label_map


SEMANTIC_CLASSES = _load_scannet_label_map()


class Operation(Enum):
    SWITCH_MODE = 1
    APPLY_MASK = 2
    CAM_TRANS = 3
    CAM_ROTATE = 4
    OBJ_TRANS = 5
    OBJ_ROT = 6
    OBJ_SCALE = 7
    SAVE_SCENE = 8
    UPDATE_SEMANTIC_IDS = 9
    OBJ_DRAG_TRANS = 10       # 右键拖拽平移


class ControlPanel(tk.Frame):
    def __init__(self, master, command_queue, cfg, width, height, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.master = master
        self.command_queue = command_queue
        master.title('SGS-SLAM Scene Editor')

        # Semantic IDs
        self.semantic_ids_label = tk.Label(master, text='Semantic IDs in Scene:')
        self.semantic_ids_label.pack()
        self.semantic_ids = set([0])
        self.semantic_ids_value = tk.Label(master, text=', '.join(str(num) for num in sorted(self.semantic_ids)))
        self.semantic_ids_value.pack()

        # 新增：悬停信息显示
        self.hover_label = tk.Label(master, text='Hover Info:', font=('Arial', 10, 'bold'))
        self.hover_label.pack(pady=(10, 2))
        self.hover_info = tk.Label(master, text='No object selected', fg='blue', wraplength=250)
        self.hover_info.pack()

        # 添加调试信息标签
        self.debug_label = tk.Label(master, text='Debug: Waiting for mouse...', fg='gray', wraplength=250)
        self.debug_label.pack(pady=(5, 0))

        # Mode
        self.mode_label = tk.Label(master, text='Render Mode:')
        self.mode_label.pack(pady=(10, 2))
        self.mode_var = tk.StringVar(value='Colors')
        self.colors_radiobutton = tk.Radiobutton(master, text='Colors', variable=self.mode_var, value='color')
        self.colors_radiobutton.pack(anchor=tk.CENTER)
        self.centers_radiobutton = tk.Radiobutton(master, text='Centers', variable=self.mode_var, value='centers')
        self.centers_radiobutton.pack(anchor=tk.CENTER)
        self.semantic_colors_radiobutton = tk.Radiobutton(master, text='Semantic Colors', variable=self.mode_var,
                                                          value='semantic_color')
        self.semantic_colors_radiobutton.pack(anchor=tk.CENTER)

        # Manipulate
        self.manipulate_label = tk.Label(master, text='Object Manipulation:')
        self.manipulate_label.pack(pady=(15, 2))
        self.semantic_id_entry_label = tk.Label(master, text='Input Semantic ID:')
        self.semantic_id_entry_label.pack()
        self.semantic_id_entry = tk.Entry(master)
        self.semantic_id_entry.pack()
        self.keep_var = tk.BooleanVar(value=True)
        self.keep_checkbox = tk.Checkbutton(master, text='Keep Only This Object', variable=self.keep_var)
        self.keep_checkbox.pack()

        # 相机控制
        self.create_buttons("Camera Translation", ["X", "Y", "Z"], "move", "-", "+")
        self.create_buttons("Camera Rotation", ["X", "Y", "Z"], "rotate", "-", "+")

        # 物体控制
        self.create_buttons("Object Translation", ["X", "Y", "Z"], "obj_trans", "-", "+")
        self.create_buttons("Object Rotation", ["X", "Y", "Z"], "obj_rot", "-", "+")
        self.create_buttons("Object Scale", ["X", "Y", "Z"], "obj_scale", "-", "+")

        # 右键拖拽平移控制区
        self.drag_label = tk.Label(master, text='Right-Drag Translation:', font=('Arial', 9, 'bold'))
        self.drag_label.pack(pady=(15, 2))

        # 灵敏度调节
        sens_frame = tk.Frame(master)
        sens_frame.pack()
        tk.Label(sens_frame, text='Sensitivity:', width=10).pack(side=tk.LEFT)
        self.drag_sensitivity = tk.DoubleVar(value=0.005)
        self.drag_sens_scale = tk.Scale(
            master, from_=0.001, to=0.05, resolution=0.001,
            orient=tk.HORIZONTAL, variable=self.drag_sensitivity,
            length=180, label='Drag Speed (m/px)'
        )
        self.drag_sens_scale.pack()

        # 轴锁定
        axis_frame = tk.Frame(master)
        axis_frame.pack(pady=(4, 0))
        tk.Label(axis_frame, text='Axis Lock:', width=8).pack(side=tk.LEFT)
        self.drag_axis_var = tk.StringVar(value='XY')
        for ax in ['XY', 'X', 'Y', 'Z']:
            tk.Radiobutton(axis_frame, text=ax, variable=self.drag_axis_var, value=ax).pack(side=tk.LEFT)

        # 拖拽状态指示灯
        self.drag_status_label = tk.Label(master, text='Drag: Idle', fg='gray')
        self.drag_status_label.pack(pady=(2, 0))

        # Apply / Show All / Reset / Save
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(pady=(15, 0))
        self.apply_button = tk.Button(self.button_frame, text='Apply', command=self.apply, width=8)
        self.apply_button.pack(side=tk.LEFT, padx=5)
        self.show_all_button = tk.Button(self.button_frame, text='Show All', command=self.show_all,
                                         width=8, bg='#d0f0d0')
        self.show_all_button.pack(side=tk.LEFT, padx=5)
        self.reset_button = tk.Button(self.button_frame, text='Reset', command=self.reset, width=8)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(self.button_frame, text='Save Scene', command=self.save_scene, width=10)
        self.save_button.pack(side=tk.LEFT, padx=5)

    def update_hover_info(self, semantic_id, position=None):
        if semantic_id is None:
            self.hover_info.config(text='No object selected', fg='gray')
            self.debug_label.config(text='No object under mouse', fg='gray')
        else:
            sid = int(semantic_id)
            class_name = SEMANTIC_CLASSES.get(sid, f'Unknown(ID={sid})')
            info_text = f"ID: {sid}  |  Class: {class_name}"
            if position is not None:
                info_text += f"\nPos: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            self.hover_info.config(text=info_text, fg='blue')
            self.debug_label.config(text=f'Hover: ID {sid} [{class_name}]', fg='green')

    def update_debug_info(self, text):
        self.debug_label.config(text=text, fg='orange')

    def update_drag_status(self, active, semantic_id=-1):
        """更新拖拽状态指示（可从任意线程调用）"""
        def _update():
            if active:
                class_name = SEMANTIC_CLASSES.get(int(semantic_id), f'Unknown(ID={semantic_id})')
                self.drag_status_label.config(
                    text=f'Drag: ACTIVE  ID={semantic_id} [{class_name}]  Axis={self.drag_axis_var.get()}',
                    fg='red'
                )
            else:
                self.drag_status_label.config(text='Drag: Idle', fg='gray')
        self.master.after(0, _update)

    def update_semantic_ids(self, new_ids):
        self.semantic_ids = set(new_ids)
        self.semantic_ids_value.config(text=', '.join(str(num) for num in sorted(self.semantic_ids)))
        self.master.update_idletasks()

    def apply(self):
        mode = self.mode_var.get()
        semantic_id = self.semantic_id_entry.get().strip()
        keep_only = self.keep_var.get()

        cmd = {'type': Operation.SWITCH_MODE, 'payload': {'mode': mode}}
        self.command_queue.put(cmd)

        if semantic_id:
            try:
                input_id = float(semantic_id)
                if input_id in self.semantic_ids:
                    cmd = {
                        'type': Operation.APPLY_MASK,
                        'payload': {
                            'semantic_id': input_id,
                            'keep_only': keep_only
                        }
                    }
                    self.command_queue.put(cmd)
                    class_name = SEMANTIC_CLASSES.get(int(input_id), f'Unknown(ID={int(input_id)})')
                    action = "Keep Only (isolate object)" if keep_only else "Select for dragging (full map visible)"
                    print(f"Success: Semantic ID {input_id} [{class_name}] | {action}")
                else:
                    print(f"Invalid ID: {input_id} not in scene's Semantic IDs!")
                    messagebox.showwarning("Invalid ID", f"ID {input_id} not found in scene!")
            except ValueError:
                print(f"Invalid input: Please enter a number!")
                messagebox.showwarning("Invalid Input", "Please enter a valid number!")

    def create_buttons(self, action, directions, transform_type, minus_text, plus_text):
        action_label = tk.Label(self.master, text=f"{action}:")
        action_label.pack(pady=(10, 2))
        for direction in directions:
            button_frame = tk.Frame(self.master)
            button_frame.pack()
            label = tk.Label(button_frame, text=f"{direction}:", width=3)
            label.pack(side=tk.LEFT)
            minus_button = tk.Button(button_frame, text=f"{minus_text}", width=3,
                                     command=lambda d=direction: self.perform_transform(transform_type, d, -1))
            minus_button.pack(side=tk.LEFT, padx=2)
            plus_button = tk.Button(button_frame, text=f"{plus_text}", width=3,
                                    command=lambda d=direction: self.perform_transform(transform_type, d, 1))
            plus_button.pack(side=tk.LEFT, padx=2)

    def perform_transform(self, transform_type, direction, factor):
        global edit_semantic_id, edit_transform
        sid = self.semantic_id_entry.get()
        if not sid.isdigit():
            print("请先输入有效语义ID！")
            return
        edit_semantic_id = int(sid)

        mat = np.eye(4)
        step = 0.05 if transform_type == 'move' else 5.0
        step *= factor

        if transform_type == 'move':
            if direction == 'X': mat[0, 3] = step
            if direction == 'Y': mat[1, 3] = step
            if direction == 'Z': mat[2, 3] = step
        else:
            rad = np.radians(step)
            if direction == 'X':
                mat[:3, :3] = [[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]]
            if direction == 'Y':
                mat[:3, :3] = [[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]]
            if direction == 'Z':
                mat[:3, :3] = [[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]]

        edit_transform = mat @ edit_transform
        print(f"已对语义ID {edit_semantic_id} 应用变换")

    def show_all(self):
        """只重置mask显示全图，保留当前selected_id（可继续拖拽移动）"""
        self.command_queue.put({'type': Operation.APPLY_MASK, 'payload': {'semantic_id': -2, 'keep_selected': True}})
        print("Show All: full map visible, selected ID retained for dragging")

    def reset(self):
        self.mode_var.set('Colors')
        self.semantic_id_entry.delete(0, tk.END)
        self.keep_var.set(True)
        self.command_queue.put({'type': Operation.SWITCH_MODE, 'payload': {'mode': 'color'}})
        self.command_queue.put({'type': Operation.APPLY_MASK, 'payload': {'semantic_id': -2}})
        print("🔄 Reset all settings: show all objects")

    def save_scene(self):
        self.command_queue.put({'type': Operation.SAVE_SCENE})


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


class SemanticPicker:
    def __init__(self, scene_data, semantic_ids, render_mask, control_panel, cfg, vis):
        self.scene_data = scene_data
        self.semantic_ids = semantic_ids
        self.render_mask = render_mask
        self.control_panel = control_panel
        self.cfg = cfg
        self.vis = vis
        self.last_pick_time = 0
        self.frame_count = 0

    def pick_semantic(self):
        """拾取当前鼠标位置下的语义ID"""
        import time
        self.frame_count += 1
        current_time = time.time()

        # 每10帧才执行一次拾取，降低频率
        if self.frame_count % 10 != 0:
            return

        if current_time - self.last_pick_time < 0.1:  # 限制频率
            return

        self.last_pick_time = current_time

        # 获取鼠标位置
        try:
            # 使用Open3D的API获取鼠标位置
            view_control = self.vis.get_view_control()

            # 方法1：尝试获取窗口内的鼠标位置
            import open3d as o3d
            if hasattr(view_control, 'get_mouse_position'):
                mouse_pos = view_control.get_mouse_position()
                if mouse_pos is not None:
                    x, y = mouse_pos
                    self.control_panel.update_debug_info(f"Mouse at: ({x:.0f}, {y:.0f})")
                else:
                    # 如果返回None，使用窗口中心
                    width = int(self.cfg['viz_w'] * self.cfg['view_scale'])
                    height = int(self.cfg['viz_h'] * self.cfg['view_scale'])
                    x, y = width // 2, height // 2
                    self.control_panel.update_debug_info(f"Using center: ({x}, {y})")
            else:
                # 降级方案：使用窗口中心
                width = int(self.cfg['viz_w'] * self.cfg['view_scale'])
                height = int(self.cfg['viz_h'] * self.cfg['view_scale'])
                x, y = width // 2, height // 2
                self.control_panel.update_debug_info(f"Using center: ({x}, {y})")

        except Exception as e:
            self.control_panel.update_debug_info(f"Error: {str(e)[:50]}")
            return

        # 获取当前相机参数
        try:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            w2c = cam_params.extrinsic
            k = cam_params.intrinsic.intrinsic_matrix
        except:
            self.control_panel.update_debug_info("Failed to get camera params")
            return

        # 创建射线
        ray_origin, ray_direction = self.get_ray_from_screen(x, y, w2c, k)

        # 获取可见高斯点
        visible_mask = self.render_mask
        if not torch.any(visible_mask):
            self.control_panel.update_hover_info(None)
            return

        means = self.scene_data['means3D'][visible_mask]
        sems = self.semantic_ids[visible_mask]

        # 计算射线与高斯点的最近距离
        ray_origin_torch = torch.tensor(ray_origin, device='cuda', dtype=torch.float32)
        ray_direction_torch = torch.tensor(ray_direction, device='cuda', dtype=torch.float32)

        # 计算点到射线的距离
        v = means - ray_origin_torch
        proj_length = torch.sum(v * ray_direction_torch, dim=1)
        proj_points = ray_origin_torch + proj_length.unsqueeze(1) * ray_direction_torch
        distances = torch.norm(means - proj_points, dim=1)

        # 只考虑在射线前方的点
        front_mask = proj_length > 0.1  # 稍微放宽条件
        if not torch.any(front_mask):
            self.control_panel.update_hover_info(None)
            return

        distances[~front_mask] = float('inf')

        # 找到最近的点
        min_dist, min_idx = torch.min(distances, dim=0)

        # 设置阈值（调大一些）
        threshold = 0.5  # 从0.1增加到0.5
        if min_dist < threshold:
            semantic_id = sems[min_idx].item()
            position = means[min_idx].cpu().numpy()
            self.control_panel.update_hover_info(semantic_id, position)
        else:
            self.control_panel.update_hover_info(None)
            self.control_panel.update_debug_info(f"Min dist: {min_dist:.3f} > threshold")

    def get_ray_from_screen(self, x, y, w2c, k):
        """从屏幕坐标计算射线"""
        # 相机内参
        fx = k[0, 0]
        fy = k[1, 1]
        cx = k[0, 2]
        cy = k[1, 2]

        # 计算相机坐标系下的射线方向
        x_cam = (x - cx) / fx
        y_cam = (y - cy) / fy
        z_cam = 1.0

        ray_dir_cam = np.array([x_cam, y_cam, z_cam])
        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)

        # 转换到世界坐标系
        c2w = np.linalg.inv(w2c)
        ray_origin = c2w[:3, 3]
        ray_direction = c2w[:3, :3] @ ray_dir_cam
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        return ray_origin, ray_direction


def visualize(scene_path, cfg):
    global semantic_ids_global, control_panel_global

    w2c, k = load_camera(cfg, scene_path)
    if 'load_semantics' in cfg:
        load_semantics = cfg['load_semantics']
    else:
        load_semantics = False

    # 加载场景数据 + 语义ID
    scene_data, scene_depth_data, scene_semantic_data, all_w2cs, semantic_ids = load_scene_data(
        scene_path, w2c, k, load_semantics=load_semantics)

    # 提取场景里的所有唯一语义ID
    unique_semantic_ids = torch.unique(semantic_ids).cpu().numpy().tolist()
    print(f"✅ Scene loaded! Semantic IDs found: {unique_semantic_ids}")

    # 同步到全局变量
    semantic_ids_global = semantic_ids
    # 更新控制面板的ID列表
    if control_panel_global is not None:
        control_panel_global.update_semantic_ids(unique_semantic_ids)

    render_mask = torch.ones(scene_data['means3D'].shape[0], dtype=torch.bool).cuda()
    selected_semantic_id_global = -1

    vis = o3d.visualization.VisualizerWithKeyCallback()
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
    obj_delta_trans = 0.1
    obj_delta_rot = 5.0
    obj_delta_scale = 0.1
    set_camera_w2c = False

    # 创建语义拾取器
    picker = SemanticPicker(scene_data, semantic_ids, render_mask, control_panel_global, cfg, vis)

    # ===== 右键拖拽（主循环轮询，兼容WSLg，无需Open3D鼠标回调）=====
    import subprocess as _subprocess
    import re as _re

    def _fix_xauthority():
        """自动探测并设置 WSLg/X11 的 XAUTHORITY 环境变量"""
        import os as _os
        if _os.environ.get('XAUTHORITY') and _os.path.exists(_os.environ['XAUTHORITY']):
            return  # 已设置且文件存在
        uid = _os.getuid()
        candidates = [
            f'/run/user/{uid}/gdm/Xauthority',
            f'/run/user/{uid}/.Xauthority',
            f'/tmp/.Xauthority',
            _os.path.expanduser('~/.Xauthority'),
        ]
        for p in candidates:
            if _os.path.exists(p):
                _os.environ['XAUTHORITY'] = p
                return
        # WSLg 特殊路径：通过 wslg socket 目录查找
        wslg = '/mnt/wslg/runtime-dir'
        if _os.path.isdir(wslg):
            for f in _os.listdir(wslg):
                if 'auth' in f.lower() or 'xauth' in f.lower():
                    _os.environ['XAUTHORITY'] = _os.path.join(wslg, f)
                    return
        # WSLg 兜底：创建空的 ~/.Xauthority
        # WSLg 的 X11 socket 不需要真实 auth，空文件可绕过 python-xlib 检查
        fallback = _os.path.expanduser('~/.Xauthority')
        try:
            open(fallback, 'ab').close()   # touch，不覆盖
            _os.environ['XAUTHORITY'] = fallback
        except Exception:
            pass

    def _get_mouse_state_xdotool():
        """返回 (abs_x, abs_y, right_btn_pressed: bool)，失败返回 None"""
        try:
            # 优先方案：纯 python-xlib（不依赖 xdotool）
            _fix_xauthority()
            from Xlib import display as _xdisplay
            _d = _xdisplay.Display()
            _root = _d.screen().root
            _info = _root.query_pointer()
            # mask: Button1Mask=256, Button2Mask=512, Button3Mask=1024
            right_pressed = bool(_info.mask & 1024)
            x, y = _info.root_x, _info.root_y
            _d.close()
            return x, y, right_pressed
        except Exception:
            pass

        try:
            # 降级方案：xdotool（仅获取坐标，右键状态用 xinput）
            out = _subprocess.check_output(
                ['xdotool', 'getmouselocation', '--shell'],
                timeout=0.05, stderr=_subprocess.DEVNULL
            ).decode()
            x = int(_re.search(r'X=(\d+)', out).group(1))
            y = int(_re.search(r'Y=(\d+)', out).group(1))
            # 用 xinput 获取按键状态
            btn_out = _subprocess.check_output(
                ['xinput', 'query-state', 'pointer:Virtual core pointer'],
                timeout=0.05, stderr=_subprocess.DEVNULL
            ).decode()
            right_pressed = 'button[3]=down' in btn_out
            return x, y, right_pressed
        except Exception:
            return None

    def _get_o3d_win_pos_xwininfo():
        """返回 Open3D 窗口左上角绝对坐标 (win_x, win_y)，失败返回 (0,0)"""
        try:
            out = _subprocess.check_output(
                ['xwininfo', '-name', 'Open3D'],
                timeout=0.1, stderr=_subprocess.DEVNULL
            ).decode()
            ax = int(_re.search(r'Absolute upper-left X:\s*(\d+)', out).group(1))
            ay = int(_re.search(r'Absolute upper-left Y:\s*(\d+)', out).group(1))
            return ax, ay
        except Exception:
            return 0, 0

    _win_pos_cache = [0, 0, 0.0]   # [win_x, win_y, last_update_time]

    def poll_right_drag():
        """每帧调用：检测右键拖拽并发送平移指令"""
        import time as _time
        global drag_state

        result = _get_mouse_state_xdotool()
        if result is None:
            return

        abs_x, abs_y, right_pressed = result

        # 每2秒刷新一次窗口位置缓存
        now = _time.time()
        if now - _win_pos_cache[2] > 2.0:
            wx, wy = _get_o3d_win_pos_xwininfo()
            _win_pos_cache[0], _win_pos_cache[1], _win_pos_cache[2] = wx, wy, now

        # 相对于 Open3D 窗口的坐标
        rel_x = abs_x - _win_pos_cache[0]
        rel_y = abs_y - _win_pos_cache[1]

        # 判断是否在 Open3D 窗口内
        win_w = int(cfg['viz_w'] * cfg['view_scale'])
        win_h = int(cfg['viz_h'] * cfg['view_scale'])
        in_window = (0 <= rel_x < win_w) and (0 <= rel_y < win_h)

        if not right_pressed or not in_window:
            if drag_state["active"]:
                drag_state["active"] = False
                if control_panel_global is not None:
                    control_panel_global.update_drag_status(False)
            return

        if not drag_state["active"]:
            # 右键刚按下：锁定目标语义ID和起始坐标
            drag_state["active"] = True
            drag_state["last_x"] = rel_x
            drag_state["last_y"] = rel_y
            drag_state["semantic_id"] = selected_semantic_id_global
            if control_panel_global is not None:
                control_panel_global.update_drag_status(True, selected_semantic_id_global)
            print(f"\U0001f5b1\ufe0f  Right-drag started on semantic ID: {selected_semantic_id_global}")
            return

        # 拖拽进行中
        dx = rel_x - drag_state["last_x"]
        dy = rel_y - drag_state["last_y"]
        drag_state["last_x"] = rel_x
        drag_state["last_y"] = rel_y

        if dx == 0 and dy == 0:
            return

        target_id = drag_state["semantic_id"]
        if target_id == -1:
            return

        sensitivity = 0.005
        axis_lock = "XY"
        if control_panel_global is not None:
            sensitivity = control_panel_global.drag_sensitivity.get()
            axis_lock = control_panel_global.drag_axis_var.get()

        # 屏幕偏移 -> 相机坐标系偏移 -> 世界坐标系偏移
        try:
            cam_p = view_control.convert_to_pinhole_camera_parameters()
            cur_w2c = cam_p.extrinsic
            c2w_rot = np.linalg.inv(cur_w2c)[:3, :3]

            cam_delta = np.zeros(3)
            if axis_lock in ("XY", "X"):
                cam_delta[0] += dx * sensitivity
            if axis_lock in ("XY", "Y"):
                cam_delta[1] += dy * sensitivity
            if axis_lock == "Z":
                cam_delta[2] -= dy * sensitivity

            world_delta = c2w_rot @ cam_delta
        except Exception as e:
            print(f"Drag: world delta error: {e}")
            return

        command_queue.put({
            "type": Operation.OBJ_DRAG_TRANS,
            "payload": {
                "semantic_id": target_id,
                "delta": world_delta,
            }
        })



    # 添加提示信息
    print("🖱️  Mouse hover detection enabled. Move mouse over objects to see their semantic IDs.")
    print("🖱️  Right-click and drag to translate the selected semantic region.")
    print("         Axis Lock: XY(default)|X|Y|Z  -- adjustable in Control Panel.")

    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        # 执行语义拾取
        picker.pick_semantic()

        # 轮询右键拖拽状态
        poll_right_drag()

        while not command_queue.empty():
            msg = command_queue.get()
            if msg['type'] == Operation.SWITCH_MODE:
                render_mode = msg['payload']['mode']
                print(f"🎨 Switched render mode to: {render_mode}")
            elif load_semantics and msg['type'] == Operation.APPLY_MASK:
                input_semantic_id = msg['payload']['semantic_id']
                keep_selected = msg['payload'].get('keep_selected', False)
                if input_semantic_id == -2:
                    # 重置mask：显示全图
                    render_mask = torch.ones(scene_data['means3D'].shape[0], dtype=torch.bool).cuda()
                    if not keep_selected:
                        # 完全Reset：清除selected_id
                        selected_semantic_id_global = -1
                    sid_info = f"retained={selected_semantic_id_global}" if keep_selected else "cleared"
                    print(f"Show all objects | selected_id {sid_info}")
                elif input_semantic_id == -1:
                    continue
                else:
                    keep_only = msg['payload']['keep_only']
                    if keep_only:
                        # Keep Only：隐藏其他，只显示选中语义
                        render_mask = (semantic_ids.squeeze() == input_semantic_id)
                        selected_semantic_id_global = input_semantic_id
                        print(f"Keep Only ID {input_semantic_id} | Visible Gaussians: {torch.sum(render_mask)}")
                    else:
                        # 不勾选Keep：全图显示，仅设置 selected_id 用于后续拖拽
                        render_mask = torch.ones(scene_data['means3D'].shape[0], dtype=torch.bool).cuda()
                        selected_semantic_id_global = input_semantic_id
                        print(f"Selected ID {input_semantic_id} for dragging | Full map visible")
                picker.render_mask = render_mask
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
            elif msg['type'] == Operation.OBJ_TRANS:
                if selected_semantic_id_global == -1:
                    print("⚠️ No object selected! Please input a valid Semantic ID first.")
                    messagebox.showwarning("No Object Selected", "Please select a Semantic ID first!")
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
                print(f"📦 Moved object ID {selected_semantic_id_global} along {dir} by {dist}")
            elif msg['type'] == Operation.OBJ_ROT:
                if selected_semantic_id_global == -1:
                    print("⚠️ No object selected! Please input a valid Semantic ID first.")
                    messagebox.showwarning("No Object Selected", "Please select a Semantic ID first!")
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
                print(
                    f"🔄 Rotated object ID {selected_semantic_id_global} around {dir} by {factor * obj_delta_rot} degrees")
            elif msg['type'] == Operation.OBJ_SCALE:
                if selected_semantic_id_global == -1:
                    print("⚠️ No object selected! Please input a valid Semantic ID first.")
                    messagebox.showwarning("No Object Selected", "Please select a Semantic ID first!")
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
                print(f"📐 Scaled object ID {selected_semantic_id_global} along {dir} by {scale}x")
            elif msg['type'] == Operation.OBJ_DRAG_TRANS:
                # 右键拖拽平移：将世界坐标偏移量直接加到语义区域的means3D
                drag_id = msg['payload']['semantic_id']
                delta = msg['payload']['delta']  # numpy (3,)
                if drag_id != -1:
                    drag_mask = (semantic_ids.squeeze() == drag_id)
                    delta_torch = torch.tensor(delta, device='cuda', dtype=torch.float32)
                    scene_data['means3D'][drag_mask] += delta_torch
            elif msg['type'] == Operation.SAVE_SCENE:
                print("💾 Saving edited scene...")
                save_path = os.path.join(os.path.dirname(scene_path), "params_edited.npz")
                params_to_save = {}
                for k, v in scene_data.items():
                    if isinstance(v, torch.Tensor):
                        params_to_save[k] = v.detach().cpu().numpy()
                    else:
                        params_to_save[k] = v
                params_to_save['semantic_ids'] = semantic_ids.detach().cpu().numpy()
                np.savez(save_path, **params_to_save)
                print(f"✅ Saved edited scene to: {save_path}")
                messagebox.showinfo("Save Success", f"Edited scene saved to:\n{save_path}")

        global edit_semantic_id, edit_transform
        if edit_semantic_id != -1 and np.any(edit_transform != np.eye(4)):
            mask = semantic_ids.squeeze() == edit_semantic_id
            means = scene_data['means3D'].clone()
            homo = torch.cat([means[mask], torch.ones(means[mask].shape[0], 1, device='cuda')], dim=1)
            trans = torch.tensor(edit_transform, device='cuda', dtype=torch.float32)
            means[mask] = (homo @ trans.T)[:, :3]
            scene_data['means3D'] = means

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

    print(f"📂 Loading scene from: {scene_path}")
    viz_cfg = experiment.config["viz"]

    root = tk.Tk()
    control_panel_global = ControlPanel(root, command_queue, viz_cfg, width=20, height=200)

    thread = Thread(target=visualize, args=(scene_path, viz_cfg))
    thread.daemon = True
    thread.start()

    root.mainloop()