from os.path import join as p_join

primary_device = "cuda:0"
seed = "seed0"
group_name = "ScanNet_postopt"
scene_name = "scene0000_00"  # 改成你自己的场景名
param_name = f"{scene_name}_{seed}"
run_name = f"postopt_{param_name}"
param_ckpt_path = f"./experiments/ScanNet/{param_name}/params.npz"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=0,
    primary_device=primary_device,
    mean_sq_dist_method="projective",
    report_iter_progress=False,
    use_wandb=False,
    wandb=dict(
        entity="my-project",
        project="SGS-SLAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        dataset_name="scannet",  # 重要：改为 scannet
        basedir="./datasets/scannet", # 重要：改为你的 scannet 路径
        gradslam_data_cfg="./configs/data/scannet.yaml",
        sequence=scene_name,
        desired_image_height=240,
        desired_image_width=320,
        start=0,
        end=-1,
        stride=20,
        num_frames=-1,
        eval_stride=5,
        eval_num_frames=-1,
        param_ckpt_path=param_ckpt_path,
        load_semantics=True,
        num_semantic_classes=20  # ScanNet 是 20 类
    ),
    train=dict(
        num_iters_mapping=15000,
        sil_thres=0.5,
        use_sil_for_loss=True,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            seg=0.1,
        ),
        lrs_mapping=dict(
            means3D=0.00032,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.005,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
            semantic_colors=0.0025,
        ),
        lrs_mapping_means3D_final=0.0000032,
        lr_delay_mult=0.01,
        use_gaussian_splatting_densification=True,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=15000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=True,
            reset_opacities_every=3000,
        ),
    ),
    viz=dict(
        render_mode='semantic_color',
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=False,
        viz_w=840, viz_h=476,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=True,
        scene_name=scene_name,
        load_semantics=True,
    ),
)