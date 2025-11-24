#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, PearsonDepthLoss, edge_aware_normal_loss
from gaussian_renderer import render, render_feature, network_gui
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.cameras import Camera
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F
from lpipsPyTorch import lpips

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree, dataset.semantic_dim)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_scale_for_log = 0.0
    ema_semantic_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_for_log = 0.0

    depth_loss, normal_loss, scale_loss = None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_plane=iteration>opt.geo_regularization_from_iter)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # image loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        Ll1 = l1_loss(image, gt_image)

        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        
        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            scale_loss = opt.scale_loss_weight * min_scale_loss.mean()
            loss += (scale_loss)
        
        # semantic distillation
        gt_semantic_map = viewpoint_cam.semantic_map.cuda()
        rendered_semantic_map = render_feature(viewpoint_cam, gaussians, pipe, background)

        target = F.softmax(gt_semantic_map, dim=0)              # [150, H, W]
        pred = F.log_softmax(rendered_semantic_map, dim=0)      # [150, H, W]
        semantic_soft_loss = opt.semantic_soft_loss_weight * F.kl_div(pred, target, reduction='batchmean')
        
        hard_labels = torch.argmax(gt_semantic_map, dim=0)      # [H, W]
        semantic_hard_loss = opt.semantic_hard_loss_weight * F.cross_entropy(rendered_semantic_map.unsqueeze(0), hard_labels.unsqueeze(0))
        
        semantic_loss = (semantic_soft_loss + semantic_hard_loss)
        loss += (semantic_loss)

        # geo regularization
        if iteration > opt.geo_regularization_from_iter:
            ## depth consistency
            rend_depth = render_pkg["plane_depth"][0]
            gt_depth = viewpoint_cam.depth_map.cuda()

            sem_mask = torch.argmax(gt_semantic_map, dim=0).float()   # [H, W]
    
            edge_mask_x = (sem_mask[:, :-1] != sem_mask[:, 1:])       # [H, W-1]
            edge_mask_y = (sem_mask[:-1, :] != sem_mask[1:, :])       # [H-1, W]

            edge_mask = torch.zeros_like(sem_mask, dtype=torch.bool)  # [H, W]
            edge_mask[:, :-1] |= edge_mask_x
            edge_mask[:-1, :] |= edge_mask_y

            non_edge_mask = ~edge_mask

            depth_loss = opt.depth_loss_weight * PearsonDepthLoss(rend_depth[non_edge_mask], gt_depth[non_edge_mask])
            loss += (depth_loss)

            ## muti-faceted normal consistency
            rend_normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            normal_error = (1 - (rend_normal * depth_normal).sum(dim=0))[None]
            normal_consitency_loss = opt.normal_consistency_loss_weight * (normal_error).mean()

            edge_aware_normal_separation_loss = opt.edge_aware_normal_loss_weight * edge_aware_normal_loss(rend_normal, gt_semantic_map)

            normal_loss = normal_consitency_loss + edge_aware_normal_separation_loss
            loss += (normal_loss)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_semantic_for_log = 0.4 * semantic_loss.item() + 0.6 * ema_semantic_for_log
            ema_scale_for_log = 0.4 * scale_loss.item() if scale_loss is not None else 0.0 + 0.6 * ema_scale_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_normal_for_log
            ema_depth_for_log = 0.4 * depth_loss.item() if depth_loss is not None else 0.0 + 0.6 * ema_depth_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Semantic": f"{ema_semantic_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Depth": f"{ema_depth_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/scale_loss', ema_scale_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/semantic_loss', ema_semantic_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/depth_loss', ema_depth_for_log, iteration)

            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = out["render"]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        try:
                            from utils.general_utils import colormap
                            depth = out["plane_depth"]
                            norm = depth.max()
                            depth = depth / norm
                            depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                            
                            rend_normal = out["rendered_normal"] * 0.5 + 0.5
                            surf_normal = out["depth_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)

                        except:
                            pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # l1_d_test += l1_loss(image_d, gt_image_depth).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image)
                    lpips_test += lpips(image, gt_image, net_type='vgg').item()


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                # l1_d_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {:0.4f} PSNR {:0.2f} SSIM {:0.3f} LPIPS {:0.3f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000, 3000, 4000, 5000, 6000, 7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 7000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5000, 7000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
