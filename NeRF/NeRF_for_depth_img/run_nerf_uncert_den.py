import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_blender import load_blender_data
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        #print("minjae test1: ", embedded_dirs)
        #print("minjae test2: ", embedded_dirs.shape)
        embedded = torch.cat([embedded, embedded_dirs], -1)
        #print("minjae test3: ", embedded)
        #print("minjae test4: ", embedded.shape)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'uncert_map', 'alpha_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    uncerts = []
    depths = []
    #################################################################
    if savedir is not None:
        txt_filename = os.path.join(savedir, 'z_result_summary.txt')
        txt_file = open(txt_filename, 'w')

    all_losses = []
    all_abs_rel = []
    all_sq_rel = []
    all_rmse = []
    all_rmse_log = []
    all_log10 = []
    all_del1 = []
    all_del2 = []
    all_del3 = []
    #################################################################
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, uncert, alpha, depth, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        uncerts.append(uncert.cpu().numpy())
        depths.append(depth.cpu().numpy())

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
        ###############################################################################################

        if i == 0 :
            print("")
        if gt_imgs is not None:
            if i >= len(gt_imgs) or gt_imgs[i] is None:
                print(f"Warning: No ground truth image available for index {i}. Skipping comparison.")
                continue
            txt_img_loss = img2mse(rgb, gt_imgs[i]) 
            all_losses.append(txt_img_loss.item())


            a  = depth * 1000 # pred depth value
            b = gt_imgs[i] * 1000 # original gt depth value
            b = b[:, :, 0] # [468, 624, 3] to [468, 624]

            txt_abs_rel = calculate_abs_rel(a,b)
            txt_sq_rel = calculate_sq_rel(a,b)
            txt_rmse = calculate_rmse(a,b)
            txt_rmse_log = calculate_rmse_log(a,b)
            txt_log10 = calculate_avg_log10_error(a,b)
            txt_del1 = calculate_threshold_accuracy(a,b,1)
            txt_del2 = calculate_threshold_accuracy(a,b,2)
            txt_del3 = calculate_threshold_accuracy(a,b,3)

            all_abs_rel.append(txt_abs_rel)
            all_sq_rel.append(txt_sq_rel)
            all_rmse.append(txt_rmse)
            all_rmse_log.append(txt_rmse_log)
            all_log10.append(txt_log10)
            all_del1.append(txt_del1)
            all_del2.append(txt_del2)
            all_del3.append(txt_del3)


            if gt_imgs is not None:
                gt_img8 = to8b_for_GT(gt_imgs[i])  # gt_imgs[i] 이미지를 8비트 형식으로 변환
                gt_filename = os.path.join(savedir, 'GT_{:03d}.png'.format(i))  # 저장할 파일명 설정
                imageio.imwrite(gt_filename, gt_img8)  # 이미지 파일로 저장
            ###############################################################################################
            if savedir is not None:
                ###############################################################################################
                base_filename = '{:03d}.png'.format(i) 
                line = f"{base_filename}, {txt_abs_rel:.4f}, {txt_sq_rel:.4f}, {txt_rmse:.4f}, {txt_rmse_log:.4f}, {txt_log10:.4f}, {txt_del1:.4f}, {txt_del2:.4f}, {txt_del3:.4f}\n"   
                txt_file.write(line)


                depth_raw = to8b_for_depth2(depths[-1])
                filename = os.path.join(savedir, 'Depth_raw{:03d}.png'.format(i))
                imageio.imwrite(filename, depth_raw)

                depth8 = to8b_for_depth(depths[-1])
                filename = os.path.join(savedir, 'Depth_{:03d}.png'.format(i))
                imageio.imwrite(filename, depth8)

                uncert8 = to8b(uncerts[-1]/ np.max(uncerts[-1]))
                filename = os.path.join(savedir, 'Uncert_{:03d}.png'.format(i))
                imageio.imwrite(filename, uncert8)

            torch.cuda.empty_cache()

    if savedir is not None:
        txt_file.close()
        avg_loss = sum(all_losses) / len(all_losses)
        avg_abs_rel = sum(all_abs_rel) / len(all_abs_rel)
        avg_sq_rel = sum(all_sq_rel) / len(all_sq_rel)
        avg_rmse = sum(all_rmse) / len(all_rmse)
        avg_rmse_log = sum(all_rmse_log) / len(all_rmse_log)
        avg_log10 = sum(all_log10) / len(all_log10)
        avg_del1 = sum(all_del1) / len(all_del1)
        avg_del2 = sum(all_del2) / len(all_del2)
        avg_del3 = sum(all_del3) / len(all_del3)
        with open(txt_filename, 'r') as f:
            data = f.read()
        with open(txt_filename, 'w') as f:
            f.write(f"Avg Loss: {avg_loss:.4f}, Avg abs_rel: {avg_abs_rel:.4f}, Avg sq_rel: {avg_sq_rel:.4f}, Avg rmse: {avg_rmse:.4f}, Avg rmselog: {avg_rmse_log:.4f}, Avg log10: {avg_log10:.4f}\n")
            f.write(f"Avg del1: {avg_del1:.4f}, Avg del2: {avg_del2:.4f}, Avg del3: {avg_del3:.4f}\n")
            f.write(data)
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    uncerts = np.stack(uncerts, 0)
    return rgbs, disps, uncerts, None


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, beta_min=args.beta_min)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, beta_min=args.beta_min)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        i_train = ckpt['i_train']
        # i_holdout = ckpt['i_holdout']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    else:
        i_train = None
        # i_holdout = None
    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, i_train #, i_holdout


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time. samples depth value
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1] # distance of each samples
    
    #dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    ##########################################################################
    last_column = dists[:, -1]
    dists = torch.cat([dists, last_column.unsqueeze(-1)], dim=-1)
    ########################################################################
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]


    uncert_map = torch.sum(z_vals * z_vals * dists * dists * raw[..., 4], -1) # color distance raw -> z_val distance raw
    depth_map = torch.sum(weights * z_vals, -1)
    ##############################################################################################
    # depth_map = torch.sum(weights * dists_for_depth -1)
    ##############################################################################################
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, uncert_map, F.relu(raw[...,3] + noise).mean(-1) # last term is alpha map



def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, uncert_map, alpha_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, uncert_map_0, alpha_map_0 = rgb_map, disp_map, acc_map, depth_map, uncert_map, alpha_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, uncert_map, alpha_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'uncert_map' : uncert_map, 'alpha_map' : alpha_map, 'depth_map' : depth_map}
    ret['raw'] = raw
    ret['weights'] = weights
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['uncert0'] = uncert_map_0
        ret['alpha0'] = alpha_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def choose_new_k(H, W, focal, batch_rays, k, **render_kwargs_train):
    
    pres = []
    posts = []
    N = H*W
    n = batch_rays.shape[1] // N
    for i in range(n):
        with torch.no_grad():
            rgb, disp, acc, uncert, alpha, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays[:,i*N:i*N+N,:],  verbose=True, retraw=True,  **render_kwargs_train)

        uncert_render = uncert.reshape(-1, H*W, 1) + 1e-9
        uncert_pts = extras['raw'][...,-1].reshape(-1, H*W, args.N_samples + args.N_importance) + 1e-9
        weight_pts = extras['weights'].reshape(-1, H*W, args.N_samples + args.N_importance)

        pre = uncert_pts.sum([1,2])
        post = (1. / (1. / uncert_pts + weight_pts * weight_pts / uncert_render)).sum([1,2])
        pres.append(pre)
        posts.append(post)
    
    pres = torch.cat(pres, 0)
    posts = torch.cat(posts, 0)
    index = torch.topk(pres-posts, k)[1].cpu().numpy()

    return index




def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    ################################################################################################################33
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')



    ################################################################################################################33

    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')

    #########################################################################################################
    # parser.add_argument("--render_factor", type=int, default=0, 
    #                     help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    ########################################################################################################

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender')
    ########################################################################################################
    # parser.add_argument("--testskip", type=int, default=8, 
    #                     help='will load 1/N images from test/val sets')
    ########################################################################################################

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    ########################################################################################################
    # parser.add_argument("--llffhold", type=int, default=8, 
    #                     help='will take every 1/N images as LLFF test set, paper uses 8')
    ########################################################################################################

    # logging/saving options
    ########################################################################################################
    parser.add_argument("--testskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets')
    parser.add_argument("--llffhold", type=int, default=2, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--i_print",   type=int, default=1000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=10000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=50000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    ########################################################################################################

    
    ########################################################################################################
    parser.add_argument("--i_all",   type=int, default=50000) # Training iterations, 500000 for full-res nerfs
    parser.add_argument('--active_iter', type=int, nargs='+', default=[50]) 
    parser.add_argument("--init_image",   type=int, default=20) # initial number of images, only for llff dataset
    parser.add_argument("--choose_k",   type=int, default=0) # The number of new captured data for each active iter
    parser.add_argument("--beta_min",   type=float, default=0.01) # Minimun value for uncertainty
    parser.add_argument("--w",   type=float, default=0.1) # Strength for regularization as in Eq.(11)
    parser.add_argument("--ds_rate",   type=int, default=2) # Quality-efficiency trade-off factor as in Sec. 5.2

    ########################################################################################################
    return parser


def train():

    # Load data

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_trainhold = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        # i_holdout = i_trainhold[args.init_image:]
        i_train = i_trainhold[:args.init_image]

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # near = 2.
        # far = 6.
        near = 0.10000000149011612
        far = 8.8439249992370609

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, i_train_load = create_nerf(args)
    global_step = start

    if i_train_load is not None:
        i_train = i_train_load
        # i_holdout = i_holdout_load

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _, uncerts, alphas = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video_uncert.mp4'), to8b(uncerts / np.max(uncerts)), fps=30, quality=8)
            return


    N_rand = args.N_rand

    print('get rays')
    rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
    print('done, concats')

    # prepare training rays
    rays_rgb_all = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    rays_rgb_all = np.transpose(rays_rgb_all, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb_train = np.stack([rays_rgb_all[i] for i in i_train], 0) # train images only
    rays_rgb_train = np.reshape(rays_rgb_train, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb_train = rays_rgb_train.astype(np.float32)

    print('shuffle rays')
    np.random.shuffle(rays_rgb_train)

    print('done')
    i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    
    rays_rgb_train = torch.Tensor(rays_rgb_train).to(device)


    N_iters = args.i_all + 1
    print('Begin')
    print('TRAIN views are', i_train)
    # print('HOLDOUT views are', i_holdout)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    start = start + 1
    num_test_nomal = 0
    num_test_uncert = 0
    for i in trange(start, N_iters):


        time0 = time.time()

        # Sample random ray batch
        batch = rays_rgb_train[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += N_rand
        if i_batch >= rays_rgb_train.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb_train.shape[0])
            rays_rgb_train = rays_rgb_train[rand_idx]
            i_batch = 0

        #####  Core optimization loop  #####
        rgb, disp, acc, uncert, alpha, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        
        print("target_s.shape: ", target_s.shape)
        print("depth_s.shape: ", depth.shape)
        print("uncert shape: ", uncert.shape)
        # print("target_s: ", target_s)
        # print("depth_s: ", depth)

        print("target_s max: ", torch.max(target_s))
        print("target_s min: ", torch.min(target_s))
        print("target_s mean:", torch.mean(target_s))

        print("depth_s max: ", torch.max(depth))
        print("depth_s min: ", torch.min(depth))
        print("depth_mean:", torch.mean(depth))

        gt_depth_target_s = torch.mean(target_s, dim=1)

        optimizer.zero_grad()




        psnr = mse2psnr(img2mse(depth, gt_depth_target_s))
        loss_imgmse = img2mse(depth, gt_depth_target_s)
        loss_imguncert = img2mse_uncert(depth, gt_depth_target_s, uncert, alpha, args.w)

        #############################################################################################################
        if i < 2000: # pretraining
            img_loss = loss_imgmse
        elif 2000<=i and i<2200:
            img_loss = loss_imgmse + 1e-6 * loss_imguncert
        elif 3000<=i and i<3200:
            img_loss = loss_imgmse + 1e-5 * loss_imguncert
        elif 4000<=i and i<4200:
            img_loss = loss_imgmse + 1e-4 * loss_imguncert
        elif 5000<=i and i<5200:
            img_loss = loss_imgmse + 1e-3 * loss_imguncert
        elif 6000<=i and i<6200:
            img_loss = loss_imgmse + 1e-2 * loss_imguncert
        elif 8000<=i and i<8200:
            img_loss = loss_imgmse + 1e-2 * loss_imguncert
        # elif 10000<=i and i<15000:
        #     img_loss = loss_imgmse + loss_imguncert            
        else:
            img_loss = loss_imgmse
        #############################################################################################################

        loss = img_loss
        print("loss : ", loss)
        ## loss code

        ##############################################
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        results_file = os.path.join(basedir, expname, 'results.txt')
        if i == N_iters-1:
            with open(results_file, 'a') as file: 
                file.write(f"Loss: {loss}, PSNR: {psnr}\n")
        ##############################################

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img2mse(extras['rgb0'], target_s))

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # new_lrate = args.lrate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate


        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'i_train': i_train,
                # 'i_holdout': i_holdout,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                # 'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)


        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps, uncerts, alphas = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            # print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, 'spiral_{:06d}_'.format(i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=15, quality=8)
            # imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'uncert.mp4', to8b(uncerts / np.max(uncerts)), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')
    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    print(args)
    train()

