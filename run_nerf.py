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
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
import torchvision.transforms as transforms

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
        embedded = torch.cat([embedded, embedded_dirs], -1)

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


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
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
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

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

    ################################################################################################################
    # k_extract = ['rgb_map', 'disp_map', 'acc_map']
    #k_extract = ['rgb_map', 'disp_map', 'acc_map', 'uncert_map', 'alpha_map']
    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'uncert_map', 'alpha_map', 'lam_map', 'S_A', 'U_A']
    ################################################################################################################
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    #################################################################
    uncerts = []
    if savedir is not None:
        txt_filename = os.path.join(savedir, 'info.txt')
        txt_file = open(txt_filename, 'w')

    all_losses = []
    all_psnrs = []
    all_ssims = []
    all_lpips = []
    #################################################################

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        #################################################################
        # rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        # rgb, disp, acc, uncert, alpha, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgb, disp, acc, uncert, alpha, lam, S_A, U_A, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        uncerts.append(uncert.cpu().numpy())
        #################################################################
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
        ###############################################################################################
        txt_img_loss = img2mse(rgb, gt_imgs[i]) 
        txt_psnr = mse2psnr(txt_img_loss)
        txt_ssim = ssim_value(rgb, gt_imgs[i])
        txt_lpips = lpips_value(rgb, gt_imgs[i])

        all_ssims.append(txt_ssim)
        all_lpips.append(txt_lpips)
        all_losses.append(txt_img_loss.item())
        all_psnrs.append(txt_psnr.item())
        ###############################################################################################

        if savedir is not None:
            ###############################################################################################
            base_filename = '{:03d}.png'.format(i) 
            line = f"{base_filename}, {txt_img_loss.item():.5f}, {txt_psnr.item():.5f}, {txt_ssim:.5f}, {txt_lpips:.5f}\n"   
            txt_file.write(line)

            uncert8 = to8b(uncerts[-1]/ np.max(uncerts[-1]))
            filename = os.path.join(savedir, 'Uncert_{:03d}.png'.format(i))
            imageio.imwrite(filename, uncert8)

            # gt_imgs8[i] = to8b(gt_imgs[i][-1])
            # filename = os.path.join(savedir, '{:03d}_GT.png'.format(i))
            # imageio.imwrite(filename, gt_imgs8[i])
            ###############################################################################################

            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
        
    if savedir is not None:
        txt_file.close()
        avg_loss = sum(all_losses) / len(all_losses)
        avg_psnr = sum(all_psnrs) / len(all_psnrs)
        avg_ssim = sum(all_ssims) / len(all_ssims)
        avg_lpips = sum(all_lpips) / len(all_lpips)

        with open(txt_filename, 'r') as f:
            data = f.read()
        with open(txt_filename, 'w') as f:
            f.write(f"Average Loss: {avg_loss:.5f}, Average PSNR: {avg_psnr:.5f}, Average SSIM: {avg_ssim:.5f}, Average LPIPS: {avg_lpips:.5f}\n")
            f.write(data)


    ###############################################################################################
    # return rgbs, disps
    uncerts = np.stack(uncerts, 0)
    return rgbs, disps, uncerts, None
    ###############################################################################################

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
    ###############################################################################################
    # model = NeRF(D=args.netdepth, W=args.netwidth,
    #              input_ch=input_ch, output_ch=output_ch, skips=skips,
    #              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, beta_min=args.beta_min).to(device)
    ##############################################################################################
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        
        # model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
        #                   input_ch=input_ch, output_ch=output_ch, skips=skips,
        #                   input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        ##############################################################################################
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, beta_min=args.beta_min).to(device)
        ##############################################################################################
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
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

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

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    # #################################################################################
    # raw2alpha = lambda raw, dists, act_fn=F.relu: act_fn(raw)*dists

    #################################################################################

    dists = z_vals[...,1:] - z_vals[...,:-1] 
    # dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    last_column = dists[:, -1]
    dists = torch.cat([dists, last_column.unsqueeze(-1)], dim=-1)
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
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3] #RGB MAP = 기존 방식이랑 동일


    
    #####################################################################################
    # raw[...,4] : uncertainty of density   
    # raw[...,3] : density mean value 
    # raw[...,:3] : color mean value 
    
    distXdensity = dists * F.relu(raw[...,3])
    distXdensity_sum = torch.cat([torch.zeros((distXdensity.shape[0], 1)), -distXdensity], axis=-1)
    distXdensity_sum = distXdensity_sum.cumsum(axis=-1)
    distXdensity_sum[:, 0] = 0. # init mu_T1
    mu_Ti = distXdensity_sum[:, :-1]
    mu_Tip1 = distXdensity_sum[:, 1:]
    Ti = torch.exp(mu_Ti)
    
    dist2Xuncertainty = dists* dists* raw[...,4] # raw[..., 4] = var > 0.01
    dist2Xuncertainty_sum = torch.cat([torch.zeros((dist2Xuncertainty.shape[0], 1)), dist2Xuncertainty], axis=-1)
    dist2Xuncertainty_sum = dist2Xuncertainty_sum.cumsum(axis=-1)
    dist2Xuncertainty_sum[:, 0] = 0.01 #init sigma2_T1
    dist2Xuncertainty_sum += 1e-9
    sigma2_Ti = dist2Xuncertainty_sum[:, :-1]
    sigma2_Tip1 = dist2Xuncertainty_sum[:, 1:]
    print("uncertainty: ", torch.mean(dist2Xuncertainty))
    
    U_ti = mu_Ti
    U_ti_p1 = mu_Tip1
    S2_ti = sigma2_Ti
    S2_ti_p1 = sigma2_Tip1

    ##### dist 없애지 않은코드 ##############
    # corr = dist2Xuncertainty_sum[:,:-1] / torch.sqrt( dist2Xuncertainty_sum[:,:-1]**2 + dist2Xuncertainty_sum[:,:-1] * dist2Xuncertainty )
    # corr = torch.clamp(corr, min=0, max=1.-1e-9)
    # sigma2_m = S2_ti_p1 + S2_ti - 2.*corr*torch.sqrt(S2_ti_p1 * S2_ti)
    sigma2_m = torch.square(torch.sqrt(S2_ti_p1) - torch.sqrt(S2_ti))
    sigma2_m = torch.clamp(sigma2_m, min=1e-9, max=None)
    Temp1 = sigma2_m /(dists*dists*raw[...,4]+1e-9) # left term of si
    Temp2 = torch.exp(U_ti_p1 + 0.5*S2_ti_p1) + torch.exp(U_ti + 0.5*S2_ti) # right term of si

    S_i = Temp1 * Temp2 # s_{i}
    S_ai = (dists*dists*raw[...,4])/(2 * torch.sqrt(sigma2_m) + 1e-9) # sigma_{alpha_i}
    print("S_ai: ", torch.mean(S_ai))
    U_ai = U_ti + (0.5*S2_ti) - (2*torch.log(dists+1e-9) + torch.log(0.5*raw[...,4] + 1e-9))
    U_ai += torch.log(torch.exp(-dists*F.relu(raw[...,3]) + 0.5 * dists*dists*raw[...,4])*S2_ti_p1 + S2_ti + 1e-9) - 0.5 * S_ai *S_ai
    print("U_ai: ", torch.mean(U_ai))
    ##### dist 없애지 않은코드 ##############

    ### lamda - C = A(r)
    S_ai_copy = S_ai.unsqueeze(-1).expand(-1, -1, 3) # fitting tensor shape
    U_ai_copy = U_ai.unsqueeze(-1).expand(-1, -1, 3)
    
    Ar_map = torch.sum( (S_i[...,None] - weights[...,None]) * rgb,-2)
    # TempA = torch.sum( torch.exp(2 * (U_ai_copy + torch.log(rgb+1e-10)) + S_ai_copy * S_ai_copy) * torch.exp(S_ai_copy * S_ai_copy -1), -2 )
    # TempB = torch.sum( torch.exp(U_ai_copy + torch.log(rgb+1e-10) + 0.5 * S_ai_copy * S_ai_copy) ,-2)
    TempA = torch.sum( rgb * rgb * torch.exp(2 * U_ai_copy + S_ai_copy * S_ai_copy) * (torch.exp(S_ai_copy * S_ai_copy) -1), -2 ) # rgb 값 밖으로 꺼냄    
    TempB = torch.sum( rgb * torch.exp(U_ai_copy + 0.5 * S_ai_copy * S_ai_copy) ,-2)
    # TempA = torch.clamp(TempA, min=None, max=1e+3)
    # TempB = torch.clamp(TempB, min=None, max=1e+6)
    S2_A = torch.log( (TempA)/(TempB * TempB + 1e-9) +1)
    # S2_A = torch.log(TempA + TempB*TempB + 1e-9) - 2*torch.log(torch.abs(TempB) + 1e-9)
    U_A = torch.log(TempB + 1e-9) - 0.5*S2_A
    bias_S2_A = 1e-4
    S2_A = S2_A + bias_S2_A

    # if torch.isnan(torch.mean(raw[...,4])):
    #     print("Mean value is NaN. Exiting the program.")
    #     sys.exit()

    print("1TempA: ", torch.mean(TempA))
    print("1TempB: ", torch.mean(TempB))
    # print("1S_A: ", torch.mean(S_A))
    # print("1U_A: ", torch.mean(U_A))
    # print("1AR: ", torch.mean(Ar_map))
    # print("1RGB: ", torch.mean(rgb_map))


    # print("2S_A: ", torch.max(S_A))
    # print("2U_A: ", torch.max(U_A))

    # print("3S_A: ", torch.min(S_A))
    # print("3U_A: ", torch.min(U_A))


    # uncert_map = S_A
    uncert_map = torch.sum(weights[...,None] * raw[...,4].unsqueeze(-1).expand(-1, -1, 3), -2)
    lam_map = torch.sum( S_i[...,None] * rgb, -2)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    

    if white_bkgd:
        # print("okay")
        rgb_map = rgb_map + (1.-acc_map[...,None])

    #####################################################################################################################
    #return rgb_map, disp_map, acc_map, weights, depth_map
    # return rgb_map, disp_map, acc_map, weights, depth_map, uncert_map, F.relu(raw[...,3] + noise).mean(-1) # last term is alpha map
    return rgb_map, disp_map, acc_map, weights, depth_map, uncert_map, F.relu(raw[...,3] + noise).mean(-1), lam_map, S2_A, U_A 
    #####################################################################################################################

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


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    #########################################################################################################
    # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    # rgb_map, disp_map, acc_map, weights, depth_map, uncert_map, alpha_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    rgb_map, disp_map, acc_map, weights, depth_map, uncert_map, alpha_map, lam_map, S_A, U_A \
    = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
    #########################################################################################################

    if N_importance > 0:
        #########################################################################################################
        #rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        # rgb_map_0, disp_map_0, acc_map_0, uncert_map_0, alpha_map_0 = rgb_map, disp_map, acc_map, uncert_map, alpha_map
        rgb_map_0, disp_map_0, acc_map_0, uncert_map_0, alpha_map_0, lam_map_0, S_A_0, U_A_0 = rgb_map, disp_map, acc_map, uncert_map, alpha_map, lam_map, S_A, U_A
        #########################################################################################################

        ########################################## fine net 문제가 여긴가 해서 sampling 방법 그냥 coarse랑 똑같이 해봄 ###########################
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        ########################################## fine net 문제가 여긴가 해서 sampling 방법 그냥 coarse랑 똑같이 해봄 ###########################



        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        #########################################################################################################
        # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        # rgb_map, disp_map, acc_map, weights, depth_map, uncert_map, alpha_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        rgb_map, disp_map, acc_map, weights, depth_map, uncert_map, alpha_map, lam_map, S_A, U_A = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        #########################################################################################################

    #########################################################################################################
    # ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    #ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'uncert_map' : uncert_map, 'alpha_map' : alpha_map}
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'uncert_map' : uncert_map, 'alpha_map' : alpha_map, 'lam_map' : lam_map, "S_A" : S_A, 'U_A' : U_A}
    
    # ret['raw'] = raw
    ret['weights'] = weights
    #########################################################################################################
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ########################################## fine net 문제가 여긴가 해서 sampling 방법 그냥 coarse랑 똑같이 해봄 ###########################
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ########################################## fine net 문제가 여긴가 해서 sampling 방법 그냥 coarse랑 똑같이 해봄 ###########################

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
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
    # parser.add_argument("--lrate", type=float, default=5e-4, # 4 
    #                     help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')

    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
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
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    ########################################################################################################   
    # parser.add_argument("--testskip", type=int, default=8, 
    #                     help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--testskip", type=int, default=2, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ########################################################################################################   

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
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
    # 이건 보니 config file에서 영향을 주는거고, 그리고 이걸 낮추면, test 이미지는 많아지지만 train갯수 작아짐
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    ########################################################################################################  

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=10000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=20000000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=2000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=100000, 
                        help='frequency of render_poses video saving')

    ########################################################################################################   
    # parser.add_argument("--lrate", type=float, default=5e-4, # 원래 4 
    #                     help='learning rate')
    parser.add_argument("--lrate", type=float, default=5e-4, # 원래 5e-4 
                        help='learning rate')
    parser.add_argument("--beta_min",   type=float, default=0.001) # 얘도 중요한 파라미터임, 건들여보자
    parser.add_argument("--w",   type=float, default=0.01) 

    parser.add_argument("--total_epoch",   type=int, default=200000, 
                        help='total_epoch')
    ########################################################################################################    
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
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
            i_test = np.arange(images.shape[0])[::args.llffhold] # index test and val
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if # index train
                        (i not in i_test and i not in i_val)])
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
        
        near = 2.
        far = 6.


        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]


    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

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
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

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

            ##################################################################################################################
            #rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            rgbs, _, uncerts, alphas = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            # imageio.mimwrite(os.path.join(testsavedir, 'video_uncert.mp4'), to8b(uncerts / np.max(uncerts)), fps=30, quality=8)
            ##################################################################################################################

            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    # if use_batching:
    #     images = torch.Tensor(images).to(device)
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    ################################################################
    # N_iters = 200000 + 1
    N_iters = args.total_epoch + 11
    ###################################################################
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    minjae_test_num = 0
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        #############################################################################################################
        # rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
        #                                         verbose=i < 10, retraw=True,
        #                                         **render_kwargs_train)
        # rgb, disp, acc, uncert, alpha, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
        #                                         verbose=i < 10, retraw=True,
        #                                         **render_kwargs_train)
        rgb, disp, acc, uncert, alpha, lam_map, S_A, U_A, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        #############################################################################################################


        optimizer.zero_grad()

        # this value is important term or used term for loss
        print("rgb: ", torch.mean(rgb))
        print("lam: ", torch.mean(lam_map))
        print("S_A: ", torch.max(S_A))
        print("U_A: ", torch.mean(U_A))
        print("target_s: ", torch.mean(target_s))

        # val 1~4 mean is term of loss element
        print("val 1: ", torch.mean(torch.log(torch.abs(lam_map - target_s) + 1e-10)))
        print("val 2: ", torch.mean(0.5 * torch.log(S_A + 1e-9)))
        print("val 3: ", torch.mean( (torch.log(torch.abs(lam_map - target_s) + 1e-9) - U_A)**2/(2*S_A + 1e-9)))
        print("val 4: ", torch.mean(1024 * F.relu(target_s - lam_map)))
        
        # print("check minus min:  ", torch.min((lam_map - target_s)))
        # print("check minus mean: ", torch.mean((lam_map - target_s)))
        print("count number that lamda - true color < 0 : ", minjae_test_num)
        print("iteration : ", i)

        if torch.min(lam_map - target_s) < 0:
            minjae_test_num += 1

        #############################################################################################################
        if i < 100: # pretraining
            img_loss = img2mse(rgb, target_s) 
        elif 100<=i and i<1000:
            img_loss = 1e+4 * img2mse(rgb, target_s) + 1e-8 * loss_uncert3(rgb, lam_map, target_s, U_A, S_A, alpha, args.w)
        elif 1000<=i and i<2000:
            img_loss = 1e+4 * img2mse(rgb, target_s) + 1e-6 * loss_uncert3(rgb, lam_map, target_s, U_A, S_A, alpha, args.w)
        elif 2000<=i and i<3000:
            img_loss = 1e+4 * img2mse(rgb, target_s) + 1e-5 * loss_uncert3(rgb, lam_map, target_s, U_A, S_A, alpha, args.w)
        elif 3000<=i and i<4000:
            img_loss = 1e+4 * img2mse(rgb, target_s) + 1e-4 * loss_uncert3(rgb, lam_map, target_s, U_A, S_A, alpha, args.w)
        else:
            img_loss = 1e+4 * img2mse(rgb, target_s) + 1e-2 * loss_uncert3(rgb, lam_map, target_s, U_A, S_A, alpha, args.w)
        #############################################################################################################
        trans = extras['raw'][...,-1]

        txt_img_loss = img2mse(rgb, target_s) 
        txt_psnr = mse2psnr(txt_img_loss)


        loss = img_loss
        psnr = txt_psnr


        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}    UNCERT: {torch.mean(uncert)}")
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
