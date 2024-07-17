import numpy as np
import torch
import torch.nn as nn
from skimage import measure
import torchvision
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import h5py
from tqdm import tqdm


from utils import rend_util
from utils.general import trans_topil
from utils.sem_util import mask2color

import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots

import random

def plot(indices, plot_data, path, epoch, img_res):

    if plot_data is not None:

        # plot images
        plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, 1, img_res, indices)
      
        # plot normal maps
        plot_normal_maps(plot_data['normal_map'], plot_data['normal_gt'], path, epoch, 1, img_res, indices)

        # plot depth maps
        plot_depth_maps(plot_data['depth_map'], plot_data['depth_gt'], path, epoch, 1, img_res, indices)

        # plot sem maps
        plot_sem_maps(plot_data['seg_map'], plot_data['seg_gt'], path, epoch, 1, img_res, indices)

        # concat output images to single large image
        images = []
        for name in [ "depth", "sem", "rendering", "normal"]:
            images.append(cv2.imread('{0}/{1}_{2}_{3}.png'.format(path, name, epoch, indices[0])))        
        
        images = np.concatenate(images, axis=1)
        cv2.imwrite('{0}/merge_{1}_{2}.png'.format(path, epoch, indices[0]), images)



avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')



def get_surface_trace(path, epoch, sdf, x, level=0, num=0):
    resolution = sdf.shape[0]
    sdf = sdf.reshape(-1, num)
    pool = nn.MaxPool1d(num)
    z = -pool((-sdf.unsqueeze(1)).squeeze(-1)).numpy()
    z = z.reshape(resolution,resolution,resolution)
   
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z,
            level=level,
            spacing=(x[2] - x[1],
                    x[2] - x[1],
                    x[2] - x[1]))

        verts = verts + np.array([x[0], x[0], x[0]])

        meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
        meshexport.export('{0}/surface_{1}_whole.ply'.format(path, epoch), 'ply')


def get_semantic_surface_trace(path, epoch, sdf, x, level=0, num=0):
    for idx in tqdm(range(num)):
        z = sdf[:, :, :, idx]
        if (not (np.min(z) > level or np.max(z) < level)):
            z = z.astype(np.float32)
            verts, faces, normals, values = measure.marching_cubes(
                volume=z,
                level=level,
                spacing=(x[2] - x[1],
                        x[2] - x[1],
                        x[2] - x[1]))

            verts = verts + np.array([x[0], x[0], x[0]])

            meshexport = trimesh.Trimesh(verts, faces, normals)
            meshexport.export('{0}/surface_{1}_{2}.ply'.format(path, epoch, idx), 'ply')

    






def colored_data(x, cmap='jet', d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(x)
    if d_max is None:
        d_max = np.max(x)
    x_relative = (x - d_min) / (d_max - d_min)
    cmap_ = plt.cm.get_cmap(cmap)
    return (255 * cmap_(x_relative)[:,:,:3]).astype(np.uint8) # H, W, C


def plot_sem_maps(sem_maps, ground_true, path, epoch, plot_nrow, img_res, indices):
    ground_true = ground_true.cuda()
    # import pdb; pdb.set_trace()
    sem_maps = torch.cat((sem_maps[..., None], ground_true), dim=0)
    sem_maps_plot = lin2img(sem_maps, img_res)
    # sem_maps_plot = mask2color(sem_maps_plot, is_argmax=False)

    tensor = torchvision.utils.make_grid(sem_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)[:,:,0]
    # import pdb; pdb.set_trace()
    tensor = colored_data(tensor)

    img = Image.fromarray(tensor)
    img.save('{0}/sem_{1}_{2}.png'.format(path, epoch, indices[0]))

    

def plot_depth_maps(depth_maps, ground_true, path, epoch, plot_nrow, img_res, indices):
    ground_true = ground_true.cuda()
    depth_maps = torch.cat((depth_maps[..., None], ground_true), dim=0)
    depth_maps_plot = lin2img(depth_maps, img_res)
    depth_maps_plot = depth_maps_plot.expand(-1, 3, -1, -1)

    tensor = torchvision.utils.make_grid(depth_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    
    save_path = '{0}/depth_{1}_{2}.png'.format(path, epoch, indices[0])
    
    plt.imsave(save_path, tensor[:, :, 0], cmap='viridis')
    
    
def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res, indices):
    ground_true = ground_true.cuda()

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}_{2}.png'.format(path, epoch, indices[0]))    

def plot_normal_maps(normal_maps, ground_true, path, epoch, plot_nrow, img_res, indices):
    ground_true = ground_true.cuda()
    normal_maps = torch.cat((normal_maps, ground_true), dim=0)
    normal_maps_plot = lin2img(normal_maps, img_res)

    tensor = torchvision.utils.make_grid(normal_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/normal_{1}_{2}.png'.format(path, epoch, indices[0]))



def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
