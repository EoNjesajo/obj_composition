import numpy as np
import torch
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

def plot(implicit_network, indices, plot_data, path, epoch, img_res, rendering_network, plot_nimgs, resolution, grid_boundary,  level=0):

    if plot_data is not None:
        cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])

        # plot images
        # plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot depth maps
        plot_depth_maps(plot_data['depth_map'], plot_data['depth_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot sem maps
        plot_sem_maps(plot_data['seg_map'], plot_data['seg_gt'], path, epoch, plot_nimgs, img_res, indices)

        # concat output images to single large image
        images = []
        # for name in ["rendering", "depth", "sem"]:
        for name in ["rendering", "depth", "sem"]:
            images.append(cv2.imread('{0}/{1}_{2}_{3}.png'.format(path, name, epoch, indices[0])))        

        # images = np.concatenate(images, axis=1)
        # cv2.imwrite('{0}/merge_{1}_{2}.png'.format(path, epoch, indices[0]), images)

    # plot each mesh
    sem_num = implicit_network.d_out
    # for indx in range(sem_num):
    # _ = get_semantic_surface_trace(path=path,
    #                                 epoch=epoch,
    #                                 #    sdf=lambda x: -f(-implicit_network(x)[:, :6]),
    #                                 sdf = lambda x: implicit_network(x)[:,:sem_num],
    #                                 resolution=resolution,
    #                                 grid_boundary=grid_boundary,
    #                                 level=level,
    #                                 num = sem_num
    #                                 )


avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')


def get_semantic_surface_trace(path, epoch, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0, num=0):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']
    z_all = []
    for pnts in tqdm(torch.split(points, 100000, dim=0)):
        z_all.append(sdf(pnts.cuda()).detach().cpu().numpy())
        # if i < 50: 
        #     print(sdf(pnts.cuda()).detach().cpu().numpy().shape)
        #     z_all.append(sdf(pnts.cuda()).detach().cpu().numpy())
        # else :
        #     print(np.ones((100000,46),dtype=np.float32).shape)
        #     z_all.append(np.ones((len(pnts),46),dtype=np.float32)*0.2)    
    print(z_all[0].shape)
    z_all = np.concatenate(z_all, axis=0)

    for idx in tqdm(range(num)):
        z = z_all[:, idx]
        if (not (np.min(z) > level or np.max(z) < level)):

            z = z.astype(np.float32)
            print(grid['xyz'][0][2] - grid['xyz'][0][1])
            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                        grid['xyz'][0][2] - grid['xyz'][0][1],
                        grid['xyz'][0][2] - grid['xyz'][0][1]))

            verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            I, J, K = faces.transpose()

            traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                i=I, j=J, k=K, name='implicit_surface',
                                color='#ffffff', opacity=1.0, flatshading=False,
                                lighting=dict(diffuse=1, ambient=0, specular=0),
                                lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

            meshexport = trimesh.Trimesh(verts, faces, normals)
            meshexport.export('{0}/surface_{1}_{2}.ply'.format(path, epoch, idx), 'ply')

    if return_mesh:
        return meshexport
    return traces
    
        
def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace





def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)
    return {"grid_points": grid_points,
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}



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

    #import pdb; pdb.set_trace()
    #trans_topil(normal_maps_plot[0, :, :, 260:260+680]).save('{0}/2normal_{1}.png'.format(path, epoch))


def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res, indices, exposure=False):
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
    if exposure:
        img.save('{0}/exposure_{1}_{2}.png'.format(path, epoch, indices[0]))
    else:
        img.save('{0}/rendering_{1}_{2}.png'.format(path, epoch, indices[0]))


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
    

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
