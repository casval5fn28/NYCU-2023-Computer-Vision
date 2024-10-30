import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys
import scipy
from sklearn.preprocessing import normalize

image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    #plt.savefig("Mask")

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')
    #plt.savefig("Normal map")

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    #plt.savefig("Depth map")

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z, filepath):
    Z_map = np.reshape(Z, (image_row, image_col)).copy()
    data = np.zeros((image_row * image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file & denoise input
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image_row, image_col = image.shape
    #denoised_img = cv2.GaussianBlur(image, (5, 5), 0)# 10
    denoised_img = cv2.medianBlur(image, 5)# 10 
    return denoised_img

""""""""""""""""""""""""""""""""""""""""""
def read_info(filepath):
    light = []
    img_list = []
    img_num = 0
    
    # light source
    with open(filepath + '/LightSource.txt', 'r') as f: 
        for line in f.readlines():
            line = line.strip()
            point = list(map(int, line[line.find('(') + 1: line.find(')')].split(',')))
            light.append(np.array(point).astype(np.float32))
            img_num += 1
    light = normalize(light, axis = 1)
    
    # image data
    for i in range(1, img_num + 1):
        img = read_bmp(filepath + '/pic' + str(i) + '.bmp')
        img_list.append(img.ravel()) 
    img_list = np.asarray(img_list)

    return light, img_list

def get_norm(L, I):
    # I = L*Kd*N -> get Kd*N
    KdN = np.linalg.solve(L.T @ L, L.T @ I)
    KdN = normalize(KdN, axis = 0).T
    return KdN

def surface_optimization(z, mask, pixel_num):
    nonzero_h, nonzero_w = np.where(mask!=0)
    
    # filter extreme points in z
    z_normed = (z - np.mean(z)) / np.std(z)
    extreme_idx = np.abs(z_normed) > 4
    z_max = np.max(z[~extreme_idx])
    z_min = np.min(z[~extreme_idx])
    
    new_depth = mask.astype(np.float32)
    
    for i in range(pixel_num):
        if z[i] > z_max:
            new_depth[nonzero_h[i], nonzero_w[i]] = z_max
        elif z[i] < z_min:
            new_depth[nonzero_h[i], nonzero_w[i]] = z_min
        else:
            new_depth[nonzero_h[i], nonzero_w[i]] = z[i]    
    
    return new_depth

def compute_depth(mask, N):
    N = np.reshape(N, (image_row, image_col, 3))
    # object's # of pixel 
    pixel_num = np.size(np.where(mask != 0)[0])
    
    # Mz = V
    M = scipy.sparse.lil_matrix((2*pixel_num, pixel_num))
    v = np.zeros((2*pixel_num, 1))
    
    nonzero_h, nonzero_w = np.where(mask!=0)
    # compute index numbers for M
    idx_arr = np.zeros((image_row, image_col)).astype(np.int16)
    for i in range(pixel_num):
        idx_arr[nonzero_h[i], nonzero_w[i]] = i
    
    for i in range(pixel_num):
        h = nonzero_h[i]
        w = nonzero_w[i]
        # normal vecs
        n_x = N[h, w, 0]
        n_y = N[h, w, 1]
        n_z = N[h, w, 2]
        
        # - nx / nz -> z(x+1, y) - z(x, y)
        j = i*2
        if mask[h, w+1]: # right
            k = idx_arr[h, w+1]
            M[j, i] = -1
            M[j, k] = 1
            v[j] = -n_x/n_z
        elif mask[h, w-1]: # left
            k = idx_arr[h, w-1]
            M[j, k] = -1
            M[j, i] = 1
            v[j] = -n_x/n_z
        
        # -ny / nz -> z(x, y+1) - z(x, y)
        j = i*2+1
        if mask[h+1, w]:   
            k = idx_arr[h+1, w] # up
            M[j, i] = 1
            M[j, k] = -1
            v[j] = -n_y/n_z
        elif mask[h-1, w]:
            k = idx_arr[h, w-1] # down
            M[j, k] = 1
            M[j, i] = -1
            v[j] = -n_y/n_z

    # M.T * M * z = M.T * v
    z = scipy.sparse.linalg.spsolve(M.T @ M, M.T @ v)
    
    return surface_optimization(z, mask, pixel_num)
"""
def remove_noise(image):
    obj = cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    transformed_e = cv2.morphologyEx(image, cv2.MORPH_DILATE, obj)
    out_gray = cv2.divide(image, transformed_e, scale=255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 
    tmp_mask = cv2.medianBlur(np.float32(out_binary), 5)

    return image*(np.float32(tmp_mask)/255)
"""

if __name__ == '__main__':
    
    #objects = ['bunny', 'star']
    #objects = ['venus', 'noisy_venus']#
    objects = ['bunny', 'star', 'venus', 'noisy_venus']#
    
    for object in objects:
        filepath = 'test/' + object
        # normal
        light, img_list = read_info(filepath) # L,I
        normal = get_norm(light, img_list)
        
        # mask
        mask = read_bmp(filepath + '/pic1.bmp')
        threshold_value = 20
        max_value = 255
        _, mask = cv2.threshold( mask, threshold_value, max_value, cv2.THRESH_BINARY )

        """
        if object == 'noisy_venus':
            mask = remove_noise(mask)
            mask = cv2.medianBlur(np.float32(mask), 5).flatten()
        """
        # depth
        depth = compute_depth(mask, normal)
        
        normal_visualization(normal)
        mask_visualization(mask)
        depth_visualization(depth)
        save_ply(depth, filepath + '/' + object + '.ply')
        show_ply(filepath + '/' + object + '.ply')

        # showing the windows of all visualization function
        plt.show()