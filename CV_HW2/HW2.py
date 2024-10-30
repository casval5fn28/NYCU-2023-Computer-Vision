import cv2
import numpy as np
import random
import math
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# function for returning cylindrically-projected imgs 
def cylindrical_proj(img):
    h, w = img.shape[:2]
    K = np.array([[1600,0,w/2],[0,1600,h/2],[0,0,1]]) # intrinsic matrix
    h_,w_ = img.shape[:2]
    y_i, x_i = np.indices((h_,w_)) # pixel coordinates

    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords

    # to get cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    #img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp img according to cylindrical coords
    return cv2.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)# img_rgba

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #brightness = cv2.mean(img_gray)[0]
    return cylindrical_proj(img), cylindrical_proj(img_gray)#, brightness

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# crop result into a form with less black border
def crop_img(img):
    h, w = img.shape[:2]
    crop_1=0.96
    crop_2=0.75
    crop_w = int(w * crop_1)
    crop_h = int(h * crop_2)

    start_x = w - crop_w
    start_y = h - crop_h
    cropped_img = img[start_y:h, start_x:w]

    return cropped_img

#######################################
# Step 1: SIFT
def SIFT(img):
    SIFT_Detector = cv2.SIFT_create()
    kp, descript = SIFT_Detector.detectAndCompute(img, None)
    return kp, descript

def SIFT_plot(img, kp):
    tmp = img.copy()
    tmp = cv2.drawKeypoints(tmp, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return tmp

# Step 2: KNN feature matching
def matcher(kp1, des1, kp2, des2, threshold):
    """
    # using BFMatcher (For debugging):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Lowe's ratio test
    good = []
    for m,n in matches:
        # m, n are DMatch
        if m.distance < threshold*n.distance:
            good.append([m])
    """
    # Brutal Force KNN
    matches = []
    for i in tqdm(range(len(kp1))):
        dmatch = {"distance":1e7, "queryIdx":0, "trainIdx":0} # smallest
        dmatch_2nd = {"distance":1e7, "queryIdx":0, "trainIdx":0} # 2nd smallest
        v1 = des1[i,:]
        for j in range(len(kp2)):
            v2 = des2[j,:]
            dist = 0
            dist = np.linalg.norm(v1-v2)
            if dist < dmatch["distance"]:
                dmatch_2nd["distance"] = dmatch["distance"]
                dmatch_2nd["queryIdx"] = dmatch["queryIdx"]
                dmatch_2nd["trainIdx"] = dmatch["queryIdx"]
                dmatch["distance"] = dist
                dmatch["queryIdx"] = i
                dmatch["trainIdx"] = j
            elif dist < dmatch_2nd["distance"]:
                dmatch_2nd["distance"] = dist
                dmatch_2nd["queryIdx"] = i
                dmatch_2nd["trainIdx"] = j
        matches.append((dmatch, dmatch_2nd))

    good_match = []
    # Lowe's ratio test
    for m,n in matches:# m, n are DMatch
        if m["distance"] < threshold*n["distance"]:
            good_match.append([m])
    
    return good_match

def get_matches(good_match, kp1, kp2):
    # matches [[x1, y1, x1', y1'], ....., [xn, yn, xn', yn']]
    matches = []
    for pair in good_match:
        matches.append(list(kp1[pair[0]["queryIdx"]].pt + kp2[pair[0]["trainIdx"]].pt))  # for self define pair
        # matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))      # for dmatch class(BFMatcher())
    matches = np.array(matches)
    return matches

#######################################
# Step 3: RANSAC
def homography_matrix(pairs):
    # pairs: [[(x1, y1), (x1', y1')],[(x2, y2), (x2', y2')] [(x3, y3), (x3', y3')], [(x4, y4), (x4', y4')]]
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)   # use the last vector(np.linalg.svd has vector sorted in descending order)
    H = H/H[2, 2] # normalize homography matrix 
    return H

def random_point(matches, k=4):
    # get k pairs from the matches (used in ransac)
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(matches, H):
    # matches: [[x1, y1, x1', y1'], ..., [xn, yn, xn', yn']]
    num_points = len(matches)
    total_pair_1 = np.concatenate((matches[:, 0:2], np.ones((num_points, 1))), axis=1) # change (x1, y1) to (x1, y1, 1)
    total_pair_2 = matches[:, 2:4]
    estimate_pair_2 = np.zeros((num_points, 2))
    for i in range(num_points):
        tmp = np.dot(H, total_pair_1[i])
        estimate_pair_2[i] = (tmp/tmp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    
    errors = np.linalg.norm(total_pair_2 - estimate_pair_2 , axis=1) ** 2
    return errors

#######################################
def ransac(matches, threshold, iters):
    best_inlier_num = 0
    best_H = np.zeros((3, 3))
    for i in tqdm(range(iters)):
        chosen_points = random_point(matches)
        H = homography_matrix(chosen_points)
        if np.linalg.matrix_rank(H) < 3:# avoid dividing by zero 
            continue
            
        errors = get_error(matches, H)
        inliers = matches[np.where(errors < threshold)[0]]

        inlier_num = len(inliers)
        if inlier_num > best_inlier_num:
            best_inliers = inliers.copy()
            best_inlier_num = inlier_num
            best_H = H.copy()
            
    print("Inliers/Matches: {}/{}".format(best_inlier_num, len(matches)))
    return best_inliers, best_H

#######################################
# for blendering the imgs
class Blender:
    def linear_blend(self, imgs):
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        constant_w = 0.01 # constant width
        
        # find left & right img mask region(pixels that aren't 0s)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
        
        # find overlap mask(overlap region of two imgs)
        overlap_mask = np.zeros((hr, wr), dtype="int")

        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        """
        # plot overlap mask
        plt.figure(21)
        plt.title("overlap_mask")
        plt.imshow(overlap_mask.astype(int), cmap="gray")
        plt.show()
        """
        # alpha mask for linear blending overlap region
        alpha_mask = np.zeros((hr, wr)) # alpha val is based on left img
        for i in range(hr): 
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            if (minIdx == maxIdx): # the row's pixels are all zero / only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            # original, for base
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
            """
            # with constant width, for challenge
            # for finding middle line of overlapping regions, only do linear blending to regions very close to the middle line.
            middleIdx = int((maxIdx + minIdx) / 2)
            # left 
            for j in range(minIdx, middleIdx + 1):
                if (j >= middleIdx - constant_w):
                    alpha_mask[i, j] = 1 - (decrease_step * (i - minIdx))
                else:
                    alpha_mask[i, j] = 1
            # right
            for j in range(middleIdx + 1, maxIdx + 1):
                if (j <= middleIdx + constant_w):
                    alpha_mask[i, j] = 1 - (decrease_step * (i - minIdx))
                else:
                    alpha_mask[i, j] = 0
            """
        linear_blend_img = np.copy(img_right)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                    linear_blend_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
                elif (np.count_nonzero(img_left_mask[i, j])>0):
                    linear_blend_img[i, j] = img_left[i,j]
                elif (np.count_nonzero(img_right_mask[i, j])>0):
                    linear_blend_img[i, j] = img_right[i,j]
        
        return linear_blend_img

def crop_black_border(img):
    h, w = img.shape[:2]
    reduced_h, reduced_w = h, w
    # from right hand side
    for col in range(w - 1, -1, -1):
        all_black = True
        for i in range(h):
            if (np.count_nonzero(img[i, col]) > 0):
                all_black = False
                break
        if (all_black == True):
            reduced_w = reduced_w - 1
            
    # from bottom
    for row in range(h - 1, -1, -1):
        all_black = True
        for i in range(reduced_w):
            if (np.count_nonzero(img[row, i]) > 0):
                all_black = False
                break
        if (all_black == True):
            reduced_h = reduced_h - 1
    
    return img[:reduced_h, :reduced_w]

#######################################
# Step 4: Stitich image
def stitch_img(left, right, H):
    
    # both convert to double & normalize, for avoiding noise.
    left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left img
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)
    y_min = min(y_min, 0)#
    x_min = min(x_min, 0)#
    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)

    # Get h&w of right img for "left" img 
    h2, w2 ,c2= right.shape
    y_min = round(y_min)
    x_min = round(x_min)
    height_new = int(round(abs(y_min)) + h2)
    width_new = int(round(abs(x_min)) + w2)
    size = (width_new, height_new)

    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
    
    """""""""""""""""""""""長寬選擇注意 !!!"""""""""""""""""""""""
    # right image
    height_r, width_r, channel_r = right.shape
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    
    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)

    # stitching process, results stored in warped_img
    blender = Blender()
    warped_img= blender.linear_blend([warped_l, warped_r])
    # cv2.imshow("blender", warped_img)
    warped_img = crop_black_border(warped_img)
    warped_l*=255.0
    warped_l = warped_l.astype(np.uint8)
    warped_r*=255.0
    warped_r = warped_r.astype(np.uint8)
    return warped_img

def stitch(img_left_g, img_right_g, img_left, img_right):
    # 1.SIFT
    print("Step 1: SIFT ...")
    kp_left, des_left = SIFT(img_left_g)
    kp_right, des_right = SIFT(img_right_g)

    """
    # plot SIFT image
    kp_left_img = SIFT_plot(img_left, kp_left)
    kp_right_img = SIFT_plot(img_right, kp_right)
    total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
    creat_im_window("total_kp", total_kp)
    """

    #Step 2: KNN feature matching
    print("Step 2: KNN Feature Matching ...")
    good_match = matcher(kp_left, des_left, kp_right, des_right, 0.7)# 0.75, good
    matches = get_matches(good_match, kp_left, kp_right)

    # 3. Homography
    print("Step 3: RANSC ...")
    inliers, H = ransac(matches, 0.4, 8000)

    # Step 4 Stitch Image
    print("Step 4: Stiching Image ...")
    img_stitched = stitch_img(img_left, img_right, H)

    # Change img back to [0, 255] for storing
    img_stitched*=255.0
    img_stitched = img_stitched.astype(np.uint8)
    return img_stitched
#######################################
# Step 5 Adjusting brightness
def adjust_brightness(img, target_bright):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    curr_bright = cv2.mean(gray)[0]
    ratio = target_bright / curr_bright
    adjusted_image = cv2.convertScaleAbs(img, alpha=ratio, beta=0)
    
    return adjusted_image

#######################################
# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()
    
    imgs = []
    img_grays = []
    base = "./Base" #Base, Challenge
    img_number = 3 #3,6
    bright_sum = 0

    for i in range(1,img_number+1):
        path = base +f"/Base{str(i)}.jpg"#Base,Challenge{str(i)}
        img, img_gray= read_img(path)
        imgs.append(img)
        img_grays.append(img_gray)
        brightness = cv2.mean(img_gray)[0]
        bright_sum += brightness
    #print(len(img_grays))
    #print(len(imgs))
    bright_avg = bright_sum / len(imgs)
    
    # Start from right hand side.
    img_left_g = img_grays[1]#[1,4]
    img_right_g = img_grays[2]#[2,5]
    img_left = imgs[1]#[1,4]
    img_right = imgs[2]#[2,5]
    # for challenge
    #img_left = adjust_brightness(imgs[4], bright_avg)#[1,4]
    #img_right = adjust_brightness(imgs[5], bright_avg)#[2,5]
    
    #img_left, img_left_g = read_img("./base_left.jpg")
    #img_right, img_right_g = read_img("./base_right.jpg")
    img = stitch(img_left_g, img_right_g, img_left, img_right)
    
    for i in range(1,-1,-1):#(1,-1,-1),(4,-1,-1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # for base
        img = stitch(img_grays[i], img_gray, imgs[i], img)
        # for challenge
        #img = stitch(img_grays[i], img_gray, adjust_brightness(imgs[i], bright_avg), adjust_brightness(img, bright_avg))
    
    #img = crop_img(img)
    cv2.imwrite("base_result.jpg",img)
    #cv2.imwrite("challenge_result.jpg",img)