import cv2
import numpy as np
import argparse
import time
import sys
def segment_ycrcb(orig, params, tola, tolb):
    ycrcb_im = cv2.cvtColor(orig, cv2.COLOR_BGR2YCrCb)
    Cb_key, Cr_key = params
    blue = ycrcb_im[:, :, 2]
    red = ycrcb_im[:, :, 1]

    diffbsq = (blue - Cb_key)**2
    diffrsq = (red - Cr_key)**2
    dist = np.sqrt(diffbsq + diffrsq).astype(np.float32)

    mask = ((dist - tola) / (tolb - tola)).astype(np.float32)
    mask[dist < tola] = 0.0
    mask[dist > tolb] = 1.0
    return mask


def get_region(img):
    # r = cv2.selectROI("Select window of background", img=img)
    # print r
    # exit(1)
    # r = cv2.selectROI("Select window of background", img=img,
    #                     fromCenter=False, showCrossair=False)
    r=(80,80,100,100)
    cv2.destroyAllWindows()
    return r


def get_params_ycrcb(img, region):
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cv2.destroyAllWindows()
    r = [int(x) for x in region]
    region = ycrcb_img[int(region[1]):int(
        region[1] + region[3]), int(region[0]):int(region[0] + region[2])]
    y_mean, Cr_mean, Cb_mean = np.mean(region, axis=(0, 1))
    y_std, Cr_std, Cb_std = np.std(region, axis=(0, 1))
    return [Cb_mean, Cr_mean]


def get_params_hls(img, region):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    r = [int(x) for x in region]
    region = hls_img[int(region[1]):int(region[1] + region[3]),
                     int(region[0]):int(region[0] + region[2])]
    h_mean, l_mean, s_mean = np.mean(region, axis=(0, 1))
    h_std, l_std, s_std = np.std(region, axis=(0, 1))
    return [h_mean, h_std, l_mean, l_std, s_mean, s_std]


def brighten(img, alpha, beta, gamma):
    cor_img = img.astype(np.float32)
    bright_img = alpha * cor_img + beta
    gam_cor = np.power(bright_img / 255, gamma) * 255
    bright_img = gam_cor.clip(0, 255).astype(np.uint8)
    return bright_img


def mod_mask(mask, low, high):
    mask = mask.copy()
    mask[mask > high] = 1.0
    mask[mask < low] = 0.0
    return mask


# def get_mask(img, bg, param_ycrcb, param_hls, tola=16, tolb=50, low_thresh=0.05, high_thresh=0.25, alpha=1.0, beta=0.0, gamma=1.0, sz=5, space=200, erode_sz=3):
def get_mask(img,param_ycrcb, tola=16, tolb=50, low_thresh=0.05, high_thresh=0.25, sz=5, space=200,erode_sz=2):
    ##ENSURE THAT param_ycrcb and param_hls correspond to brigthened img

    #     brimg = brighten(img,alpha,beta,gamma)
    brimg = img

    if not (sz<=0 or space<=1):
        brimg = cv2.bilateralFilter(brimg, sz, space, space)

    mask = segment_ycrcb(brimg, param_ycrcb, tola, tolb)
    mask = mod_mask(mask, low_thresh, high_thresh)

    if not(erode_sz <= 1):
        kernel = np.ones((erode_sz,erode_sz),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
    return mask
def get_bgra(img,mask):
    assert (len(mask.shape) == 2)  # (x,y)
    assert (len(img.shape) == 3)  # (x,y,3)
    alpha = (mask * 255).astype(np.uint8)
    alpha = np.expand_dims(alpha,-1) ##(x,y,1)
    return np.concatenate((img,alpha),axis=2) ##(x,y,4)
def write_alpha_img(img, mask, path):
    r_channel, g_channel, b_channel = cv2.split(img)
    alpha = (mask * 255).astype(np.uint8)
    img_RGBA = cv2.merge((r_channel, g_channel, b_channel, alpha))
    cv2.imwrite(path, img_RGBA)
    return img_RGBA
def get_key_param(img):
    key_region = get_region(img)
    ycrcb = get_params_ycrcb(img, key_region)
    hls = get_params_hls(img, key_region)
    return (ycrcb, hls)
def segmenter(img):
    # green_img_path = sys.argv[1]
    # img = cv2.imread(green_img_path)
    print img.shape
    # exit(1)
    key_param = get_key_param(img)
    cv2.namedWindow('controls')
    def nothing(gyxfdd):
        pass

    cv2.createTrackbar('Keying tol low', 'controls', 23 , 100, nothing)
    cv2.createTrackbar('Keying tol high', 'controls', 50, 200, nothing)
    cv2.createTrackbar('Mask low Thresh (x100)', 'controls', 5, 100, nothing)
    cv2.createTrackbar('Mask high Thresh (x100)', 'controls', 25, 100, nothing)
    cv2.createTrackbar('Erode size', 'controls', 3, 10, nothing)
    cv2.createTrackbar('BiLat size', 'controls', 5, 100, nothing)
    cv2.createTrackbar('BiLat space', 'controls', 200, 500, nothing)
    cv2.createTrackbar('Sat mul low', 'controls', 5, 100, nothing)
    cv2.createTrackbar('Sat mul high', 'controls', 7, 100, nothing)
    cv2.createTrackbar('Light mask strength', 'controls', 20, 100, nothing)
    cv2.createTrackbar('Light mask size', 'controls', 3, 20, nothing)
    # cv2.imshow("nn",img)
    # cv2.waitKey(0)

    # while True:
    tola = cv2.getTrackbarPos('Keying tol low', 'controls')
    tolb = cv2.getTrackbarPos('Keying tol high', 'controls')
    low_thresh = cv2.getTrackbarPos('Mask low Thresh (x100)', 'controls')/100
    high_thresh = cv2.getTrackbarPos('Mask high Thresh (x100)', 'controls')/100
    erode_sz = cv2.getTrackbarPos('Erode size', 'controls')
    sz = cv2.getTrackbarPos('BiLat size', 'controls')
    space  = cv2.getTrackbarPos('BiLat space', 'controls')
    sat_mul_lo  = cv2.getTrackbarPos('Sat mul low', 'controls')
    sat_mul_hi  = cv2.getTrackbarPos('Sat mul high', 'controls')
    scale_blur  = cv2.getTrackbarPos('Light mask strength', 'controls')
    blur_size  = cv2.getTrackbarPos('Light mask size', 'controls')

    key_mask = get_mask( img, key_param[0], tola, tolb, low_thresh, high_thresh, sz, space, erode_sz)
        # cv2.imshow("image",key_mask*255)
        # cv2.waitKey(300)
    cv2.imwrite('lolu.jpg',key_mask*255)
    return key_mask*255

if __name__=="__main__":
    segmenter(cv2.imread(sys.argv[1]))

