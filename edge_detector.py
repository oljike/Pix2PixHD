import sys
import cv2
import random
import warnings
import numpy as np
from PIL import Image
from skimage import feature
from scipy.optimize import curve_fit
import torchvision.transforms as transforms

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def make_power_2(n, base=32.0):
    return int(round(n / base) * base)

def get_img_params(size):
    w, h = size
    loadSize = 512

    # scale image width to be loadSize
    new_w = loadSize
    new_h = loadSize * h // w

    new_w = int(round(new_w / 4)) * 4
    new_h = int(round(new_h / 4)) * 4
    new_w, new_h = make_power_2(new_w), make_power_2(new_h)

    flip = (random.random() > 0.5) and (True)
    return {'new_size': (new_w, new_h), 'flip': flip}

def get_transform(params, method=Image.BICUBIC, normalize=True, toTensor=True):

    transform_list = []
    no_flip = False

    ### resize input image
    transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))

    ### random flip
    if not no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def interpPoints(x, y):
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
        if curve_y is None:
            return None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(x) < 3:
                popt, _ = curve_fit(linear, x, y)
            else:
                popt, _ = curve_fit(func, x, y)
                if abs(popt[0]) > 1:
                    return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1] - x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)

def func(x, a, b, c):
    return a * x ** 2 + b * x + c

def drawEdge(im, x, y, bw=1, color=(255, 255, 255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h - 1, y + i))
                xx = np.maximum(0, np.minimum(w - 1, x + j))
                setColor(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw * 2, bw * 2):
                for j in range(-bw * 2, bw * 2):
                    if (i ** 2) + (j ** 2) < (4 * bw ** 2):
                        yy = np.maximum(0, np.minimum(h - 1, np.array([y[0], y[-1]]) + i))
                        xx = np.maximum(0, np.minimum(w - 1, np.array([x[0], x[-1]]) + j))
                        setColor(im, yy, xx, color)

def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
        else:
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]

def linear(x, a, b):
    return a * x + b

def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size
    return img.resize((w, h), method)

def __crop(img, size, pos):
    ow, oh = img.size
    tw, th = size
    x1, y1 = pos
    if (ow > tw or oh > th):
        return img.crop((x1, y1, min(ow, x1 + tw), min(oh, y1 + th)))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


class FaceEdge():
    def __init__(self, ):
        self.scale_ratio = np.array(
            [[0.9, 1], [1, 1], [0.9, 1], [1, 1.1], [0.9, 0.9], [0.9, 0.9]])  # np.random.uniform(0.9, 1.1, size=[6, 2])
        self.scale_ratio_sym = np.array(
            [[1, 1], [0.9, 1], [1, 1], [0.9, 1], [1, 1], [1, 1]])  # np.random.uniform(0.9, 1.1, size=[6, 2])
        self.scale_shift = np.zeros((6, 2))  # np.random.uniform(-5, 5, size=[6, 2])

    def get_face_image(self, points, transform_A, size, img, add_face_keypoints):

        # read face keypoints from path and crop face region
        keypoints, part_list, part_labels = self.read_keypoints(points, size)

        # draw edges and possibly add distance transform maps
        im_edges = self.draw_face_edges(keypoints, part_list, size, add_face_keypoints)

        # canny edge for background
        if True:
            edges = feature.canny(np.array(img.convert('L')))
            im_edges += (edges * 255).astype(np.uint8)

        # final output tensor
        edge_tensor = transform_A(Image.fromarray(self.crop(im_edges)))

        return edge_tensor

    def read_keypoints(self, points, size):
        # mapping from keypoints to face part
        part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]],  # face
                     [range(17, 22)],  # right eyebrow
                     [range(22, 27)],  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],  # nose
                     [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
                     [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
                     [range(48, 55), [54, 55, 56, 57, 58, 59, 48]],  # mouth
                     [range(60, 65), [64, 65, 66, 67, 60]]  # tongue
                     ]
        label_list = [1, 2, 2, 3, 4, 4, 5, 6]  # labeling for different facial parts
        keypoints = points

        # add upper half face by symmetry
        pts = keypoints[:17, :].astype(np.int32)
        baseline_y = (pts[0, 1] + pts[-1, 1]) / 2
        upper_pts = pts[1:-1, :].copy()
        upper_pts[:, 1] = baseline_y + (baseline_y - upper_pts[:, 1]) * 2 // 3
        keypoints = np.vstack((keypoints, upper_pts[::-1, :]))

        # label map for facial part
        w, h = size
        part_labels = np.zeros((h, w), np.uint8)
        for p, edge_list in enumerate(part_list):
            indices = [item for sublist in edge_list for item in sublist]
            pts = keypoints[indices, :].astype(np.int32)
            cv2.fillPoly(part_labels, pts=[pts], color=label_list[p])

        # move the keypoints a bit
        self.scale_points(keypoints, part_list[1] + part_list[2], 1, sym=True)
        self.scale_points(keypoints, part_list[4] + part_list[5], 3, sym=True)
        for i, part in enumerate(part_list):
            self.scale_points(keypoints, part, label_list[i] - 1)

        return keypoints, part_list, part_labels

    def draw_face_edges(self, keypoints, part_list, size, add_face_keypoints=True):
        w, h = size
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges

        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8)  # edge map for all edges

        for edge_list in part_list:
            for edge in edge_list:
                for i in range(0, max(1, len(edge) - 1), edge_len - 1):  # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i + edge_len]
                    x = keypoints[sub_edge, 0]  # *0.0
                    y = keypoints[sub_edge, 1]  # *0.0

                    curve_x, curve_y = interpPoints(x, y)  # interp keypoints to get the curve shape
                    if add_face_keypoints:
                        drawEdge(im_edges, curve_x, curve_y)

        return im_edges

    def get_crop_coords(self, keypoints, size):
        min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()
        min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
        offset = (max_x - min_x) // 2
        min_y = max(0, min_y - offset * 2)
        min_x = max(0, min_x - offset)
        max_x = min(size[0], max_x + offset)
        max_y = min(size[1], max_y + offset)
        self.min_y, self.max_y, self.min_x, self.max_x = int(min_y), int(max_y), int(min_x), int(max_x)

    def crop(self, img):
        if isinstance(img, np.ndarray):
            return img[self.min_y:self.max_y, self.min_x:self.max_x]
        else:
            return img.crop((self.min_x, self.min_y, self.max_x, self.max_y))

    def scale_points(self, keypoints, part, index, sym=False):
        if sym:
            pts_idx = sum([list(idx) for idx in part], [])
            pts = keypoints[pts_idx]
            ratio_x = self.scale_ratio_sym[index, 0]
            ratio_y = self.scale_ratio_sym[index, 1]
            mean = np.mean(pts, axis=0)
            mean_x, mean_y = mean[0], mean[1]
            for idx in part:
                pts_i = keypoints[idx]
                mean_i = np.mean(pts_i, axis=0)
                mean_ix, mean_iy = mean_i[0], mean_i[1]
                new_mean_ix = (mean_ix - mean_x) * ratio_x + mean_x
                new_mean_iy = (mean_iy - mean_y) * ratio_y + mean_y
                pts_i[:, 0] = (pts_i[:, 0] - mean_ix) + new_mean_ix
                pts_i[:, 1] = (pts_i[:, 1] - mean_iy) + new_mean_iy
                keypoints[idx] = pts_i

        else:
            pts_idx = sum([list(idx) for idx in part], [])
            pts = keypoints[pts_idx]
            ratio_x = self.scale_ratio[index, 0]
            ratio_y = self.scale_ratio[index, 1]
            mean = np.mean(pts, axis=0)
            mean_x, mean_y = mean[0], mean[1]
            pts[:, 0] = (pts[:, 0] - mean_x) * ratio_x + mean_x + self.scale_shift[index, 0]
            pts[:, 1] = (pts[:, 1] - mean_y) * ratio_y + mean_y + self.scale_shift[index, 1]
            keypoints[pts_idx] = pts