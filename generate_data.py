import io
import os
import glob
import dlib
import argparse
import numpy as np
import torchvision
from skimage import io
from PIL import Image
from edge_detector import FaceEdge, get_img_params, get_transform

def draw_edge_dir(detector, predictor, source_dir, target_dir, add_face_keypoints):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    img_paths = glob.glob(os.path.join(source_dir, '*.jpg'))

    face_edger = FaceEdge()
    en=0
    for img_path in img_paths:
        if en%1000==0 and en!=0:
            print("The number of processed images is", en)

        img = io.imread(img_path)
        dets = detector(img, 1)
        if len(dets) > 0:

            shape = predictor(img, dets[0])
            points = np.empty([68, 2], dtype=int)
            for b in range(68):
                points[b, 0] = shape.part(b).x
                points[b, 1] = shape.part(b).y

            img_ = Image.open(img_path)
            img_size = img_.size

            face_edger.get_crop_coords(points, img_size)
            B_crop = face_edger.crop(img_)

            params = get_img_params(B_crop.size)
            transform_scale = get_transform(params, method=Image.BILINEAR, normalize=False)

            edge_image = face_edger.get_face_image(points, transform_scale, img_size, img_, add_face_keypoints)

            save_path = os.path.join(target_dir, os.path.basename(img_path))
            torchvision.utils.save_image(edge_image, filename=(save_path))

            en+=1

    print("The total number of edge images is ", en)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source_dir', default='celeb_pics/imgs',
                        help='Directory with source images')
    parser.add_argument('--target_dir', default='celeb_pics/edges',
                        help='Directory for edge images')
    parser.add_argument('--predictor_path', default='predictor/shape_predictor_68_face_landmarks.dat',
                        help='Path to 68 face keypoints predictor')
    parser.add_argument('--add_face_keypoints', default=True,
                        help='Boolean to add face countour based on keypoints')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.predictor_path)

    draw_edge_dir(detector, predictor, source_dir=args.source_dir,
                  target_dir=args.target_dir, add_face_keypoints=args.add_face_keypoints)
