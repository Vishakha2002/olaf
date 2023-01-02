import glob
import logging
import os

import numpy as np
import pretrainedmodels
import torch
from PIL import Image
from pretrainedmodels import utils
from torch import nn
from torchvision import transforms

from .avqa_fusion_net import AVQA_Fusion_Net

log = logging.getLogger(__name__)


def TransformImage(img):

    transform_list = []
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

    transform_list.append(transforms.Resize([224, 224]))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    trans = transforms.Compose(transform_list)
    frame_tensor = trans(img)

    return frame_tensor


def load_frame_info(img_path):

    img = Image.open(img_path).convert("RGB")
    frame_tensor = TransformImage(img)

    return frame_tensor


def extract_feats(model, framepath, load_image_fn):
    C, H, W = 3, 224, 224
    raw_name = framepath.split("/")[-1]

    model.eval()
    output_directory = os.path.join(os.getcwd(), "data/features/video_resnet18")
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    outfile = os.path.join(output_directory, raw_name + ".npy")
    if os.path.exists(outfile):
        log.info(outfile, "already exist!")
        return outfile

    ### image
    select_img = []
    image_list = sorted(glob.glob(os.path.join("data/frames/video", raw_name, "*.jpg")))
    log.info(f"Count of frame images: {len(image_list)}" )

    samples = np.round(np.linspace(0, len(image_list) - 1, len(image_list)))

    image_list = [image_list[int(sample)] for sample in samples]
    image_list = image_list[::1]  # 1 fps
    for img in image_list:
        frame_tensor_info = load_frame_info(img)
        select_img.append(frame_tensor_info.cpu().numpy())
    select_img = np.array(select_img)
    select_img = torch.from_numpy(select_img)

    select_img = select_img.unsqueeze(0)

    with torch.no_grad():
        visual_out = model(select_img.cuda())
    fea = visual_out.cpu().numpy()

    log.info(f"Feature shape {fea.shape}")
    np.save(outfile, fea)

    return outfile


def extract_video_feature(framepath):
    # XXX maybe needed in Super computer.
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
    torch.cuda.empty_cache()

    pretrained_resnet_model = pretrainedmodels.resnet18(pretrained="imagenet")
    load_image_fn = utils.LoadTransformImage(pretrained_resnet_model)

    model = AVQA_Fusion_Net()
    model = nn.DataParallel(model)
    model = model.cuda()

    file_path = extract_feats(model, framepath, load_image_fn)
    return file_path
