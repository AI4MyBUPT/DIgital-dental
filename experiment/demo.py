import argparse

import torch
import numpy as np

from aagt.utils.data import registration_collate_fn_stack_mode, precompute_neibors
from aagt.utils.torch import to_cuda, release_cuda
# from aagt.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from aagt.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default='../data/src.npy', help="src point cloud numpy file")
    parser.add_argument("--ref_file", default='../data/ref.npy', help="ref point cloud numpy file")
    parser.add_argument("--src_feats", default='../data/src_curnor.npy', help="src feats numpy file")
    parser.add_argument("--ref_feats", default='../data/ref_curnor.npy', help="ref feats numpy file")
    parser.add_argument("--gt_file", default='../data/gt.npy', help="ground-truth transformation file")
    parser.add_argument("--weights", default='../model/demo.pth.tar', help="model weights file")
    return parser


def load_data(args):
    src_points = np.load(args.src_file)
    ref_points = np.load(args.ref_file)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])
    src_extra_feats = np.load(args.src_feats)
    ref_extra_feats = np.load(args.ref_feats)

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "ref_curnor": ref_extra_feats.astype(np.float32),
        "src_curnor": src_extra_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)

    return data_dict


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()

    # prepare data
    data_dict = load_data(args)
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.num_neighbors, cfg.backbone.subsample_ratio
    )

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])

    # prediction
    data_dict = to_cuda(data_dict)
    data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                              cfg.backbone.num_stages,
                              cfg.backbone.num_neighbors,
                              )
    data_dict.update(data)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]

    # compute error
    if args.gt_file is not None:
        transform = data_dict["transform"]
        rre, rte = compute_registration_error(transform, estimated_transform)
        print(f"RRE: {rre:.3f}, RTE: {rte:.3f}")


if __name__ == "__main__":
    main()
