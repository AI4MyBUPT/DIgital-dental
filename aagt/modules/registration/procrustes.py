import pdb

import torch
import torch.nn as nn
import ipdb



# def solve_local_rotations(Am, Bm, weights=None, weight_threshold=0):
#     """
#     Input:
#         - A:       [bs, num_corr, 3], source point cloud
#         - B:       [bs, num_corr, 3], target point cloud
#         - weights: [bs, num_corr]     weight for each correspondence
#         - weight_threshold: float,    clips points with weight below threshold
#     Output:
#         - R, t
#     """
#     bs = Am.shape[0]
#     if weights is None:
#         weights = torch.ones_like(Am[:, :, 0])
#     weights[weights < weight_threshold] = 0

#     # construct weight covariance matrix
#     Weight = torch.diag_embed(weights)
#     H = Am.permute(0, 2, 1) @ Weight @ Bm

#     # find rotation
#     U, S, Vt = torch.svd(H.cpu())
#     U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
#     delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
#     eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(Am.device)
#     eye[:, -1, -1] = delta_UV
#     R = Vt @ eye @ U.permute(0, 2, 1)
#     return R

def solve_local_rotations(Am, Bm, weights=None, weight_threshold=0):
    """
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = Am.shape[0]
    device = Am.device
    
    if weights is None:
        weights = torch.ones((bs, Am.shape[1]), device=device, dtype=Am.dtype)
    weights[weights < weight_threshold] = 0

    # --- 修改 1: 提升计算精度 (Float32 -> Double) ---
    # 在计算 H 之前就转为 double，防止权重极小时出现数值下溢
    Am_dbl = Am.double()
    Bm_dbl = Bm.double()
    weights_dbl = weights.double()

    # construct weight covariance matrix
    # H 此时是 double 类型
    Weight_dbl = torch.diag_embed(weights_dbl)
    H = Am_dbl.permute(0, 2, 1) @ Weight_dbl @ Bm_dbl
    
    # --- 修改 2: 正则化与 SVD (解决不收敛的核心) ---
    H_cpu = H.cpu()
    
    # 加上 epsilon 防止矩阵病态 (Ill-conditioned)
    epsilon = 1e-7
    if H_cpu.shape[-1] == 3:
        identity = torch.eye(3, dtype=torch.float64, device='cpu').unsqueeze(0)
        H_cpu = H_cpu + identity * epsilon

    # find rotation
    try:
        # 这里的 Vt 其实是 V (torch.svd 返回 U, S, V)
        U, S, Vt = torch.svd(H_cpu)
    except RuntimeError:
        raise(f"[Error] SVD failed in solve_local_rotations")
        # 兜底策略：返回单位阵
        # U = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(bs, 1, 1)
        # Vt = U.clone()

    # --- 修改 3: 保持 Double 精度计算 R ---
    # 移回 GPU (或者原设备)
    U = U.to(device)
    Vt = Vt.to(device)
    
    # 计算行列式用于修正反射 (Reflection)
    # 保持 double 精度计算 det
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    
    # 构造 eye 矩阵 (确保是 double)
    eye = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0).repeat(bs, 1, 1)
    eye[:, -1, -1] = delta_UV
    
    # 计算 R (Double)
    R_dbl = Vt @ eye @ U.permute(0, 2, 1)
    
    # 最后转回原始精度 (Float32) 返回
    return R_dbl.type(Am.dtype)

def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    # H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    # U, _, V = torch.svd(H.cpu())  # H = USV^T
    # Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    # eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    # eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    # R = V @ eye @ Ut

    # t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    # t = t.squeeze(2)
     # 1. 转换输入为 double (Float64) 以提升计算 H 的精度
    # 相比先算 H 再转 double，从源头转 double 能保留更多细节
    src_p_dbl = src_points_centered.double()
    ref_p_dbl = ref_points_centered.double()
    weights_dbl = weights.double()
    # 2. 计算协方差矩阵 H (Double 精度)
    H = src_p_dbl.permute(0, 2, 1) @ (weights_dbl * ref_p_dbl)
    H_cpu = H.cpu() # 保持在 CPU 上计算 SVD
    # 3. [关键修改] 正则化：解决 Ill-conditioned 报错的核心
    # 给对角线加上极小的 epsilon，防止矩阵秩亏或奇异值过于接近
    epsilon = 1e-7
    if H_cpu.shape[-1] == 3:
        # 创建单位阵，注意要在 CPU 上
        identity = torch.eye(3, dtype=torch.float64, device='cpu').unsqueeze(0)
        H_cpu = H_cpu + identity * epsilon

    # 4. 执行 SVD (此时 H_cpu 既是 double 又是 regularized 的)
    try:
        U, _, V = torch.svd(H_cpu)
    except RuntimeError:
        # 极端的兜底策略：如果还是崩，返回单位阵，防止训练中断
        raise("SVD failed. Returning Identity.")
        # batch_sz = H.shape[0]
        # U = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(batch_sz, 1, 1)
        # V = U.clone()

    # 5. 恢复到 GPU 并计算 R
    # 此时 U, V 都是 double，计算 R 时保持 double 以确保正交性
    device = src_points_centered.device
    Ut = U.transpose(1, 2).to(device)
    V = V.to(device)

    eye = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))

    R_dbl = V @ eye @ Ut

    # 6. 计算 t (在 double 精度下计算)
    # 原始代码中 t 的计算直接用了 permute 后的结果，这里保持一致
    ref_c_dbl = ref_centroid.double()
    src_c_dbl = src_centroid.double()

    t_dbl = ref_c_dbl.permute(0, 2, 1) - R_dbl @ src_c_dbl.permute(0, 2, 1)
    t_dbl = t_dbl.squeeze(2)

    # 7. 最后转回 Float32 接入网络后续
    R = R_dbl.float()
    t = t_dbl.float()

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t
def cal_leading_eigenvector( M, method='power'):
    """
    Calculate the leading eigenvector using power iteration algorithm or torch.symeig
    Input:
        - M:      [bs, num_corr, num_corr] the compatibility matrix
        - method: select different method for calculating the learding eigenvector.
    Output:
        - solution: [bs, num_corr] leading eigenvector
    """
    if method == 'power':
        # power iteration algorithm
        leading_eig = torch.ones_like(M[:, :, 0:1])
        leading_eig_last = leading_eig
        for i in range(10):
            leading_eig = torch.bmm(M, leading_eig)
            leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
            if torch.allclose(leading_eig, leading_eig_last):
                break
            leading_eig_last = leading_eig
        leading_eig = leading_eig.squeeze(-1)
        return leading_eig
    elif method == 'eig':  # cause NaN during back-prop
        e, v = torch.symeig(M, eigenvectors=True)
        leading_eig = v[:, :, -1]
        return leading_eig
    else:
        exit(-1)
def soft_weight(src_points, ref_points, valid=None):
    knn_M = torch.norm(src_points[:, :, None, :] - src_points[:, None, :, :], 2, -1) - torch.norm(ref_points[:, :, None, :] - ref_points[:, None, :, :], 2, -1)
    knn_M = torch.clamp(1 - knn_M ** 2 / 0.3 ** 2, min=0)
    if valid is not None:
        knn_M.masked_fill_(~(valid * valid.permute(0, 2, 1)), 0.)
    knn_M[:, torch.arange(knn_M.shape[1]), torch.arange(knn_M.shape[1])] = 0
    weights = cal_leading_eigenvector(knn_M)
    return weights


def procrustes(
    src_points,
    ref_points,
    valid_points=None,
    return_transform=False,
    src_feats=None,
    ref_feats=None
):
    r"""Compute rigid transformation from `src_points` to `ref_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3,)
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        valid_points = valid_points.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    valid_points = valid_points.unsqueeze(2)
    # weights = soft_weight(src_points, ref_points, valid_points)
    # valid_points = weights.unsqueeze(2)

    src_centroid = torch.sum(src_points * valid_points, dim=1, keepdim=True) / valid_points.sum(dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * valid_points, dim=1, keepdim=True) / valid_points.sum(dim=1, keepdim=True)# (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    if src_feats is not None:
        # src_points_centered = torch.cat([src_points_centered, src_feats], 1)
        # ref_points_centered = torch.cat([ref_points_centered, ref_feats], 1)
        # valid_points = torch.cat([valid_points, torch.ones_like(src_feats[:, :, :1])], 1)

        src_points_centered = src_feats
        ref_points_centered = ref_feats
        valid_points = soft_weight(src_feats, ref_feats).unsqueeze(2)

    # 原来的单精度计算，在参数设置出现问题时可能会因为病态矩阵导致训练中断
    # H = src_points_centered.permute(0, 2, 1) @ (valid_points * ref_points_centered)
    # U, _, V = torch.svd(H.cpu())  # H = USV^T
    # Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    # eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    # eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    # R = V @ eye @ Ut

    # t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    # t = t.squeeze(2)


    # 1. 转换输入为 double (Float64) 以提升计算 H 的精度
    # 相比先算 H 再转 double，从源头转 double 能保留更多细节
    src_p_dbl = src_points_centered.double()
    ref_p_dbl = ref_points_centered.double()
    valid_p_dbl = valid_points.double()
    # 2. 计算协方差矩阵 H (Double 精度)
    H = src_p_dbl.permute(0, 2, 1) @ (valid_p_dbl * ref_p_dbl)
    H_cpu = H.cpu() # 保持在 CPU 上计算 SVD
    # 3. [关键修改] 正则化：解决 Ill-conditioned 报错的核心
    # 给对角线加上极小的 epsilon，防止矩阵秩亏或奇异值过于接近
    epsilon = 1e-7
    if H_cpu.shape[-1] == 3:
        # 创建单位阵，注意要在 CPU 上
        identity = torch.eye(3, dtype=torch.float64, device='cpu').unsqueeze(0)
        H_cpu = H_cpu + identity * epsilon

    # 4. 执行 SVD (此时 H_cpu 既是 double 又是 regularized 的)
    try:
        U, _, V = torch.svd(H_cpu)
    except RuntimeError:
        # 极端的兜底策略：如果还是崩，返回单位阵，防止训练中断
        raise("SVD failed. Returning Identity.")
        # batch_sz = H.shape[0]
        # U = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(batch_sz, 1, 1)
        # V = U.clone()

    # 5. 恢复到 GPU 并计算 R
    # 此时 U, V 都是 double，计算 R 时保持 double 以确保正交性
    device = src_points_centered.device
    Ut = U.transpose(1, 2).to(device)
    V = V.to(device)

    eye = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))

    R_dbl = V @ eye @ Ut

    # 6. 计算 t (在 double 精度下计算)
    # 原始代码中 t 的计算直接用了 permute 后的结果，这里保持一致
    ref_c_dbl = ref_centroid.double()
    src_c_dbl = src_centroid.double()

    t_dbl = ref_c_dbl.permute(0, 2, 1) - R_dbl @ src_c_dbl.permute(0, 2, 1)
    t_dbl = t_dbl.squeeze(2)

    # 7. 最后转回 Float32 接入网络后续
    R = R_dbl.float()
    t = t_dbl.float()

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0.0, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(self, src_points, tgt_points, weights=None):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
        )

