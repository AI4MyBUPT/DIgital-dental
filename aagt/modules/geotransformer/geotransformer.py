import pdb

import numpy as np
import torch
import torch.nn as nn

from aagt.modules.ops import pairwise_distance
from aagt.modules.transformer import SinusoidalPositionalEmbedding, RPEConditionalTransformer

class CNEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, sigma_c=0.03, dropout_rate=0.1):
        super(CNEmbedding, self).__init__()
        
        self.log_sigma_d = nn.Parameter(torch.log(torch.tensor(sigma_d)))
        self.log_sigma_a = nn.Parameter(torch.log(torch.tensor(sigma_a)))
        self.log_sigma_c = nn.Parameter(torch.log(torch.tensor(sigma_c)))
        
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_dist = nn.Linear(hidden_dim, hidden_dim)
        self.proj_ppf = nn.Linear(hidden_dim * 3, hidden_dim)
        self.proj_curv = nn.Linear(hidden_dim, hidden_dim)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, points, curvatures, normals):
        """
        Args:
            points: (B, N, 3)
            normals: (B, N, 3) normalized
            curvatures: (B, N, 1)
        """
        batch_size, num_point, _ = points.shape
        
        diff_map = points.unsqueeze(1) - points.unsqueeze(2) 
        dist_map = torch.norm(diff_map, dim=-1)
        safe_dist = torch.clamp(dist_map, min=1e-6)
        dir_map = diff_map / safe_dist.unsqueeze(-1)
        
        n_i = normals.unsqueeze(2)
        n_j = normals.unsqueeze(1)
        
        cos_alpha = torch.clamp(torch.sum(n_i * dir_map, dim=-1), -1 + 1e-6, 1 - 1e-6)
        cos_beta  = torch.clamp(torch.sum(n_j * dir_map, dim=-1), -1 + 1e-6, 1 - 1e-6)
        cos_gamma = torch.clamp(torch.sum(n_i * n_j, dim=-1),     -1 + 1e-6, 1 - 1e-6)
        
        angle_alpha = torch.acos(cos_alpha)
        angle_beta  = torch.acos(cos_beta)
        angle_gamma = torch.acos(cos_gamma)
        
        c_i = curvatures.unsqueeze(2)
        c_j = curvatures.unsqueeze(1)
        curv_diff = torch.abs(c_i - c_j).squeeze(-1)
        sigma_d = torch.exp(self.log_sigma_d)
        sigma_a = torch.exp(self.log_sigma_a)
        sigma_c = torch.exp(self.log_sigma_c)     
        emb_d = self.proj_dist(self.embedding(dist_map / sigma_d))
        feat_a1 = self.embedding(angle_alpha / sigma_a)
        feat_a2 = self.embedding(angle_beta / sigma_a)
        feat_a3 = self.embedding(angle_gamma / sigma_a)
        emb_ppf = self.proj_ppf(torch.cat([feat_a1, feat_a2, feat_a3], dim=-1))  
        emb_c = self.proj_curv(self.embedding(curv_diff / sigma_c))

        concat_features = torch.cat([emb_d, emb_ppf, emb_c], dim=-1)
        
        final_embedding = self.fusion_mlp(concat_features)
        
        return final_embedding

class GeometryAwareEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, sigma_c=0.03):
        super(GeometryAwareEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.sigma_c = sigma_c
        
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        
        self.proj_dist = nn.Linear(hidden_dim, hidden_dim)
        self.proj_angle_i = nn.Linear(hidden_dim, hidden_dim) 
        self.proj_angle_j = nn.Linear(hidden_dim, hidden_dim) 
        self.proj_angle_nn = nn.Linear(hidden_dim, hidden_dim) 
        self.proj_curv = nn.Linear(hidden_dim, hidden_dim)    
        
        self.output_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, points, curvatures, normals):
        batch_size, num_point, _ = points.shape
        diff_map = points.unsqueeze(1) - points.unsqueeze(2) 
        
        dist_map = torch.norm(diff_map, dim=-1) # (B, N, N)
        
        safe_dist = torch.clamp(dist_map, min=1e-6)
        dir_map = diff_map / safe_dist.unsqueeze(-1) # (B, N, N, 3) Unit vectors from i to j
        n_i = normals.unsqueeze(2)      # Source normals (row)
        n_j = normals.unsqueeze(1)      # Target normals (col)
        
        # Angle 1: n_i dot dir_ij
        cos_alpha = torch.sum(n_i * dir_map, dim=-1) # (B, N, N)
        # Angle 2: n_j dot dir_ij
        cos_beta = torch.sum(n_j * dir_map, dim=-1)  # (B, N, N)
        # Angle 3: n_i dot n_j
        cos_gamma = torch.sum(n_i * n_j, dim=-1)     # (B, N, N)
        angle_alpha = torch.acos(torch.clamp(cos_alpha, -1 + 1e-6, 1 - 1e-6))
        angle_beta  = torch.acos(torch.clamp(cos_beta,  -1 + 1e-6, 1 - 1e-6))
        angle_gamma = torch.acos(torch.clamp(cos_gamma, -1 + 1e-6, 1 - 1e-6))
        
        c_i = curvatures.unsqueeze(2)
        c_j = curvatures.unsqueeze(1)
        curv_diff = torch.abs(c_i - c_j).squeeze(-1) # (B, N, N)
        emb_d = self.proj_dist(self.embedding(dist_map / self.sigma_d))
        emb_a1 = self.proj_angle_i(self.embedding(angle_alpha / self.sigma_a))
        emb_a2 = self.proj_angle_j(self.embedding(angle_beta / self.sigma_a))
        emb_a3 = self.proj_angle_nn(self.embedding(angle_gamma / self.sigma_a))
        emb_c = self.proj_curv(self.embedding(curv_diff / self.sigma_c))
        
        total_embedding = emb_d + emb_a1 + emb_a2 + emb_a3 + emb_c
        final_embedding = self.output_fusion(total_embedding)
        
        return final_embedding

class SurfaceGeometricEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d):
        super(SurfaceGeometricEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.hidden_dim = hidden_dim
        
        self.dist_embed = SinusoidalPositionalEmbedding(hidden_dim)
        self.angle_embed = SinusoidalPositionalEmbedding(hidden_dim)
        self.curv_embed = SinusoidalPositionalEmbedding(hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, points, curvatures, normals):
        B, N, _ = points.shape
       
        dist_vec = points.unsqueeze(2) - points.unsqueeze(1) 
        dist = torch.norm(dist_vec, dim=-1) # (B, N, N)
        
        safe_dist = torch.clamp(dist, min=1e-6)
        normalized_dist_vec = dist_vec / safe_dist.unsqueeze(-1)
        
        d_emb = self.dist_embed(dist / self.sigma_d) # (B, N, N, C)

        n_i = normals.unsqueeze(2).expand(B, N, N, 3) # (B, N, N, 3)
        n_j = normals.unsqueeze(1).expand(B, N, N, 3) # (B, N, N, 3)
        
        cos_ni_nj = torch.sum(n_i * n_j, dim=-1) # (B, N, N)
    
        cos_ni_d = torch.sum(n_i * normalized_dist_vec, dim=-1) # (B, N, N)
        
        cos_nj_d = torch.sum(n_j * normalized_dist_vec, dim=-1) # (B, N, N)
        
        ang_emb_1 = self.angle_embed(cos_ni_nj)
        ang_emb_2 = self.angle_embed(cos_ni_d)
        ang_emb_3 = self.angle_embed(cos_nj_d)
        
        c_i = curvatures.unsqueeze(2).expand(B, N, N, 1)
        c_j = curvatures.unsqueeze(1).expand(B, N, N, 1)
        
        c_mean = (c_i + c_j) * 0.5

        curv_emb = self.curv_embed(c_mean.squeeze(-1)) # (B, N, N, C)
        cat_embeddings = torch.cat([
            d_emb, 
            ang_emb_1, 
            ang_emb_2, 
            ang_emb_3, 
            curv_emb
        ], dim=-1)
        
        final_embedding = self.fusion_mlp(cat_embeddings)
        
        return final_embedding

class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

        Args:
            points: torch.Tensor (B, N, 3), input point cloud

        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        batch_size, num_point, _ = points.shape

        dist_map = torch.sqrt(pairwise_distance(points, points))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        k = self.angle_k
        knn_indices = dist_map.topk(k=k + 1, dim=2, largest=False)[1][:, :, 1:]  # (B, N, k)
        knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, k, 3)  # (B, N, k, 3)
        expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
        ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)

        # ref_vectors = normals.unsqueeze(2)

        anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
        ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, k, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a

        return d_indices, a_indices

    def forward(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)

        d_embeddings = self.embedding(d_indices)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(a_indices)
        a_embeddings = self.proj_a(a_embeddings)
        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=3)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=3)
        # a_embeddings = a_embeddings[:, :, :, 0, :]
        embeddings = d_embeddings + a_embeddings

        return embeddings


class GeometricTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        blocks,
        sigma_d,
        sigma_a,
        angle_k,
        dropout=None,
        activation_fn='ReLU',
        reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).

        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)
        self.embedding_CN = CNEmbedding(hidden_dim, sigma_d, sigma_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn, return_attention_scores=True, parallel=False
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        ref_points,
        src_points,
        ref_feats,
        src_feats,
        ref_points_extra,
        src_points_extra,
        ref_feats_extra,
        src_feats_extra,
        ref_curvature,
        src_curvature,
        ref_normals,
        src_normals,
        ref_masks=None,
        src_masks=None,
    ):
        r"""Geometric Transformer

        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)
        ref_embeddings_extra = self.embedding_CN(ref_points_extra, ref_curvature, ref_normals)
        src_embeddings_extra = self.embedding_CN(src_points_extra, src_curvature, src_normals)
        ref_feats = self.in_proj(ref_feats)
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats, scores_list = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            ref_feats_extra,
            src_feats_extra,
            ref_embeddings_extra,
            src_embeddings_extra,
            masks0=ref_masks,
            masks1=src_masks,
        )

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)

        return ref_feats, src_feats, scores_list
