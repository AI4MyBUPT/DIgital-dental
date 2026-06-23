import torch
import torch.nn as nn
from aagt.modules.layers import VNLinear, VNLinearLeakyReLU, VNStdFeature
from backbone import PARE_Conv_Block, PARE_Conv_Resblock

class CNPointWeightNet(nn.Module):
    def __init__(self, init_dim=32, output_dim=64, k_neighbors=16, conv_way='edge_conv', use_xyz=True):
        super(CNPointWeightNet, self).__init__()
        self.k = k_neighbors
        conv_info = {'conv_way': conv_way, 'use_xyz': use_xyz}
        
        # Shared Layers    
        # 1. Vector Neuron Layers
        # Input: [N, 1, 3]
        self.conv_in = PARE_Conv_Block(1, init_dim, kernel_size=k_neighbors)
        
        # ResBlocks for Deep Feature Extraction
        self.res_block1 = PARE_Conv_Resblock(init_dim, init_dim, k_neighbors, conv_info=conv_info)
        self.res_block2 = PARE_Conv_Resblock(init_dim, output_dim, k_neighbors, shortcut_linear=True, conv_info=conv_info)
        
        # 2. Rotation-invariant transformation (Vector -> Scalar)
        self.std_feature = VNStdFeature(output_dim, dim=3, normalize_frame=True)
        
        # 3. Weight prediction head
        mlp_in_dim = output_dim * 3 + 1 
        
        self.weight_head = nn.Sequential(
            nn.Linear(mlp_in_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def compute_knn_graph(self, pts, k):
        """Dynamically compute the KNN graph"""
        # pts: [N, 3]
        dist = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(0) # [N, N]
        _, indices = dist.topk(k, dim=1, largest=False) # [N, k]
        return indices

    def _forward_single(self, pts, feats):
        pts = pts.float() 
        feats = feats.float()
        # 1. Feature separation
        # feats: [N, 4] -> normals[N, 3], curvatures[N, 1]
        normals = feats[:, :3]
        curvature = feats[:, 3:]

        # 2. Dynamic Composition
        neighbor_indices = self.compute_knn_graph(pts, self.k)
        
        # 3. Prepare VN input
        # [N, 1, 3]
        x_vec = normals.unsqueeze(1) 
        
        # 4. VN Backbone Feature Extraction
        x_vec = self.conv_in(pts, pts, x_vec, neighbor_indices)
        x_vec = self.res_block1(pts, pts, x_vec, neighbor_indices)
        x_vec = self.res_block2(pts, pts, x_vec, neighbor_indices) # -> [N, output_dim, 3]
    
        x_inv, _ = self.std_feature(x_vec) # -> [N, output_dim*3]
        if x_inv.dim() == 3:
            x_inv = x_inv.reshape(x_inv.shape[0], -1)
        final_feats = torch.cat([x_inv, curvature], dim=1)
        
        # 5. Prediction Weight
        weights = self.weight_head(final_feats) # [N, 1]
        
        return weights

    def forward(self, data_dict):
        src_points = data_dict['points_extra'][0]   # Source points
        ref_points = data_dict['points_extra'][-1]  # Reference points
        
        src_feats = data_dict['feats_extra'][0]     # Source feats [N, 4]
        ref_feats = data_dict['feats_extra'][-1]    # Reference feats [M, 4]
        
        src_weights = self._forward_single(src_points, src_feats)
        ref_weights = self._forward_single(ref_points, ref_feats)
        
        return src_weights, ref_weights

def create_encoder(encoder_type='esw'):
    if encoder_type == 'esw':
        return CNPointWeightNet()
    else:
        raise ValueError(f"Unkown Weighting Module Type: {encoder_type}")

   