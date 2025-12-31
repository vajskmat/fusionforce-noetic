"""
This project uses and adapts components from the PointPillars implementation:
https://github.com/zhulf0804/PointPillars

Licensed under the MIT License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lss import BevEncode

def GetVoxelParams(grid_conf):
    # Ensure voxel size is consistent in x and y dimensions
    assert grid_conf['xbound'][2] == grid_conf['ybound'][2], 'Voxel size must be the same in x and y dimensions'

    # Voxel size in x, y, z dimensions
    voxel_size = [grid_conf['xbound'][2], grid_conf['ybound'][2], grid_conf['zbound'][2]]

    # Grid size in terms of number of voxels along each dimension
    grid_size = (
        int((grid_conf['xbound'][1] - grid_conf['xbound'][0]) / voxel_size[0]),
        int((grid_conf['ybound'][1] - grid_conf['ybound'][0]) / voxel_size[1]),
        int((grid_conf['zbound'][1] - grid_conf['zbound'][0]) / (grid_conf['zbound'][1] - grid_conf['zbound'][0]))
    )

    # Matches the bounds of x, y, and z
    point_cloud_range = [
        grid_conf['xbound'][0], grid_conf['ybound'][0], grid_conf['zbound'][0],
        grid_conf['xbound'][1], grid_conf['ybound'][1], grid_conf['zbound'][1]
    ]

    # Maximum number of points per voxel (originally 32)
    max_num_points = 64

    # Compute maximum number of voxels for training and testing
    max_voxels_train = int(grid_size[0] * grid_size[1] * 1.0)  # Allow 100% of potential voxels
    max_voxels_test = int(grid_size[0] * grid_size[1] * grid_size[2] * 0.9)  # Allow 90%
    max_voxels = (max_voxels_train, max_voxels_test)

    # Final Parameters
    print(f"Voxel Size: {voxel_size}")
    print(f"Grid Size: {grid_size}")
    print(f"Point Cloud Range: {point_cloud_range}")
    print(f"Max Number of Points: {max_num_points}")
    print(f"Max Voxels (Train, Test): {max_voxels}")

    return voxel_size, point_cloud_range, max_num_points, max_voxels

class Voxelization(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer("point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32))
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels[0]
        self.register_buffer("grid_size", ((self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size).long())

    @torch.no_grad()
    def forward(self, points):
        """
        points: (N, C)
        returns:
            voxels: (M, max_num_points, C)
            coords: (M, 3) voxel indices (x,y,z)
            num_points: (M,)
        """
        device = points.device
        xyz = points[:, :3]

        # 1. map to voxel grid
        # normalize coords to voxel grid
        voxel_coords = ((xyz - self.point_cloud_range[:3]) / self.voxel_size).floor().long()
        # keep only points inside bounds
        mask = ((voxel_coords >= 0) & (voxel_coords < self.grid_size)).all(dim=1)
        voxel_coords, points = voxel_coords[mask], points[mask]

        if len(points) == 0:
            return (
                torch.zeros((0, self.max_num_points, points.shape[1]), device=device),
                torch.zeros((0, 3), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )

        # 2. unique voxel coords + inverse
        # unique voxel indices (if two points fall into the same voxel, they will be merged, [[0,0], [0, 0]] -> uniq[0]=[0, 0], 
        # inv- Gives you a mapping (inv) from each original point back to its voxel’s unique index. 
        # Example: If point #17 landed in voxel (10,20,0) and that voxel is row 5 in uniq, then inv[17] = 5.)
        uniq, inv = torch.unique(voxel_coords, dim=0, return_inverse=True)
        M = min(len(uniq), self.max_voxels)

        # Drop points from voxels beyond limit
        mask_valid_voxels = inv < M
        inv, points = inv[mask_valid_voxels], points[mask_valid_voxels]

        # 3. shuffle point order *within each voxel* before truncating
        #    (ensures unbiased sampling when many points fall into one voxel)
        rand = torch.rand(len(inv), device=device)
        sort_key = inv.to(torch.float32) + rand * 1e-6  # stable tie-breaking
        order = torch.argsort(sort_key)
        points_sorted = points[order]
        inv_sorted = inv[order]

        # 4. rank of each point inside its voxel group (0,1,2,...)
        left_idx = torch.searchsorted(inv_sorted, inv_sorted, right=False)
        rank = torch.arange(len(inv_sorted), device=device) - left_idx

        # 5. keep only first max_num_points per voxel
        keep = rank < self.max_num_points
        points_kept, voxel_ids, ranks = points_sorted[keep], inv_sorted[keep], rank[keep]

        # 6. scatter points into voxel tensor
        C = points.shape[1]
        voxels = torch.zeros((M, self.max_num_points, C), device=device, dtype=points.dtype)
        linear_index = voxel_ids * self.max_num_points + ranks
        flat_voxels = voxels.view(-1, C)
        flat_voxels[linear_index] = points_kept

        # 7. count how many points per voxel (clamped)
        num_points = torch.bincount(voxel_ids, minlength=M).clamp(max=self.max_num_points)

        coords = uniq[:M]
        
        return voxels, coords, num_points[:M]



class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''

        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):           
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)

        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i)) 
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / (npoints_per_pillar[:, None, None]).clamp(min=1) # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:,:,0:1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1) 
        y_offset_pi_center = pillars[:,:,1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1) 

        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp
        # In consitent with mmdet3d. 
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 0, 1).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, in_channel, self.x_l, self.y_l)
        return batched_canvas


class PillarNet(nn.Module):
    def __init__(self, grid_conf=None, n_features=16):
        super().__init__()
        self.grid_conf = grid_conf
        voxel_size, point_cloud_range, max_num_points, max_voxels = GetVoxelParams(grid_conf) # Voxel Size: [0.1, 0.1, 6.4],
                                                                                                # Point Cloud Range: [-6.4, -6.4, -3.2, 6.4, 6.4, 3.2],
                                                                                                # Max Number of Points: 32,
                                                                                                # Max Voxels (Train, Test): (16384, 14745)
        self.voxel_size = voxel_size
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, 
                                        point_cloud_range=point_cloud_range, 
                                        max_num_points=max_num_points, 
                                        max_voxels=max_voxels)
        
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, 
                                            point_cloud_range=point_cloud_range, 
                                            in_channel=8, # in_channel=9 if reflectivity is included
                                            out_channel=n_features)
                                        
    def forward(self, points, mode='train'): # points: (bs, C, N)
        # Filter out points with NaNs
        # and create a list of tensors for each batch points: (bs, N, 3 + c) -> batched_pts: list[tensor]
        # Vectorized valid mask: shape [B, N]
        valid_mask = ~torch.isnan(points).any(dim=1)

        batched_pts = []
        for b in range(points.size(0)):
            p = points[b, :, valid_mask[b]]  # [C, N_valid]
            batched_pts.append(p.transpose(0, 1))  # [N_valid, C]

        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c), 
        #                              coors_batch: (p1 + p2 + ... + pb, 3 + 1), 
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts) 
        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, x_l, y_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)

        # print(f"Batched points shapes: {[bp.shape for bp in batched_pts]}")
        # print(f"pillars shape (pillars: (p1 + p2 + ... + pb, num_points, c)): {pillars.shape}")
        # print(f"coors_batch shape ((p1 + p2 + ... + pb, 3 + 1)): {coors_batch.shape}")
        # print(f"npoints_per_pillar shape ((p1 + p2 + ... + pb, ), (b: batch size)): {npoints_per_pillar.shape}")
        # print("Pillar features shape (bs, out_channel, x_l, y_l):", pillar_features.shape)
        return pillar_features

class PointPillars(nn.Module):
    def __init__(self, grid_conf=None, outC=1, n_features=64): # TODO: work with the number of features 
        super().__init__()
        self.pillar_net = PillarNet(grid_conf=grid_conf, n_features=n_features)
        self.bevencode = BevEncode(in_channels=n_features, out_channels=outC)

    def forward(self, points, mode='train'):
        pillar_features = self.pillar_net(points)
        out = self.bevencode(pillar_features)

        return out
    
    def from_pretrained(self, modelf):
        if not modelf:
            return self
        print(f'Loading pretrained {self.__class__.__name__} model from', modelf)
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
        model_dict = self.state_dict()
        pretrained_model = torch.load(modelf)
        model_dict.update(pretrained_model)
        self.load_state_dict(model_dict)
        return self

        