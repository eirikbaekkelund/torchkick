import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional


class PositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional encoding.
    The idea is that 2D positional encoding for lines on a football pitch can help
    the model better localize and identify line features by providing spatial context.

    Args:
        d_model (int): Dimension of the model.
        max_h (int): Maximum height for positional encoding.
        max_w (int): Maximum width for positional encoding.
    """

    def __init__(self, d_model: int, max_h: int = 256, max_w: int = 256):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model // 2)
        self.col_embed = nn.Embedding(max_w, d_model // 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        h = min(h, self.row_embed.num_embeddings)
        w = min(w, self.col_embed.num_embeddings)
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(h, 1, 1),
                y_emb.unsqueeze(1).repeat(1, w, 1),
            ],
            dim=-1,
        )
        return pos.permute(2, 0, 1).unsqueeze(0)


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale features.
    The FPN combines low-resolution, semantically strong features with
    high-resolution, semantically weak features via lateral connections.
    The idea is to provide rich, multi-scale feature representations that
    can help in detecting lines of varying sizes and orientations on the football pitch.

    Args:
        in_channels_list (list): List of input channels for each feature map.
        out_channels (int): Number of output channels for each feature map.
    """

    def __init__(self, in_channels_list: list, out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, features: list) -> list:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outs


class DeformableAttention(nn.Module):
    """
    Deformable Attention for spatial reasoning.
    The DA mechanism allows the model to focus on relevant spatial locations and
    adaptively sample features, which is beneficial for detecting lines that may
    be partially occluded or vary in appearance.

    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        n_points (int): Number of sampling points per head.
    """

    def __init__(self, d_model: int, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.0)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.xavier_uniform_(self.output_proj.weight.data)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: tuple,
    ) -> torch.Tensor:
        B, N, C = query.shape
        H, W = spatial_shapes

        value = self.value_proj(value).view(B, H * W, self.n_heads, -1)

        offsets = self.sampling_offsets(query).view(B, N, self.n_heads, self.n_points, 2)
        offsets = offsets.tanh() * 0.5

        attn_weights = self.attention_weights(query).view(B, N, self.n_heads, self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        ref = reference_points.view(B, N, 1, 1, 2)
        sampling_locations = ref + offsets

        sampling_locations = sampling_locations * 2 - 1

        value_2d = value.permute(0, 2, 3, 1).contiguous().reshape(B * self.n_heads, -1, H, W)
        sampling_locs_2d = sampling_locations.reshape(B * self.n_heads, N * self.n_points, 1, 2)

        sampled = F.grid_sample(value_2d, sampling_locs_2d, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.view(B, self.n_heads, -1, N, self.n_points)
        sampled = sampled.permute(0, 3, 1, 4, 2)

        output = (sampled * attn_weights.unsqueeze(-1)).sum(dim=3)
        output = output.view(B, N, -1)

        return self.output_proj(output)


class IterativeRefinementDecoder(nn.Module):
    """
    DETR-style decoder with iterative refinement.
    Each layer refines the previous prediction.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.layers = nn.ModuleList()
        self.ref_point_heads = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                        "cross_attn": DeformableAttention(d_model, nhead, n_points=4),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, dim_feedforward),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim_feedforward, d_model),
                            nn.Dropout(dropout),
                        ),
                        "norm1": nn.LayerNorm(d_model),
                        "norm2": nn.LayerNorm(d_model),
                        "norm3": nn.LayerNorm(d_model),
                    }
                )
            )
            self.ref_point_heads.append(nn.Linear(d_model, 2))

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: tuple,
        reference_points: Optional[torch.Tensor] = None,
    ) -> tuple:
        B, N, C = query.shape

        if reference_points is None:
            reference_points = torch.zeros(B, N, 2, device=query.device)
            reference_points[:, :, 0] = 0.5
            reference_points[:, :, 1] = 0.5

        outputs = []
        ref_points_list = [reference_points]

        for i, layer in enumerate(self.layers):
            query = layer["norm1"](query + layer["self_attn"](query, query, query)[0])

            query = layer["norm2"](query + layer["cross_attn"](query, reference_points, memory, spatial_shapes))

            query = layer["norm3"](query + layer["ffn"](query))

            delta_ref = self.ref_point_heads[i](query).sigmoid()
            reference_points = reference_points + (delta_ref - 0.5) * 0.1
            reference_points = reference_points.clamp(0, 1)

            outputs.append(query)
            ref_points_list.append(reference_points)

        return outputs, ref_points_list


class LineRepresentationHead(nn.Module):
    """
    Predicts line representations:
    - For straight lines: (θ, ρ) Hough parameterization + 2 t-values for endpoints
    - For circles: center (cx, cy) + radius + arc angles
    - Visibility/confidence scores
    """

    def __init__(self, d_model: int, num_classes: int, max_points: int = 12):
        super().__init__()
        self.num_classes = num_classes
        self.max_points = max_points
        self.d_model = d_model

        self.line_head = nn.Sequential(
            nn.Linear(d_model * max_points, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )

        self.keypoint_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

        self.visibility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model * max_points, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]
        x = x.view(B, self.num_classes, self.max_points, self.d_model)

        x_flat = x.view(B, self.num_classes, -1)
        line_params = self.line_head(x_flat)
        theta = torch.tanh(line_params[..., 0]) * math.pi
        rho = torch.sigmoid(line_params[..., 1])
        t1 = torch.sigmoid(line_params[..., 2])
        t2 = torch.sigmoid(line_params[..., 3])

        keypoints = torch.sigmoid(self.keypoint_head(x))

        # NOTE: logits for visibility/confidence (sigmoid applied in loss for AMP safety)
        visibility_logits = self.visibility_head(x).squeeze(-1)
        confidence_logits = self.confidence_head(x_flat).squeeze(-1)

        return {
            "theta": theta,
            "rho": rho,
            "t_params": torch.stack([t1, t2], dim=-1),
            "keypoints": keypoints,
            "visibility": torch.sigmoid(visibility_logits),  # For inference
            "visibility_logits": visibility_logits,  # For training loss
            "confidence": torch.sigmoid(confidence_logits),  # For inference
            "confidence_logits": confidence_logits,  # For training loss
        }


class HeatmapOffsetHead(nn.Module):
    """
    CenterNet-style heatmap + offset head.
    Predicts coarse heatmap and fine offset for subpixel precision.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes * 2, kernel_size=1),
        )

        self.heatmap[-1].bias.data.fill_(-2.19)

    def forward(self, x: torch.Tensor) -> dict:
        heatmap = torch.sigmoid(self.heatmap(x))
        offset = self.offset(x)

        B, _, H, W = heatmap.shape
        offset = offset.view(B, self.num_classes, 2, H, W)

        return {"heatmap": heatmap, "offset": offset}


class KeyPointLineDetector(nn.Module):
    """
    State-of-the-art line detection model combining:
    1. ResNet50 backbone with FPN
    2. Deformable attention decoder with iterative refinement
    3. Dual prediction heads:
       - Heatmap + offset for dense keypoint detection
       - Query-based for line parameterization
    4. Auxiliary losses from intermediate layers
    """

    def __init__(
        self,
        num_classes: int = 28,
        max_points: int = 12,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_points = max_points
        self.d_model = d_model

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.fpn = FPN([256, 512, 1024, 2048], d_model)

        self.pos_encoding = PositionalEncoding2D(d_model)

        self.query_embed = nn.Embedding(num_classes * max_points, d_model)
        self.query_pos = nn.Embedding(num_classes * max_points, d_model)

        self.decoder = IterativeRefinementDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
        )

        self.line_heads = nn.ModuleList(
            [LineRepresentationHead(d_model, num_classes, max_points) for _ in range(num_decoder_layers)]
        )

        self.heatmap_head = HeatmapOffsetHead(d_model, num_classes)

        self.fuse_conv = nn.Conv2d(d_model * 4, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        fpn_features = self.fpn([c1, c2, c3, c4])

        target_size = fpn_features[0].shape[-2:]
        fused = torch.cat(
            [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in fpn_features],
            dim=1,
        )
        fused = self.fuse_conv(fused)

        heatmap_out = self.heatmap_head(fused)

        _, _, H, W = fused.shape
        memory = fused.flatten(2).permute(0, 2, 1)
        pos = self.pos_encoding(fused).flatten(2).permute(0, 2, 1).expand(B, -1, -1)
        memory = memory + pos

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        queries = queries + query_pos

        decoder_outputs, ref_points = self.decoder(queries, memory, (H, W))

        layer_outputs = []
        for i, (dec_out, head) in enumerate(zip(decoder_outputs, self.line_heads)):
            layer_outputs.append(head(dec_out))

        final_output = layer_outputs[-1]

        return {
            "keypoints": final_output["keypoints"],
            "visibility": final_output["visibility"],
            "visibility_logits": final_output["visibility_logits"],
            "confidence": final_output["confidence"],
            "confidence_logits": final_output["confidence_logits"],
            "theta": final_output["theta"],
            "rho": final_output["rho"],
            "t_params": final_output["t_params"],
            "heatmap": heatmap_out["heatmap"],
            "offset": heatmap_out["offset"],
            "aux_outputs": layer_outputs[:-1],
            "reference_points": ref_points,
        }


class KeyPointLoss(nn.Module):
    """
    Combined loss for line keypoint detection:
    1. Keypoint L1 loss with visibility masking
    2. Visibility BCE loss
    3. Heatmap focal loss
    4. Offset L1 loss at keypoint locations
    5. Auxiliary losses from intermediate decoder layers
    """

    def __init__(
        self,
        coord_weight: float = 5.0,
        vis_weight: float = 1.0,
        heatmap_weight: float = 1.0,
        offset_weight: float = 1.0,
        aux_weight: float = 0.5,
    ):
        super().__init__()
        self.coord_weight = coord_weight
        self.vis_weight = vis_weight
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.aux_weight = aux_weight

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0):
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        neg_weights = torch.pow(1 - target, beta)

        pos_loss = -torch.pow(1 - pred, alpha) * torch.log(pred.clamp(min=1e-6)) * pos_mask
        neg_loss = -torch.pow(pred, alpha) * torch.log((1 - pred).clamp(min=1e-6)) * neg_weights * neg_mask

        num_pos = pos_mask.sum()
        if num_pos > 0:
            return (pos_loss.sum() + neg_loss.sum()) / num_pos
        return neg_loss.sum()

    def forward(
        self,
        outputs: dict,
        gt_keypoints: torch.Tensor,
        gt_visibility: torch.Tensor,
        gt_heatmaps: Optional[torch.Tensor] = None,
    ) -> dict:
        pred_coords = outputs["keypoints"]
        # logits for AMP-safe loss computation
        pred_vis_logits = outputs.get("visibility_logits", outputs["visibility"])

        vis_mask = gt_visibility > 0.5

        if vis_mask.sum() > 0:
            coord_loss = F.l1_loss(pred_coords[vis_mask], gt_keypoints[vis_mask], reduction="mean")
        else:
            coord_loss = torch.tensor(0.0, device=pred_coords.device)

        # binary_cross_entropy_with_logits for AMP safety
        vis_loss = F.binary_cross_entropy_with_logits(pred_vis_logits, gt_visibility, reduction="mean")

        heatmap_loss = torch.tensor(0.0, device=pred_coords.device)
        offset_loss = torch.tensor(0.0, device=pred_coords.device)

        if gt_heatmaps is not None and "heatmap" in outputs:
            pred_heatmap = outputs["heatmap"]
            if pred_heatmap.shape[-2:] != gt_heatmaps.shape[-2:]:
                gt_heatmaps = F.interpolate(
                    gt_heatmaps.unsqueeze(0) if gt_heatmaps.dim() == 3 else gt_heatmaps,
                    size=pred_heatmap.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            heatmap_loss = self.focal_loss(pred_heatmap, gt_heatmaps)

        aux_loss = torch.tensor(0.0, device=pred_coords.device)
        if "aux_outputs" in outputs:
            for aux_out in outputs["aux_outputs"]:
                aux_coords = aux_out["keypoints"]
                aux_vis_logits = aux_out.get("visibility_logits", aux_out["visibility"])
                if vis_mask.sum() > 0:
                    aux_loss = aux_loss + F.l1_loss(aux_coords[vis_mask], gt_keypoints[vis_mask], reduction="mean")
                aux_loss = aux_loss + F.binary_cross_entropy_with_logits(
                    aux_vis_logits, gt_visibility, reduction="mean"
                )
            aux_loss = aux_loss / max(len(outputs["aux_outputs"]), 1)

        total = (
            self.coord_weight * coord_loss
            + self.vis_weight * vis_loss
            + self.heatmap_weight * heatmap_loss
            + self.offset_weight * offset_loss
            + self.aux_weight * aux_loss
        )

        return {
            "total": total,
            "coord_loss": coord_loss,
            "vis_loss": vis_loss,
            "heatmap_loss": heatmap_loss,
            "offset_loss": offset_loss,
            "aux_loss": aux_loss,
        }


if __name__ == "__main__":
    print("Testing KeyPoint Line Detector...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 540, 960).to(device)
    model = KeyPointLineDetector(num_classes=28, max_points=12).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    with torch.no_grad():
        out = model(x)

    loss_fn = KeyPointLoss()
    gt_coords = torch.rand(2, 28, 12, 2).to(device)
    gt_vis = (torch.rand(2, 28, 12) > 0.7).float().to(device)

    losses = loss_fn(out, gt_coords, gt_vis)
    print(f"  Total: {losses['total']:.4f}")
    print(f"  Coord: {losses['coord_loss']:.4f}")
    print(f"  Vis: {losses['vis_loss']:.4f}")
    print(f"  Aux: {losses['aux_loss']:.4f}")
