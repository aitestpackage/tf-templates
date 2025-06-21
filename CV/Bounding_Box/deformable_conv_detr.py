class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.path1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride),
        )

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        return x1 + x2
    
class BBoxHead(nn.Module):
    # Just alot of MLP layers
    # Adding Residual because I think that's why it's hard to train
    def __init__(self, in_shape, out_shape, middle_dim=256, num_layers=4, act_fn=nn.GELU):
        super().__init__()
        self.in_proj = nn.Linear(in_shape, middle_dim)
        self.out_proj = nn.Linear(middle_dim, out_shape)
        self.middle = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(middle_dim, middle_dim),
                    act_fn(),
                )
             for i in range(num_layers - 2)
            ]
        )
        self.act_fn = act_fn

    def forward(self, x):
        x = self.act_fn()(self.in_proj(x))
        for layer in self.middle:
            x = layer(x) + x
        return nn.Sigmoid()(self.out_proj(x)) # bounding box is scaled between 0 and 1
    
import torchvision.ops as ops

class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ops.DeformConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, 2 * 1 * 3 * 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.offset_scale = nn.Parameter(torch.tensor(3.0))
    def forward(self, x):
        pred_offset = self.offset(x) * self.offset_scale
        x = self.conv(x, pred_offset)
        x = nn.GELU()(x)
        return x

class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dconv1 = DeformConvBlock(in_channels, out_channels)
        self.dconv2 = DeformConvBlock(out_channels, out_channels)
        self.dconv3 = DeformConvBlock(out_channels, out_channels)
    def forward(self, x):
        x = self.dconv1(x)
        x = self.dconv2(x) + x
        x = self.dconv3(x) + x
        return x
    
from torchvision.models import *

class SimpleDETR(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, n_heads, num_layers, num_classes):
        super().__init__()

        # self.backbone = SimpleBackBone(3, 64)
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.proj = Projection(512, hidden_dim)

        # self.proj = nn.Sequential(
        #     nn.Conv2d(512, 128, kernel_size=3, padding=1, stride=1),
        #     nn.GELU(),
        #     nn.Conv2d(128, hidden_dim, kernel_size=1, padding=0, stride=1),
        #     nn.GELU(),
        # )
        sine_embed = torch.zeros(hidden_dim, 32, 32)
        sine_embed = sine_embed + torch.arange(32).expand(hidden_dim, 32, 32)
        sine_embed = sine_embed + torch.arange(32).expand(hidden_dim, 32, 32).permute(0, 2, 1)
        sine_embed = sine_embed + torch.arange(hidden_dim).expand(32, 32, hidden_dim).permute(2, 0, 1)
        sine_embed = torch.sin(sine_embed).unsqueeze(0)
        self.pos_embed = nn.Parameter(sine_embed) # 32 is max image width/height after backbone
        self.pos_embed_weight = nn.Parameter(torch.tensor(0.1))
        self.transformer = nn.Transformer(
            hidden_dim, 
            n_heads, 
            num_layers[0], 
            num_layers[1], 
            ffn_dim, 
            activation = "gelu",
            dropout=0.1,
            batch_first=True
        )

        self.query_1 = nn.Sequential(
            nn.Conv1d(64, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Linear(80 * 80, 10)
        )
        self.query_2 = nn.Sequential(
            nn.Conv1d(128, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Linear(40 * 40, 10)
        )
        self.query_3 = nn.Sequential(
            nn.Conv1d(256, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Linear(20 * 20, 10)
        )
        self.box_query = nn.Linear(10 * 10, 10)
        self.box_embed = nn.Parameter(torch.rand(1, 10, hidden_dim))
        self.query_weights = nn.Parameter(torch.tensor([0.05, 0.05, 0.2, 1.0, 0.1]))

        self.fc_class = nn.Linear(hidden_dim, num_classes + 1) # +1 for the <no object> class
        self.fc_bbox = BBoxHead(hidden_dim, 4, num_layers=3)

        self.impt_pts = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
            nn.Conv2d(128, 2, kernel_size=1, padding=0, stride=1),
            nn.Softmax(dim=1),
        ) # custom auxiliary loss to predict whether a point is important

        self.bbox_sizes = BBoxHead(hidden_dim, 2, num_layers=3) # custom auxiliary loss to predict size of bboxes
        # hopefully help with small ones
        
    def forward(self, x):
        batch_size = x.shape[0]
        w = x.shape[2]
        h = x.shape[3]
        
        # BACKBONE PORTION
        # x = self.backbone(x)
        x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.maxpool(x)
        # print(x.shape)
        # 80 by 80
        x1 = self.backbone.layer1(x)
        # print(x1.shape)
        # 80 by 80
        x2 = self.backbone.layer2(x1)
        # print(x2.shape)
        # 40 by 40
        x3 = self.backbone.layer3(x2)
        # print(x3.shape)
        # 20 by 20
        x = self.backbone.layer4(x3)
        # print(x.shape)
        # END BACKBONE PORTION
        
        x = x.reshape(batch_size, 512, w//32, h//32)
        x = self.proj(x)
        batch_size = x.shape[0]
        hidden = x.shape[1]
        w = x.shape[2]
        h = x.shape[3]
        x = x + self.pos_embed[:, :, :w, :h] * self.pos_embed_weight
        # [batch size, hidden_dim, width, height]
        impt_pixels = self.impt_pts(x)
        # [batch size, 2, width, height]
        x = x.reshape(batch_size, hidden, -1)
        seq_len = x.shape[2]
        # [batch size, hidden_dim, width * height]
        # Treat it like [batch size, hidden_dim, sequence len] WILL PERMUTE LATER

        box_Q = self.query_weights[3] * self.box_query(x).permute(0, 2, 1)
        # [batch size, 100, hidden_dim]
        box_Q = box_Q + self.query_weights[4] * self.box_embed
        # [batch size, 100, hidden_dim]
        # x1 = model.backbone.layer1[1](x1)
        x1 = rearrange(x1, 'b c w h -> b c (w h)')
        box_Q = box_Q + self.query_weights[0] * self.query_1(x1).permute(0, 2, 1)
        # x2 = model.backbone.layer2[1](x2)
        x2 = rearrange(x2, 'b c w h -> b c (w h)')
        box_Q = box_Q + self.query_weights[1] * self.query_2(x2).permute(0, 2, 1)
        # x3 = model.backbone.layer3[1](x3)
        x3 = rearrange(x3, 'b c w h -> b c (w h)')
        box_Q = box_Q + self.query_weights[2] * self.query_3(x3).permute(0, 2, 1)
        # [batch size, 100, hidden_dim]
        x = x.permute(0, 2, 1)
        # [batch size, width * height, hidden_dim] # w * h = seq len
        x = self.transformer(x, box_Q) # encoder input and decoder input
        # [batch size, 100, hidden_dim] comes from box_Q aka decoder input length

        class_pred = self.fc_class(x)
        # [batch size, 100, num classes + 1]
        box_pred = self.fc_bbox(x)
        # [batch size, 100, 4]
        bbox_size = self.bbox_sizes(x)
        # [batch size, 100, 2]
        return class_pred, box_pred, impt_pixels, bbox_size
    
class_loss_weights = torch.ones(20 + 1).to(device)
class_loss_weights[-1] = class_loss_weights[-1] / 20
class_loss_weights = class_loss_weights / torch.sum(class_loss_weights)
class_loss_weights

from scipy.optimize import linear_sum_assignment
# just import package to do bipartite matching
from torchvision.ops import generalized_box_iou, box_iou, generalized_box_iou_loss, box_convert

class_criteria = nn.CrossEntropyLoss(reduction="none", weight=None, label_smoothing=0.05)
no_labels_target_id = 20

pixel_criteria = nn.CrossEntropyLoss()

bbox_size_criteria = nn.SmoothL1Loss()

def match_bbox_class(
    class_pred_in, 
    box_pred_in, 
    pred_pixels, 
    bbox_size_in,
    target_classes_in, 
    target_bbox_in, 
    target_pixels, 
    weights=[2, 5, 0]
):
    # class_pred: [batch size, 100, num_classes + 1]
    # box_pred: [batch size, 100, 4]
    # target_classes: not batched
    # target_bbox: not batched
    batch_size, num_bb, num_class = class_pred_in.shape
    # num_class already contain the +1

    # Stores the total loss
    final_loss = 0
    # Stores total IoU
    total_iou = 0

    # Stores each type of loss
    total_class_loss = 0
    total_bbox_loss = 0
    total_no_label_loss = 0
    total_mask_loss = 0
    total_size_loss = 0

    # For-loop over batch size dimension because input can't really be batched efficiently
    for i in range(batch_size):
        class_pred = class_pred_in[i]
        box_pred = box_pred_in[i]
        bbox_size = bbox_size_in[i]
        target_classes = target_classes_in[i]
        target_bbox = target_bbox_in[i]
        target_len = len(target_classes)
        
        # CLASS LOSS
        class_pred = class_pred.unsqueeze(1).expand(-1, target_len, -1)
        # class_pred: [100, target_len, num_classes + 1]
        target_classes = target_classes.unsqueeze(0).expand(num_bb, -1) 
        # note that this is transposed relative to above to ensure the correct targets lines up
        # and we actually get proper [i, j] pairs
        # might be more useful to imagine the first 100 as N, and second 100 as M
        # target_classes: [100, target_len]
        
        class_loss = class_criteria(class_pred.reshape(-1, num_class), target_classes.reshape(-1))
        class_loss = class_loss.reshape(num_bb, target_len)
        # class_loss: [100, target_len]
    
        # BBOX L1 LOSS
        bbox_loss = 1 * torch.cdist(box_pred, target_bbox, p=1.0) 
        # bbox_loss = bbox_loss + 10 * torch.square(torch.cdist(box_pred, target_bbox, p=2.0))
        # bbox_loss: [100, target_len]
    
        # BBOX G-IOU LOSS
        giou_loss = 1 - generalized_box_iou(
            box_pred, 
            target_bbox
        )
        # giou_loss: [100, target_len]
        # assert torch.min(giou_loss) >= -20.0
    
        # Total Cost
        total_cost = (weights[0] * class_loss + weights[1] * bbox_loss + weights[2] * giou_loss)
        # Positive because linear_sum_assignment minimizers total weight by default
        # Compute the bipartite matching problem and final loss
        row_idx, col_idx = linear_sum_assignment(total_cost.detach().cpu().numpy())
        total_cost = (weights[0] * class_loss + weights[1] * bbox_loss + weights[2] * giou_loss) / (weights[0] + weights[1] + weights[2])
        final_loss += (torch.mean(total_cost[row_idx, col_idx]))
        total_class_loss += torch.mean(weights[0] * class_loss[row_idx, col_idx]) / (weights[0] + weights[1])
        total_bbox_loss += torch.mean(weights[1] * bbox_loss[row_idx, col_idx]) / (weights[0] + weights[1])

        # Force the non-matched classes to be "no labels" class
        mask = torch.ones(num_bb)
        mask[row_idx] = mask[row_idx] - 1
        mask = mask > 0.5 # Select the indicies not matched
        if torch.sum(mask) > 0:
            bad_class = class_pred[mask, :, :].reshape(-1, num_class)
            # reshape to allow loss computation
            no_label_target = torch.ones(bad_class.shape[0], device=device, dtype=torch.long) * no_labels_target_id
            # construct the target that forces everything to "no label" class
            final_loss += class_criteria(bad_class, no_label_target).mean()
            total_no_label_loss += class_criteria(bad_class, no_label_target).mean()

        # BBox sizes loss (hopefully help with smaller objects)
        target_bbox_size = target_bbox[:, [2, 3]] - target_bbox[:, [0, 1]]
        size_loss = bbox_size_criteria(bbox_size[row_idx].reshape(-1), target_bbox_size[col_idx].reshape(-1)) * 10
        final_loss += size_loss
        total_size_loss += size_loss
        
        # Compute IoU
        iou_matrix = box_iou(box_pred, target_bbox)
        total_iou += torch.mean(iou_matrix[row_idx, col_idx])

        # In addition, to encourage matching, we also add all those with IoU > 0.6 or IoU < 0.3
        iou_mask = (iou_matrix > 0.6) | (iou_matrix < 0.3)
        final_loss += torch.mean(iou_matrix[iou_mask]) * 0.5
    exact_losses = torch.tensor([total_class_loss, total_bbox_loss, total_no_label_loss, total_mask_loss, total_size_loss]) / batch_size
    return final_loss, total_iou / batch_size, exact_losses