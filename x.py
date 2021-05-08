

# UnetAdaptiveBins
return centers.view(N, n_bins, 1), pred

# loss 
class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bin_center, ground_truth):
        # (n, p, 1) bins
        # n, c, h, w = ground_truth.shape

        gt_points = ground_truth.flatten(1)  # n, hwc
        mask = gt_points.ge(1e-3)  # only valid ground truth points
        gt_points = [p[m] for p, m in zip(gt_points, mask)]
        gt_length = torch.Tensor([len(t) for t in gt_points]).long().to(ground_truth.device)
        gt_points = pad_sequence(gt_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=bin_center, y=gt_points, y_lengths=gt_length)
        return loss