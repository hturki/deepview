from typing import Dict

import torch

import torch.nn.functional as F
import lpips as plips


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def lpips(image_pred, image_gt) -> Dict[str, float]:
    gt = image_gt.permute([2, 0, 1]).contiguous()
    pred = image_pred.permute([2, 0, 1]).contiguous()

    lpips_vgg = plips.LPIPS(net='vgg').eval().to(image_pred.device)
    lpips_vgg_i = lpips_vgg(gt, pred, normalize=True)

    lpips_alex = plips.LPIPS(net='alex').eval().to(image_pred.device)
    lpips_alex_i = lpips_alex(gt, pred, normalize=True)

    lpips_squeeze = plips.LPIPS(net='squeeze').eval().to(image_pred.device)
    lpips_squeeze_i = lpips_squeeze(gt, pred, normalize=True)

    return {'vgg': lpips_vgg_i.item(), 'alex': lpips_alex_i.item(), 'squeeze': lpips_squeeze_i.item()}


# From plenoctree repo
def ssim(
        img0,
        img1,
        max_val,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        return_map=False,
):
    """Computes SSIM from two images.
    This function was modeled after tf.image.ssim, and should produce comparable
    output.
    Args:
      img0: torch.tensor. An image of size [..., width, height, num_channels].
      img1: torch.tensor. An image of size [..., width, height, num_channels].
      max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
      filter_size: int >= 1. Window size.
      filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
      k1: float > 0. One of the SSIM dampening parameters.
      k2: float > 0. One of the SSIM dampening parameters.
      return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned
    Returns:
      Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    device = img0.device
    ori_shape = img0.size()
    width, height, num_channels = ori_shape[-3:]
    img0 = img0.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = torch.mean(ssim_map.reshape([-1, num_channels * width * height]), dim=-1)
    return ssim_map if return_map else ssim


def get_metrics(im: torch.Tensor, im_gt: torch.Tensor):
    eval_rgbs = im[:, im.shape[1] // 2:].contiguous()
    eval_result_rgbs = im_gt.view(*im.shape)[:, im_gt.shape[1] // 2:].reshape(-1, 3)

    val_psnr = psnr(eval_result_rgbs, eval_rgbs.view(-1, 3))
    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)
    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

    return val_psnr, val_ssim, val_lpips_metrics