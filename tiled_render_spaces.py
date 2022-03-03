# Tiled render for spaces dataset (try 2k too!)

import os
import pathlib
from collections import defaultdict

import torch

import misc
import model_deepview
import utils_dset
from dset_realestate.dronenerf import DroneNeRF
from metrics import get_metrics


def crop_model_input(inp, x0, y0, tile_w, tile_h):
    x = utils_dset.unwrap_input(inp)
    _, base_h, base_w, _ = x['in_img'].shape

    ref_pixel_x = x0 + tile_w // 2
    ref_pixel_y = y0 + tile_h // 2

    x['in_intrin_base'] = x['in_intrin']
    x['ref_intrin_base'] = x['ref_intrin']
    x['in_img_base'] = x['in_img']
    ref_cam_z = x['mpi_planes'][x['mpi_planes'].shape[0] // 2]

    ref_pos = (ref_pixel_x, ref_pixel_y, ref_cam_z)

    res = utils_dset.crop_scale_things(x, ref_pos, tile_w, tile_h, base_w, base_h)
    return utils_dset.wrap_input(res)


def create_html_viewer(outpath, model, device, x, num_planes, tile_w, tile_h):
    """Infer a batch, and create an HTML viewer from the template"""

    print(outpath)
    # breakpoint()
    _, _, base_h, base_w, _ = x['in_img'].shape

    # w_factor = 4
    # attempts = 0
    # while tile_w % w_factor != 0:
    #     if attempts > 10:
    #         raise Exception(tile_w, w_factor)
    #     w_factor += 1
    #     attempts += 1
    #
    # h_factor = 4
    # attempts = 0
    # while tile_h % h_factor != 0:
    #     if attempts > 10:
    #         raise Exception(tile_h, h_factor)
    #     h_factor += 1
    #     attempts += 1

    margin_w = tile_w // 4
    margin_h = tile_h // 4

    subtile_w = tile_w - 2 * margin_w
    subtile_h = tile_h - 2 * margin_h
    iters_w = (base_w - 2 * margin_w) // subtile_w
    iters_h = (base_h - 2 * margin_h) // subtile_h

    yi_rgba_layers = []
    for yi in range(0, iters_h):
        y0 = yi * subtile_h

        min_y = 0 if yi == 0 else margin_h
        max_y = tile_h if yi == (iters_h - 1) else (tile_h - margin_h)
        xi_rgba_layers = []
        for xi in range(0, iters_w):
            x0 = xi * subtile_w
            x_ij = crop_model_input(x, x0, y0, tile_w, tile_h)
            x_ij = misc.to_device(x_ij, device)  # One batch

            with torch.no_grad():
                out = model(x_ij)
            out = torch.sigmoid(out)
            rgba_layers_x0 = out.permute(0, 3, 4, 1, 2)  # result: [batch, height, width, layers, colours]

            min_x = 0 if xi == 0 else margin_w
            max_x = tile_w if xi == (iters_w - 1) else (tile_w - margin_w)

            xi_rgba_layers.append(rgba_layers_x0[:, min_y:max_y, min_x:max_x])

        rgba_layers_y0 = torch.cat(xi_rgba_layers, dim=2)
        yi_rgba_layers.append(rgba_layers_y0)

    rgba_layers = torch.cat(yi_rgba_layers, dim=1)

    rgb = torch.zeros(rgba_layers.shape[1], rgba_layers.shape[2], 3, device=rgba_layers.device)
    remaining = torch.ones(rgba_layers.shape[1], rgba_layers.shape[2], 1, device=rgba_layers.device)
    for rgba_layer in rgba_layers.squeeze().permute(2, 0, 1, 3):
        weight = (rgba_layer[:, :, 3:] * remaining)
        weighted = rgba_layer[:, :, :3] * weight
        rgb += weighted
        remaining -= weight

    outpath.mkdir(parents=True)
    misc.save_image(rgb, outpath / 'result.png')
    target = x['tgt_img'].squeeze()
    misc.save_image(target, outpath / 'target.png')

    # By now we have RGBA MPI in the [0, 1] range
    # Export them to the HTML
    p_viewer = pathlib.Path(outpath)
    if not p_viewer.exists():
        p_viewer.mkdir()
    # print(rgba_layers.shape, rgba_layers.dtype)
    for i in range(num_planes):
        layer = i
        file_path = '{}/mpi{}.png'.format(outpath, ("0" + str(layer))[-2:])
        img = rgba_layers[0, :, :, layer, :]
        misc.save_image(img, file_path)

    image_srcs = [misc.get_base64_encoded_image('{}/mpi{}.png'.format(outpath, ("0" + str(i))[-2:])) for i in
                  range(num_planes)]

    with open("./deepview-mpi-viewer-template.html", "r") as template_file:
        template_str = template_file.read()

    MPI_SOURCES_DATA = ",".join(['\"' + img_src + '\"' for img_src in image_srcs])
    template_str = template_str.replace("const mpiSources = MPI_SOURCES_DATA;",
                                        "const mpiSources = [{}];".format(MPI_SOURCES_DATA))

    with open("{}/deepview-mpi-viewer.html".format(outpath), "w") as output_file:
        output_file.write(template_str)

    start_h = (target.shape[0] - rgb.shape[0]) // 2
    start_w = (target.shape[1] - rgb.shape[1]) // 2

    val_psnr, val_ssim, val_lpips_metrics = get_metrics(rgb.cpu(), target.cpu()[start_h:start_h + rgb.shape[0],
                                                                   start_w:start_w + rgb.shape[1]])

    with (outpath / 'metrics.txt').open('w') as f:
        f.write('PSNR: {}\n'.format(val_psnr))
        f.write('SSIM: {}\n'.format(val_ssim))
        for network in val_lpips_metrics:
            f.write('LPIPS/{}: {}\n'.format(network, val_lpips_metrics[network]))

    return val_psnr, val_ssim, val_lpips_metrics


##### LLFF
def stack_picker(inn, picks):
    return torch.stack([inn[i] for i in picks], dim=0)


########################################################################################################################
def main():
    device_type = os.environ.get('DEVICE', 'cuda')
    device = torch.device(device_type)

    num_planes = 10
    patch_factor = int(os.environ['PATCH_FACTOR'])
    dset = DroneNeRF(os.environ['DSET_PATH'], True, None, None, resize_factor=int(os.environ['RESIZE_FACTOR']))

    # load model
    model = model_deepview.DeepViewLargeModel()
    filepath = pathlib.Path(os.environ['MODEL_PATH'])
    model.load_state_dict(torch.load(filepath, map_location='cpu')['model_state_dict'])

    model = model.to(device=device)

    output = pathlib.Path(os.environ['OUT_DIR'])
    output.mkdir()

    count = 0
    val_metrics = defaultdict(float)
    dloader = torch.utils.data.DataLoader(dset, batch_size=1)
    for i, x in enumerate(dloader):
        w_patch_factor = patch_factor
        # attempts = 0
        # while x['tgt_img'].shape[3] % w_patch_factor != 0:
        #     if attempts > 10:
        #         raise Exception(x['tgt_img'].shape[3], w_patch_factor)
        #     w_patch_factor += 1
        #     attempts += 1

        h_patch_factor = patch_factor
        # attempts = 0
        # while x['tgt_img'].shape[2] % h_patch_factor != 0:
        #     if attempts > 10:
        #         raise Exception(x['tgt_img'].shape[2], h_patch_factor)
        #     h_patch_factor += 1
        #     attempts += 1

        print(x['tgt_img'].shape, x['tgt_img'].shape[3] // w_patch_factor, x['tgt_img'].shape[2] // h_patch_factor)

        val_psnr, val_ssim, val_lpips_metrics = create_html_viewer(output / str(i), model, device, x, num_planes,
                                                                   x['tgt_img'].shape[3] // w_patch_factor,
                                                                   x['tgt_img'].shape[2] // h_patch_factor)
        val_metrics['val/psnr'] += val_psnr
        val_metrics['val/ssim'] += val_ssim
        for network in val_lpips_metrics:
            val_metrics['val/lpips/{}'.format(network)] += val_lpips_metrics[network]

        count += 1

    with (output / 'metrics.txt').open('w') as f:
        for key in val_metrics:
            avg_val = val_metrics[key] / count
            message = 'Average {}: {}'.format(key, avg_val)
            print(message)
            f.write('{}\n'.format(message))


########################################################################################################################
if __name__ == '__main__':
    main()
