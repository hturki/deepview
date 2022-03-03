import os
import pathlib
import time
from collections import defaultdict

import torch
import torch.nn.functional
import torch.utils.data
import tqdm

import misc
import model_deepview
import test_engine
import utils_render
import vgg
########################################################################################################################
from dset_realestate.dronenerf import DroneNeRF
from metrics import get_metrics
from tiled_render_spaces import crop_model_input


class TrainerDeepview:
    """
    Trainer class for "mini_deep_view_no_lgd"
    """

    def __init__(self, dset_dir, dset_options={}, device=torch.device('cuda'), lr=1.e-3, batch_size=1,
                 im_w=200, im_h=200, borders=(0, 0)):
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.num_planes = 10
        self.num_workers = 0
        self.borders = borders
        # Model
        self.model = model_deepview.DeepViewLargeModel().to(device=device)
        # VGG loss
        self.vgg_loss = vgg.VGGPerceptualLoss1(resize=False, device=device)

        self.iteration_count = 0

        # Dataset+loaders
        print(
            f'TrainerDeepview: dset_dir={dset_dir}, dset_options={dset_options}, device={device}')

        resize_factor = int(os.environ['RESIZE_FACTOR']) if 'RESIZE_FACTOR' in os.environ else None
        self.dset_train = DroneNeRF(dset_dir, False, im_w, im_h, resize_factor=resize_factor, **dset_options)
        self.dset_val = DroneNeRF(dset_dir, True, im_w, im_h, resize_factor=resize_factor, **dset_options)

        # if dset_name == 'spaces:1deterministic':
        #     self.dset_train = dset_spaces.dset1.DsetSpaces1(dset_dir, False, im_w=im_w, im_h=im_h, **dset_options)
        #     self.dset_val = dset_spaces.dset1.DsetSpaces1(dset_dir, True, im_w=im_w, im_h=im_h, **dset_options)
        # elif dset_name == 're:1random':
        #     self.dset_train = dset_realestate.dset1.DsetRealEstate1(dset_dir, False, im_w=im_w, im_h=im_h,
        #                                                             **dset_options)
        #     self.dset_val = dset_realestate.dset1.DsetRealEstate1(dset_dir, True, im_w=im_w, im_h=im_h, **dset_options)
        # elif dset_name =="blender":
        #     self.dset_train = dset_blender.dset1.DsetBlender(dset_dir, False, im_w=im_w, im_h=im_h,
        #                                                             **dset_options)
        #     self.dset_val = dset_blender.dset1.DsetBlender(dset_dir, True, im_w=im_w, im_h=im_h, **dset_options)
        # else:
        #     raise ValueError(f'Wrong dset_name={dset_name} !')

        print(f'Datasets : train: {len(self.dset_train)} {len},  val: {len(self.dset_val)}')
        self.loader_train = torch.utils.data.DataLoader(self.dset_train, batch_size=self.batch_size,
                                                        num_workers=self.num_workers, shuffle=True)
        self.loader_val = torch.utils.data.DataLoader(self.dset_val, batch_size=self.batch_size,
                                                      num_workers=self.num_workers)
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def process_one(self, out, x):
        """Process and render one network output+target"""
        out = torch.sigmoid(out)
        rgba_layers = out.permute(0, 3, 4, 1, 2)

        # print('SHAPE=', rgba_layers.shape)
        # print(f'MIN={rgba_layers.min().item()}, MAX={rgba_layers.max().item()}')
        n_targets = x['tgt_img'].shape[1]

        # t1 = time.time()
        batch_size = rgba_layers.shape[0]
        # i_batch = 0  # Replace later with a loop over batch
        out_images_batch = []
        # Can we fully batch the second version, I wonder?
        for i_batch in range(batch_size):
            if False:
                # Version with a loop over targets
                outs = []
                for i_target in range(n_targets):
                    rel_pose = torch.matmul(x['tgt_cfw'][i_batch, i_target, :, :], x['ref_wfc'][i_batch]).unsqueeze(0)
                    intrin_tgt = x['tgt_intrin'][i_batch, i_target, :, :].unsqueeze(0)
                    intrin_ref = x['ref_intrin'][i_batch].unsqueeze(0)
                    out_image = utils_render.mpi_render_view_torch(rgba_layers, rel_pose, x['mpi_planes'][i_batch],
                                                                   intrin_tgt, intrin_ref)
                    outs.append(out_image)
                out_images = torch.cat(outs, 0)
            else:
                # Version batched over targets, but we have to repeat a relatively large tensor rgba, which one is better?
                # Results are very close (but not to machine precision !)
                rel_pose = torch.matmul(x['tgt_cfw'][i_batch], x['ref_wfc'][i_batch])
                intrin_tgt = x['tgt_intrin'][i_batch]
                intrin_ref = x['ref_intrin'][i_batch].unsqueeze(0).repeat(n_targets, 1, 1)
                rgba = rgba_layers[i_batch].unsqueeze(0).repeat(n_targets, 1, 1, 1, 1)

                out_images = utils_render.mpi_render_view_torch(rgba, rel_pose, x['mpi_planes'][i_batch], intrin_tgt,
                                                                intrin_ref)

            # t2 = time.time()
            # print('TIME RENDER', t2-t1)
            out_images_batch.append(out_images)
        out_images_batch = torch.cat(out_images_batch, dim=0)
        targets = x['tgt_img']
        targets = targets.reshape(batch_size * n_targets, *targets.shape[2:])
        return out_images_batch, targets

    def loss(self, out, x):
        """Calculate VGG loss"""
        output_image, target = self.process_one(out, x)
        output_image = misc.my_crop(output_image, self.borders)
        target = misc.my_crop(target, self.borders)
        loss = self.vgg_loss(output_image, target)
        return loss

    @staticmethod
    def composite_image(model, device, x, tile_w, tile_h):
        """Infer a batch, and create an HTML viewer from the template"""

        # breakpoint()
        _, _, base_h, base_w, _ = x['in_img'].shape
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

        val_psnr, val_ssim, val_lpips_metrics = get_metrics(rgb.cpu(), x['tgt_img'].squeeze().cpu())
        return val_psnr, val_ssim, val_lpips_metrics

    def val(self):
        """Validate, 1 run"""
        self.model.eval()
        loss_sum = 0
        with torch.no_grad():
            val_metrics = defaultdict(float)

            for batch in self.loader_val:
                val_psnr, val_ssim, val_lpips_metrics = self.composite_image(self.model, 'cuda', batch,
                                                                             self.dset_val.im_w, self.dset_val.im_h)
                val_metrics['val/psnr'] += val_psnr
                val_metrics['val/ssim'] += val_ssim
                for network in val_lpips_metrics:
                    val_metrics['val/lpips/{}'.format(network)] += val_lpips_metrics[network]
                x = misc.to_device(batch, self.device)
                out = self.model(x)
                loss = self.loss(out, x)
                loss_sum += loss.item()

        for key in val_metrics:
            val_metrics[key] /= len(self.loader_val)
        return loss_sum / len(self.loader_val), val_metrics

    def train(self):
        """Train 1 epoch"""
        self.model.train()
        loss_sum = 0
        for batch in tqdm.tqdm(self.loader_train):
            x = misc.to_device(batch, self.device)
            # t1 = time.time()
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.loss(out, x)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
            # t2 = time.time()
            # print('TIME TRAIN RUN', t2-t1)

            self.iteration_count += 1

        return loss_sum / len(self.loader_train)

    def train_loop(self, n_epoch=10):
        """Train the model"""
        for i_epoch in range(n_epoch):
            t1 = time.time()
            loss_train = self.train()
            loss_val, val_metrics = self.val()
            t2 = time.time()
            print(
                f'\nEpoch {i_epoch}/{n_epoch} : iterations = {self.iteration_count} : loss_train = {loss_train}, loss_val = {loss_val}, val_metrics = {val_metrics}, time = {t2 - t1:6.2f}s')
            self.save_model()

        # Save the trained model
        self.save_model()

    def save_model(self):
        dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iteration': self.iteration_count,
        }
        torch.save(dict, pathlib.Path(os.environ['SAVE_PATH']) / '{}.pt'.format(self.iteration_count))

    def load_model(self, path: pathlib.Path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iteration_count = checkpoint['iteration']

    def create_html_viewer(self, scene_idx=0):
        """Infer a batch, and create an HTML viewer from the template"""
        self.model.eval()

        # Choose your scene_idx from the val set !
        # Dataloaders like this adds batch dimension to a chosen element !
        scene_loader = torch.utils.data.DataLoader(self.dset_val, batch_sampler=[[scene_idx]])
        x = misc.to_device(next(iter(scene_loader)), self.device)  # One batch

        with torch.no_grad():
            out = self.model(x)
        out = torch.sigmoid(out)
        rgba_layers = out.permute(0, 3, 4, 1, 2)

        # By now we have RGBA MPI in the [0, 1] range
        # Export them to the HTML
        p_viewer = pathlib.Path('generated-html')
        if not p_viewer.exists():
            p_viewer.mkdir()
        # print(rgba_layers.shape, rgba_layers.dtype)
        for i in range(self.num_planes):
            layer = i
            file_path = 'generated-html/mpi{}.png'.format(("0" + str(layer))[-2:])
            img = rgba_layers[0, :, :, layer, :]
            misc.save_image(img, file_path)

        image_srcs = [misc.get_base64_encoded_image('./generated-html/mpi{}.png'.format(("0" + str(i))[-2:])) for i in
                      range(self.num_planes)]

        with open("./deepview-mpi-viewer-template.html", "r") as template_file:
            template_str = template_file.read()

        MPI_SOURCES_DATA = ",".join(['\"' + img_src + '\"' for img_src in image_srcs])
        template_str = template_str.replace("const mpiSources = MPI_SOURCES_DATA;",
                                            "const mpiSources = [{}];".format(MPI_SOURCES_DATA))

        with open("./generated-html/deepview-mpi-viewer.html", "w") as output_file:
            output_file.write(template_str)

    def test(self):
        """Test on the val set with the test engine (SSIM)"""
        self.model.eval()
        engine = test_engine.TestEngine()
        with torch.no_grad():
            for batch in tqdm.tqdm(self.loader_val):
                x = misc.to_device(batch, self.device)
                out = self.model(x)
                outs, targets = self.process_one(out, x)
                engine.run_batch(x, outs, targets, self.borders)
        engine.print_stats()

########################################################################################################################
