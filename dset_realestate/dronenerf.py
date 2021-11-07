import pathlib
from typing import Optional

import numpy as np
import PIL.Image
import torch
import torch.utils.data
import torch.nn.functional
import random

import utils_dset


class MetadataItem:
    def __init__(self, image_path: pathlib.Path, c2w: torch.Tensor, intrinsics: torch.Tensor):
        self.image_path = image_path
        self.c2w = c2w
        self.intrinsics = intrinsics


########################################################################################################################
class DroneNeRF(torch.utils.data.Dataset):

    def __init__(self, dataset_path, is_valid=False, im_w=200, im_h=200, num_planes=10, num_views=3, no_crop=False, resize_factor=None, patch_factor=None):
        print(f'DroneNeRF: dataset_path={dataset_path}, is_valid={is_valid}')
        self.is_valid = is_valid
        self.dataset_path = pathlib.Path(dataset_path)
        self.im_w = im_w
        self.im_h = im_h
        self.num_planes = num_planes
        self.num_views = num_views
        self.no_crop = no_crop
        self.resize_factor = resize_factor
        self.patch_factor = patch_factor

        self.metadata_items = []
        metadatas = list((self.dataset_path / 'train' / 'metadata').iterdir()) + list(
            (self.dataset_path / 'val' / 'metadata').iterdir())
        for metadata_path in sorted(metadatas, key=lambda x: x.name):
            metadata_item = torch.load(metadata_path)
            image_path = metadata_path.parent.parent / 'rgbs' / '{}.jpg'.format(metadata_path.stem)
            if not image_path.exists():
                image_path = metadata_path.parent.parent / 'rgbs' / '{}.png'.format(metadata_path.stem)
                assert image_path.exists()
            pose = torch.eye(4)
            pose[:3] = metadata_item['c2ws']['gt']
            self.metadata_items.append(MetadataItem(image_path, pose, metadata_item['focal']))

        self.used_indices = []
        if is_valid:
            used_range = []
            for i in range(len(self.metadata_items)):
                if '/val/' in str(self.metadata_items[i].image_path):
                    print('Val item: {}, index: {}'.format(self.metadata_items[i].image_path, i))
                    used_range.append(i)
        else:
            used_range = np.arange(self.num_views, len(self.metadata_items) - self.num_views)

        for i in used_range:
            self.used_indices.append(i)

    def __len__(self):
        return len(self.used_indices)

    def __getitem__(self, i):
        index = self.used_indices[i]
        if self.is_valid:
            indices = [index - 1, index + 1, index]
        else:
            indices = np.random.choice(np.arange(-5, 5), self.num_views)

        selected_metadata = []
        for idx in indices:
            intrinsics0 = self.metadata_items[idx].intrinsics
            img = PIL.Image.open(self.metadata_items[idx].image_path).convert('RGB')
            # create intrinsic camera matrix
            # [fx fy cx cy]
            intrinsics = utils_dset.make_intrinsics_matrix(
                intrinsics0[0], intrinsics0[1],
                intrinsics0[2], intrinsics0[3]
            )

            if len(selected_metadata) == 0 and self.resize_factor is not None:
                img, intrinsics = utils_dset.resize_totensor_intrinsics(
                    img, intrinsics, img.size[0] // self.resize_factor, img.size[1] // self.resize_factor)
            elif len(selected_metadata) > 0 and (img.size[1] != selected_metadata[0]['image'].shape[0] or img.size[0] !=
                                               selected_metadata[0]['image'].shape[1]):
                img, intrinsics = utils_dset.resize_totensor_intrinsics(
                    img, intrinsics, selected_metadata[0]['image'].shape[1], selected_metadata[0]['image'].shape[0])
            else:
                img = torch.FloatTensor(np.array(img)) / 255.

            # Metadata
            pose = self.metadata_items[idx].c2w.clone()
            pose[:3, 3] -= self.metadata_items[indices[0]].c2w[:3, 3]
            selected_metadata.append({
                # 'timestamp': scene['timestamps'][idx],
                'intrinsics': intrinsics,
                'pose': torch.inverse(pose),
                'image': img,
            })

        ref_img = selected_metadata[0]  # this is the camera view pose we'll use to create the MPI
        src_imgs = selected_metadata[:-1]  # all the camera views that are input to the NN
        tgt_img = selected_metadata[-1]  # this is the dependent variable what the output should be

        # the list of plane depths we're going to consider for the MPI
        psv_planes = torch.FloatTensor(utils_dset.inv_depths(1, 100, self.num_planes))
        intrinsics_final = torch.stack([m['intrinsics'] for m in src_imgs])

        res = {
            'in_img': torch.stack([m['image'] for m in src_imgs]),
            'in_intrin': intrinsics_final,
            'in_cfw': torch.stack([m['pose'] for m in src_imgs]),
            'tgt_img': tgt_img['image'].unsqueeze(0),
            'tgt_intrin': tgt_img['intrinsics'].unsqueeze(0),
            'tgt_cfw': tgt_img['pose'].unsqueeze(0),
            'mpi_planes': psv_planes,
            'ref_intrin': ref_img['intrinsics'],
            'ref_cfw': ref_img['pose'],
            'ref_wfc': torch.inverse(ref_img['pose']),
        }

        res['in_intrin_base'] = res['in_intrin']
        res['ref_intrin_base'] = res['ref_intrin']
        res['in_img_base'] = res['in_img']

        if not self.no_crop:
            ref_cam_z = res['mpi_planes'][res['mpi_planes'].shape[0] // 2]
            width = res['in_img'][0].shape[1]
            height = res['in_img'][0].shape[0]

            w = tgt_img['image'].shape[1] #if self.patch_factor is None else tgt_img['image'].shape[1] // self.patch_factor
            h = tgt_img['image'].shape[0]  #if self.patch_factor is None else tgt_img['image'].shape[0] // self.patch_factor
            if self.is_valid:
                ref_pixel_x = (width / 2)
                ref_pixel_y = (height / 2)
            else:
                ref_pixel_x = (width / 2) + random.uniform(-0.25, 0.25) * (width - w)
                ref_pixel_y = (height / 2) + random.uniform(-0.25, 0.25) * (height - h)

            ref_pos = (ref_pixel_x, ref_pixel_y, ref_cam_z)

            res = utils_dset.crop_scale_things(res, ref_pos, w, h, width, height)

        return res

########################################################################################################################
