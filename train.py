# Train DeepView!

import os
from pathlib import Path

import torch
import torch.utils.data

import trainer_deepview


########################################################################################################################
def main():
    print('train.py')

    do_training = os.environ.get('TRAIN', 'True') == 'True'

    #dset_name = os.environ.get('DSET_NAME', 'spaces:1deterministic')
    #dset_options = dict(tiny=True, layout='large_4_9')
    # dset_name = 're:1random'
    dset_options = {}

    # dset_path_spaces = os.environ.get('SPACES_PATH', '/big/workspace/spaces_dataset/')
    # dset_path_re = os.environ.get('RE_PATH', '/big/workspace/real-estate-10k-run0')
    # dset_path_blender = os.environ.get('BLENDER_PATH', '/big/workspace/negatives-wupi/felix-london-july-74/blender0/')
    #
    # if dset_name.startswith('spaces'):
    #     dset_path = dset_path_spaces
    # elif dset_name == 're:1random':
    #     dset_path = dset_path_re
    # elif dset_name =="blender":
    #     dset_path = dset_path_blender
    
    device_type = os.environ.get('DEVICE', 'cuda')
    device = torch.device(device_type)
    batch_size = 1
    lr = 1e-4
    epochs = 100
    im_w, im_h = int(os.environ['IM_W']), int(os.environ['IM_H'])

    trainer = trainer_deepview.TrainerDeepview(dset_dir=os.environ['DSET_PATH'],
                                               dset_options=dset_options,
                                               device=device,
                                               lr=lr,
                                               batch_size=batch_size,
                                               im_w=im_w,
                                               im_h=im_h,
                                               borders=(im_w // 5, im_h // 5))

    save_path = Path(os.environ['SAVE_PATH'])
    print('Save path', save_path)
    if save_path.exists() and len(list(save_path.iterdir())) > 0:
        latest_checkpoint = sorted(list(save_path.iterdir()), key=lambda x: int(x.stem))[-1]
        print('Loading from {}'.format(latest_checkpoint))

        # try loading the model
        trainer.load_model(latest_checkpoint)
    else:
        save_path.mkdir(parents=True, exist_ok=True)

    if do_training:   # Train or load ?
        trainer.train_loop(n_epoch=epochs)

    #trainer.demo_draw()
    trainer.create_html_viewer()

    # print('VAL LOSS=', trainer.val())


########################################################################################################################
if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    main()
