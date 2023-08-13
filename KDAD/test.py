from argparse import ArgumentParser
from utils.utils import get_config
from dataloader import load_data, load_localization_data
from test_functions import detection_test, localization_test
from models.network import get_networks
import matplotlib
import matplotlib.pyplot as plt
import os
import einops
import numpy as np

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")


def vis_maps(imgs, loc_maps, save_dir, filenames):
    for i in range(loc_maps.shape[0]):
        mp = loc_maps[i]
        im = imgs[i].numpy()
        im = einops.rearrange(im, 'c h w -> h w c')
        print(im.shape, mp.shape)
        img = np.concatenate([im, mp], axis=1)
        print(img.shape)
        # img = einops.rearrange([im, m], 't h w c -> h (t w) c')
        fig = plt.figure()
        plt.imshow(img)
        f = filenames[i].split('/')[-1]
        fpath = os.path.join(save_dir, f'{f}.png')
        print(fpath)
        plt.savefig(fpath, dpi=300)

def main():
    args = parser.parse_args()
    config = get_config(args.config)
    load_checkpoint_path = config['load_checkpoint_path']
    vgg, model = get_networks(config, load_checkpoint=True, load_checkpoint_path=load_checkpoint_path)

    # Localization test
    if 'emdc_localization_test' in config and config['emdc_localization_test']:
        _, test_dataloader = load_data(config)
        print(len(test_dataloader.dataset))
        imgs, loc_maps = localization_test(model=model, vgg=vgg, test_dataloader=test_dataloader, ground_truth=None,
                                     config=config, return_maps=True)
        save_dir = config['map_save_dir']
        filenames = test_dataloader.dataset.x
        vis_maps(imgs, loc_maps, save_dir, filenames)
    else:
        if config['localization_test']:
            test_dataloader, ground_truth = load_localization_data(config)
            roc_auc = localization_test(model=model, vgg=vgg, test_dataloader=test_dataloader, ground_truth=ground_truth,
                                        config=config)

        # Detection test
        else:
            _, test_dataloader = load_data(config)
            roc_auc = detection_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)
        last_checkpoint = config['last_checkpoint']
        print("RocAUC after {} epoch:".format(last_checkpoint), roc_auc)


if __name__ == '__main__':
    main()
