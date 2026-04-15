import torch
import numpy as np
from tqdm import tqdm
import argparse
import yaml

from pyiqa.data import build_dataset, build_dataloader
from pyiqa.models import build_model
from pyiqa.metrics import calculate_srcc, calculate_plcc, calculate_krcc


def main(args):
    # -------- Dataset --------
    with open('/kaggle/working/ipa-topiq/pyiqa/default_dataset_configs.yml') as f:
        options = yaml.safe_load(f)

    dataset_opt = options['tid2013']
    dataset_opt['type'] = 'GeneralNRDataset'
    dataset_opt['dataroot_target'] = args.dataroot_target
    dataset_opt['meta_info_file'] = args.meta_info
    dataset_opt['phase'] = 'test'
    dataset_opt['batch_size_per_gpu'] = 1
    dataset_opt['num_worker_per_gpu'] = 2

    dataset = build_dataset(dataset_opt)
    dataloader = build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False)

    print(f"Dataset size: {len(dataset)}")

    # -------- Model --------
    opt = {
        'model_type': 'GeneralIQAModel',
        'is_train': False,
        'train': {},

        'network': {
            'type': 'CFANet',
            'use_ref': False,
            'pretrained': False,
            'num_crop': 1,
            'num_attn_layers': 1,
            'crop_size': [384, 384],
            'semantic_model_name': 'resnet50',
            'block_pool': 'weighted_avg',
        },

        'path': {
            'pretrain_network_g': args.checkpoint,
            'strict_load_g': True,
        },

        'val': {
            'metrics': {
                'srcc': {'type': 'calculate_srcc'},
                'plcc': {'type': 'calculate_plcc'},
                'krcc': {'type': 'calculate_krcc'},
            }
        },

        'num_gpu': 1,
        'dist': False,
        'rank': 0,
    }

    model = build_model(opt)
    model.net.cuda()
    model.net.eval()

    preds = []
    gts = []

    # -------- Inference --------
    with torch.no_grad():
        for data in tqdm(dataloader):
            model.feed_data(data)
            model.test()

            pred = model.output_score
            gt = model.gt_mos

            # FORCE SCALAR (critical fix)
            if isinstance(pred, torch.Tensor):
                pred = pred.view(-1)[0].item()
            if isinstance(gt, torch.Tensor):
                gt = gt.view(-1)[0].item()

            preds.append(pred)
            gts.append(gt)

    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    # 🔥 ALIGN LENGTHS (critical fix)
    min_len = min(len(preds), len(gts))
    preds = preds[:min_len]
    gts = gts[:min_len]

    print(f"Final length → preds: {len(preds)}, gts: {len(gts)}")

    # -------- Metrics --------
    srcc = calculate_srcc([preds, gts], {})
    plcc = calculate_plcc([preds, gts], {})
    krcc = calculate_krcc([preds, gts], {})

    print("\n========== TID2013 RESULTS ==========")
    print(f"SRCC: {srcc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"KRCC: {krcc:.4f}")
    print("=====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataroot_target', type=str)
    parser.add_argument('--meta_info', type=str)
    args = parser.parse_args()

    main(args)
