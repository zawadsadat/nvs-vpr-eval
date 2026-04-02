#   =====================================================================
#   Copyright (C) 2023  Stefan Schubert, stefan.schubert@etit.tu-chemnitz.de
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#   =====================================================================

import argparse
import configparser
import os

import numpy as np

from evaluation.metrics import createPR, recallAt100precision, recallAtK
from evaluation import show_correct_and_wrong_matches
from matching import matching
from datasets.load_dataset import (
    GardensPointDataset, StLuciaDataset, SFUDataset, CorridorDataset, ESSEX3IN1Dataset
)

# Headless-safe matplotlib backend
if os.environ.get('DISPLAY', '') == '':
    os.environ.setdefault('MPLBACKEND', 'Agg')
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.'
    )
    parser.add_argument(
        '--descriptor', type=str, default='HDC-DELF',
        choices=['HDC-DELF', 'AlexNet', 'NetVLAD', 'PatchNetVLAD', 'CosPlace', 'EigenPlaces', 'SAD'],
        help='Select descriptor (default: HDC-DELF)'
    )
    parser.add_argument(
        '--dataset', type=str, default='GardensPoint',
        choices=['GardensPoint', 'StLucia', 'SFU', 'Corridor', 'ESSEX3IN1'],
        help='Select dataset (default: GardensPoint)'
    )
    args = parser.parse_args()

    out_dir = "output_images"
    os.makedirs(out_dir, exist_ok=True)

    print(f'========== Start VPR with {args.descriptor} descriptor on dataset {args.dataset}')

    # load dataset
    print('===== Load dataset')
    if args.dataset == 'GardensPoint':
        dataset = GardensPointDataset()
    elif args.dataset == 'StLucia':
        dataset = StLuciaDataset()
    elif args.dataset == 'SFU':
        dataset = SFUDataset()
    elif args.dataset == 'Corridor':
        dataset = CorridorDataset()
    elif args.dataset == 'ESSEX3IN1':
        dataset = ESSEX3IN1Dataset()
    else:
        raise ValueError('Unknown dataset: ' + args.dataset)

    imgs_db, imgs_q, GThard, GTsoft = dataset.load()

    if args.descriptor == 'HDC-DELF':
        from feature_extraction.feature_extractor_holistic import HDCDELF
        feature_extractor = HDCDELF()
    elif args.descriptor == 'AlexNet':
        from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
        feature_extractor = AlexNetConv3Extractor()
    elif args.descriptor == 'SAD':
        from feature_extraction.feature_extractor_holistic import SAD
        feature_extractor = SAD()
    elif args.descriptor in ('NetVLAD', 'PatchNetVLAD'):
        from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor
        from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
        if args.descriptor == 'NetVLAD':
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/netvlad_extract.ini')
        else:
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/speed.ini')
        assert os.path.isfile(configfile)
        config = configparser.ConfigParser()
        config.read(configfile)
        feature_extractor = PatchNetVLADFeatureExtractor(config)
    elif args.descriptor == 'CosPlace':
        from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
        feature_extractor = CosPlaceFeatureExtractor()
    elif args.descriptor == 'EigenPlaces':
        from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
        feature_extractor = EigenPlacesFeatureExtractor()
    else:
        raise ValueError('Unknown descriptor: ' + args.descriptor)

    if args.descriptor not in ('PatchNetVLAD', 'SAD'):
        print('===== Compute reference set descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        print('===== Compute cosine similarities S')
        db_D_holistic = db_D_holistic / np.linalg.norm(db_D_holistic, axis=1, keepdims=True)
        q_D_holistic  = q_D_holistic  / np.linalg.norm(q_D_holistic,  axis=1, keepdims=True)
        S = np.matmul(db_D_holistic, q_D_holistic.transpose())
    elif args.descriptor == 'SAD':
        print('===== Compute reference set descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        print('===== Compute similarities S from sum of absolute differences (SAD)')
        S = np.empty([len(imgs_db), len(imgs_q)], 'float32')
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                diff = db_D_holistic[i] - q_D_holistic[j]
                dim = len(db_D_holistic[0]) - np.sum(np.isnan(diff))
                diff[np.isnan(diff)] = 0
                S[i, j] = -np.sum(np.abs(diff)) / dim
    else:
        print('=== WARNING: The PatchNetVLAD code in this repository is not optimised and will be slow and memory consuming.')
        print('===== Compute reference set descriptors')
        db_D_holistic, db_D_patches = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic, q_D_patches = feature_extractor.compute_features(imgs_q)
        S = feature_extractor.local_matcher_from_numpy_single_scale(q_D_patches, db_D_patches)

    # similarity matrix
    fig = plt.figure()
    plt.imshow(S)
    plt.axis('off')
    plt.title('Similarity matrix S')
    sim_path = os.path.join(out_dir, "similarity_matrix.jpg")
    plt.savefig(sim_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {sim_path}")

    # matching
    print('===== Match images')
    M1 = matching.best_match_per_query(S)
    M2 = matching.thresholding(S, 'auto')

    # If no GT available, skip evaluation
    if GThard is None or GTsoft is None:
        print("===== Evaluation skipped (no ground truth provided for this dataset).")
        return

    # evaluation
    print('===== Evaluation')
    TP = np.argwhere(M2 & GThard)
    FP = np.argwhere(M2 & ~GTsoft)

    show_correct_and_wrong_matches.show(imgs_db, imgs_q, TP, FP)
    ex_path = os.path.join(out_dir, "examples_tp_fp.jpg")
    plt.savefig(ex_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {ex_path}")

    plt.figure(figsize=(8, 8))
    plt.imshow(M1, cmap='viridis')
    plt.title('Best match per query (M1)', fontsize=18)
    plt.axis('off')
    m1_path = os.path.join(out_dir, "matching_best_per_query.jpg")
    plt.savefig(m1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {m1_path}")

    plt.figure(figsize=(8, 8))
    plt.imshow(M2, cmap='viridis')
    plt.title('Thresholded matches (M2)', fontsize=18)
    plt.axis('off')
    m2_path = os.path.join(out_dir, "matching_thresholded.jpg")
    plt.savefig(m2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {m2_path}")

    P, R = createPR(S, GThard, GTsoft, matching='multi', n_thresh=100)
    plt.figure()
    plt.plot(R, P)
    plt.xlim(0, 1), plt.ylim(0, 1.01)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Result on {args.dataset}')
    plt.grid('on')
    plt.draw()
    pr_path = os.path.join(out_dir, "pr_curve.jpg")
    plt.savefig(pr_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {pr_path}")

    AUC = np.trapz(P, R)
    print(f'\n===== AUC (area under curve): {AUC:.3f}')

    maxR = recallAt100precision(S, GThard, GTsoft, matching='multi', n_thresh=100)
    print(f'\n===== R@100P (maximum recall at 100% precision): {maxR:.2f}')

    RatK = {}
    for K in [1, 5, 10]:
        RatK[K] = recallAtK(S, GThard, K=K)
    print(f'\n===== recall@K (R@K) -- R@1: {RatK[1]:.3f}, R@5: {RatK[5]:.3f}, R@10: {RatK[10]:.3f}')

    if os.environ.get('DISPLAY', '') != '':
        plt.show()


if __name__ == "__main__":
    main()
