#!/usr/bin/env python3
import argparse
import os.path as osp
import pickle


# from garage import wrap_experiment
# from garage.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_dir', type=str, default='./data')
    args = parser.parse_args()

    saved_dir = osp.expanduser(args.saved_dir)

    # @wrap_experiment
    # def load_sac_metaworld_batch(ctxt, saved_dir):
    #     """Resume a PyTorch experiment.
    #
    #     Args:
    #         ctxt (garage.experiment.ExperimentContext): The experiment
    #             configuration used by Trainer to create the snapshotter.
    #         saved_dir (str): Path where snapshots are saved.
    #
    #     """
    #     ctxt.snapshot_dir = saved_dir
    #
    #     trainer = Trainer(snapshot_config=ctxt)
    #     trainer.restore(from_dir=saved_dir)
    #     # trainer.resume()
    #
    #     print()


    # load_sac_metaworld_batch(saved_dir=saved_dir)

    with open(osp.join(saved_dir, 'params.pkl'), 'rb') as f:
        snapshot = pickle.load(f)

    replay_buffer = snapshot['replay_buffer']

    print()
