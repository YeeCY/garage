#!/usr/bin/env python3
import click
import os.path as osp


from garage import wrap_experiment
from garage.trainer import Trainer


if __name__ == "__main__":
    @click.command()
    @click.option('--saved_dir',
                  required=True,
                  help='Path where snapshots are saved.')
    @wrap_experiment
    def load_sac_metaworld_batch(ctxt, saved_dir):
        """Resume a PyTorch experiment.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by Trainer to create the snapshotter.
            saved_dir (str): Path where snapshots are saved.

        """
        saved_dir = osp.expanduser(saved_dir)

        trainer = Trainer(snapshot_config=ctxt)
        trainer.restore(from_dir=saved_dir)
        trainer.resume()

        print()


    # s = np.random.randint(0, 1000)
    load_sac_metaworld_batch()
