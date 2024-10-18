import argparse
import nibabel as nib
from wirehead import WireheadGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_AUTOGRAPH_VERBOSITY"] = "0"

import tensorflow as tf

WIREHEAD_CONFIG = "configL.yaml"
DATA_FILES = []
DATA_FILES += glob("/data/users1/afani1/babyBrains/strip2/labels/*nii.gz")

prior_means = '/data/users1/afani1/babyBrains/strip2/sr/SynthSR/SynthSR/mask_priors2/prior_means.npy'
prior_stds = '/data/users1/afani1/babyBrains/strip2/sr/SynthSR/SynthSR/mask_priors2/prior_stds.npy'

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except Exception:
    pass


def create_generator(file_id=0):
    """Creates an iterator that returns data for mongo.
    Should contain all the dependencies of the brain generator
    Preprocessing should be applied at this phase
    yields : tuple ( data: tuple ( data_idx: torch.tensor, ) , data_kinds : tuple ( kind : str))
    """
    training_seg = DATA_FILES[file_id]
    nifti_img = nib.load(training_seg)
    affine = nifti_img.affine
    print(affine)

    brain_generator = BrainGenerator(
        training_seg,
        randomise_res=True,
    )

    print(f"Generator {file_id}: SynthSeg is using {training_seg}", flush=True)

    while True:
        img, lab = brain_generator.generate_brain()
        img, lab = preprocessing_pipe(img, lab, affine)
        print(img)
        print(img.shape)
        print(lab)
        print(lab.shape)
        yield (img, lab)
        gc.collect()


def run_wirehead_generator(get_file_id, n_samples=10000):
    # or whatever 15 minutes equals
    idx = 0
    while True:
        brain_generator = create_generator(get_file_id(idx))
        wirehead_generator = WireheadGenerator(
            generator=brain_generator, config_path=WIREHEAD_CONFIG
        )
        for i in range(n_samples):
            wirehead_generator.cycle()
        del brain_generator
        gc.collect()  # i am paranoid
        idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run wirehead generators")
    parser.add_argument("num_tasks", type=int, help="Number of running tasks")
    parser.add_argument(
        "num_generators", type=int, help="Number of generators to run"
    )
    parser.add_argument(
        "generator_id", type=int, help="Which of the generators to run"
    )
    args = parser.parse_args()

    if slurm_job_id := os.environ.get("SLURM_JOB_ID"):
        slurm_job_id = int(slurm_job_id)
    else:
        print("Not in a slurm job, exiting")
        exit()

    def get_file_id(idx):
        return (
            (slurm_job_id + idx * args.num_tasks) * args.num_generators
            + args.generator_id
        ) % len(DATA_FILES)

    # Calculate file index
    file_idx = (slurm_job_id * args.num_generators + args.generator_id) % len(
        DATA_FILES
    )
    # Run the generator
    run_wirehead_generator(get_file_id)
