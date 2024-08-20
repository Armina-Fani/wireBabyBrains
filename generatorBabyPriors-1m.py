import os
import gc
from wirehead import WireheadGenerator
import argparse
from nobrainer.processing.brain_generator import BrainGenerator
from preprocessing1 import preprocessing_pipe
import tensorflow as tf
from glob import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_AUTOGRAPH_VERBOSITY"] = "0"

WIREHEAD_CONFIG     = "config.yaml"
#PATH_TO_DATA        = "/data/users1/afani1/babyBrains/strip2/labels/"

prior_means = '/data/users2/tle89/here/priors/t1_priors-1m/prior_means.npy'
prior_stds = '/data/users2/tle89/here/priors/t1_priors-1m/prior_stds.npy'

DATA_FILES  = []
DATA_FILES += glob("/data/users2/tle89/here/1m/*nii.gz")

NUM_GENERATORS = 1

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

def create_generator(file_id=0):
    """ Creates an iterator that returns data for mongo.
        Should contain all the dependencies of the brain generator
        Preprocessing should be applied at this phase
        yields : tuple ( data0, data1, ...) """
    print("in generator")
    training_seg = DATA_FILES[file_id%len(DATA_FILES)]
    brain_generator = BrainGenerator(
        training_seg,
        randomise_res = True,
        prior_means=prior_means,
        prior_stds=prior_stds
    )
    print(f"Generator: SynthSeg is using {training_seg}")
    while True:
        print("In the while loop")
        img, lab = preprocessing_pipe(brain_generator.generate_brain())
        print("preprocessing has been completed")
        #print(np.unique(lab))
        yield (img, lab)
        gc.collect()

def run_wirehead_generator(file_id):
    brain_generator = create_generator(file_id)
    wirehead_generator = WireheadGenerator(
        generator=brain_generator, config_path=WIREHEAD_CONFIG
    )
    wirehead_generator.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run wirehead generators")
    parser.add_argument(
        "num_generators", type=int, help="Number of generators to run"
    )
    parser.add_argument(
        "generator_id", type=int, help="Which of the generators to run"
    )
    args = parser.parse_args()

    slurm_job_id = int(os.environ.get("SLURM_JOB_ID"))

    # Calculate file index
    file_idx = (slurm_job_id * args.num_generators + args.generator_id) % len(
        DATA_FILES
    )
    # Run the generator
    run_wirehead_generator(file_idx)
