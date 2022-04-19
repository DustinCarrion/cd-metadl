""" Combine the ingestion and scoring processes. 
Usage example: 
    python run.py --input_dir=<dir> --submission_dir=<submission_dir_path> 
"""

from shlex import split
from subprocess import call

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("verbose", False, "Verbose mode.")

flags.DEFINE_integer("debug_mode", 0, "Debug mode.")

flags.DEFINE_integer("max_time", 60, 
    "Maximum time in seconds per testing task.")

flags.DEFINE_boolean("overwrite_previous_results", True, 
    "Overwrite results flag.")

flags.DEFINE_string("input_dir", "../../public_data", "Path to the dataset "
    "directory containing the meta_train and meta_test data.")

flags.DEFINE_string("output_dir_ingestion","../../sample_result_submission", 
    "Path to the output directory for the ingestion program.")

flags.DEFINE_string("submission_dir", "../baselines/random", 
    "Path to the directory containing the algorithm to use.")

flags.DEFINE_string("output_dir_scoring", "../../scoring_output", 
    "Path to the ourput directory for the scoring program.")


def main(argv) -> None:
    """ Runs the ingestion and scoring programs sequentially, as they are 
    handled in CodaLab.
    """
    del argv
    
    verbose = FLAGS.verbose
    debug_mode = FLAGS.debug_mode
    max_time = FLAGS.max_time
    overwrite_previous_results = FLAGS.overwrite_previous_results
    input_dir = FLAGS.input_dir
    output_dir_ingestion = FLAGS.output_dir_ingestion
    submission_dir = FLAGS.submission_dir
    output_dir_scoring = FLAGS.output_dir_scoring
    
    command_ingestion = "python -m cdmetadl.ingestion_program.ingestion " \
        + f"--verbose={verbose} " \
        + f"--debug_mode={debug_mode} " \
        + f"--max_time={max_time} " \
        + f"--overwrite_previous_results={overwrite_previous_results} " \
        + f"--input_dir={input_dir} " \
        + f"--output_dir_ingestion={output_dir_ingestion} " \
        + f"--submission_dir={submission_dir}"

    command_scoring = "python -m cdmetadl.scoring_program.scoring " \
        + f"--verbose={verbose} " \
        + f"--debug_mode={debug_mode} " \
        + f"--overwrite_previous_results={overwrite_previous_results} " \
        + f"--output_dir_ingestion={output_dir_ingestion} " \
        + f"--output_dir_scoring={output_dir_scoring}"
        
    cmd_ing = split(command_ingestion)
    cmd_sco = split(command_scoring)
    
    call(cmd_ing)
    call(cmd_sco)


if __name__ == "__main__":
    app.run(main)