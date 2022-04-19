# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, 
# AND THE WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL 
# PROPERTY RIGHTS. IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE 
# LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES 
# WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF 
# SOFTWARE, DOCUMENTS, MATERIALS, PUBLICATIONS, OR INFORMATION MADE AVAILABLE 
# FOR THE CHALLENGE. 
#
# Main contributors: Dustin Carrión-Ojeda, Ihsan Ullah, Isabelle Guyon, and 
# Sergio Escalera.
# March-August 2022
# Originally inspired by code: Ben Hamner, Kaggle, March 2013
#
# This is the "ingestion program" written by the organizers. This program also 
# runs on the challenge platform to test your code.
#
# Usage: 
# python ingestion.py input_dir output_dir program_dir submission_dir
#
# The input directory input_dir (e.g. public_data/) contains several datasets 
# formatted following this instructions: 
# https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat
#
# The output directory output_dir (e.g. sample_result_submission/) will store 
# the ground truth, the predicted values and the metadata (no subdirectories):
# 	task_{task_id}.true
# 	task_{task_id}.predict
#   task_{task_id}_metadata.pkl
#
# The ingestion program directory program_dir must be this directory
#
# The code directory submission_dir (e.g. baselines/random/) must contain your 
# code submission model.py, it can also contain other helpers and configuration 
# files.


import os
import time 
import datetime
import pickle
import numpy as np
from sys import path, exit, version
from absl import app, flags
from cdmetadl.ingestion_program.ingestion_helpers import *
from cdmetadl.ingestion_program.data_generator import TrainGenerator, \
    TestGenerator


# =============================== BEGIN OPTIONS ===============================
FLAGS = flags.FLAGS

# Verbose mode 
# True: show various progression messages (recommended value)
# False: no progression messages are shown
flags.DEFINE_boolean("verbose", False, "Verbose mode.")

# Debug mode
# 0: no debug
# 1: run the code normally, but limits the time to MAX_TIME (recommended value)
# 2: same as 1 + show the Python version and list the directories 
flags.DEFINE_integer("debug_mode", 0, "Debug mode.")

# Time budget
# Maximum time in seconds PER TESTING TASK
flags.DEFINE_integer("max_time", 60, 
    "Maximum time in seconds per testing task.")

# Overwrite results flag
# True: the previous output directory is overwritten
# False: the previous output directory is renamed (recommended value)
flags.DEFINE_boolean("overwrite_previous_results", True, 
    "Overwrite results flag.")

# Default location of directories
# If no arguments to ingestion.py are provided, these are the directories used. 
flags.DEFINE_string("input_dir", "../../../public_data", "Path to the dataset "
    "directory containing the meta_train and meta_test data.")
flags.DEFINE_string("output_dir_ingestion","../../../sample_result_submission", 
    "Path to the output directory for the ingestion program.")
flags.DEFINE_string("submission_dir", "../../baselines/random", 
    "Path to the directory containing the algorithm to use.")


# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# Program version
VERSION = 1.1


def ingestion(argv) -> None:
    del argv
    
    VERBOSE = FLAGS.verbose
    DEBUG_MODE = FLAGS.debug_mode
    MAX_TIME = FLAGS.max_time
    OVERWRITE_PREVIOUS_RESULTS = FLAGS.overwrite_previous_results
    
    start_time = time.time() 
    
    # Define the path to the directories
    input_dir = os.path.abspath(FLAGS.input_dir)
    output_dir = os.path.abspath(FLAGS.output_dir_ingestion)
    submission_dir = os.path.abspath(FLAGS.submission_dir)
    
    vprint(f"Using input_dir: {input_dir}", VERBOSE)
    vprint(f"Using output_dir: {output_dir}", VERBOSE)
    vprint(f"Using submission_dir {submission_dir}", VERBOSE)
        
    # Show library version and directory structure
    if DEBUG_MODE >= 2: 
        print(f"Ingestion program version: {VERSION}\n")
        print(f"Python version: {version}")
        show_dir(".")
        
    # Import your model
    path.insert(1, submission_dir)
    try:
        from model import MyMetaLearner, MyLearner
    except:
        print(f"MyMetaLearner and MyLearner not found in {submission_dir}"
            + "/model.py")
        exit(1)        
        
    vprint(f"{'#'*45}\nIngestion Program Starts\n{'-'*45}\n", VERBOSE)
    
    vprint(f"Checking directories", VERBOSE)
    exist_dir(input_dir)
    exist_dir(submission_dir)

    vprint(f"Creating train generator", VERBOSE)
    # Define the configuration for the train generator
    train_generator_config = {
        "data_format": "episode",
        "train_pool_size": 0.75,
        "num_ways": 5,
        "min_s": 1,
        "max_s": 20,
        "fixed_query_size": True,
        "query_size": 20
    }
    config_file = os.path.join(submission_dir, "config.yaml")
    if os.path.exists(config_file):
        train_generator_custom_config = load_yaml(config_file)
        train_generator_config.update(train_generator_custom_config)
    
    # Initialize the train generator
    train_generator = TrainGenerator(input_dir, 
        data_format = train_generator_config["data_format"],
        train_pool_size = train_generator_config["train_pool_size"],
        num_ways = train_generator_config["num_ways"],
        min_s = train_generator_config["min_s"],
        max_s = train_generator_config["max_s"],
        fixed_query_size = train_generator_config["fixed_query_size"],
        query_size = train_generator_config["query_size"],
        verbose = VERBOSE)
    meta_train_generator = train_generator.meta_train_generator
    meta_valid_generator = train_generator.meta_valid_generator

    vprint(f"Training the meta-learner", VERBOSE)
    meta_learner = MyMetaLearner(train_generator.num_ways, 
        train_generator.total_train_classes)
    learner = meta_learner.meta_fit(meta_train_generator, meta_valid_generator)
    learner.save(submission_dir)
    
    vprint(f"Creating test generator", VERBOSE)
    test_generator = TestGenerator(input_dir, 
        min_w = 2,
        max_w = 20,
        min_s = 1,
        max_s = 20,
        fixed_query_size = False,
        private_info = False,
        verbose = VERBOSE)
    meta_test_generator = test_generator.meta_test_generator
    
    vprint(f"Creating output directory", VERBOSE)
    if not OVERWRITE_PREVIOUS_RESULTS:
        timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        mvdir(output_dir, f"{output_dir}_{timestamp}") 
    mkdir(output_dir) 
    
    vprint(f"Testing the submission", VERBOSE)
    number_of_test_tasks = 1000
    for i, task in enumerate(meta_test_generator(number_of_test_tasks)):
        task_start_time = time.time()
        vprint(f"Creating the learner for task {i+1}", VERBOSE)
        learner = MyLearner() 
        learner.load(submission_dir)   

        vprint(f"Training the learner", VERBOSE)        
        dataset_train = (task.support_set[0], 
            task.support_set[1], 
            task.num_ways, 
            task.num_shots)
        predictor = learner.fit(dataset_train)
        
        vprint(f"Predicting the labels of the query set", VERBOSE)
        y_pred = predictor.predict(task.query_set[0])
        
        task_execution_time = time.time() - task_start_time
        if DEBUG_MODE >= 1:
            if task_execution_time > MAX_TIME:
                print(f"Task {i+1} exceeded the maximum time allowed. Max "
                    + f"time {MAX_TIME}, execution time {task_execution_time}")
                exit(1)
        
        vprint(f"Saving the predictions", VERBOSE)
        file_name = f"{output_dir}/task_{i+1}"
        np.savetxt(f"{file_name}.true", task.query_set[1], fmt='%d')
        np.savetxt(f"{file_name}.predict", y_pred, fmt='%f')
        
        vprint(f"Saving the metadata", VERBOSE)
        task_metadata = {
            "dataset": task.dataset,
            "num_ways": task.num_ways,
            "classes": sorted(task.classes),
            "num_shots": task.num_shots,
            "query_size": len(task.query_set[1]),
            "exec_time": task_execution_time
        }
        with open(f"{file_name}_metadata.pkl", "wb") as f: 
            pickle.dump(task_metadata, f)
            
        vprint(f"Task {i+1} finished in {task_execution_time} seconds", 
            VERBOSE)   
    
    total_execution_time = time.time() - start_time
    with open(f"{output_dir}/metadata_ingestion", "w") as global_metadata_file:
        global_metadata_file.write(f"Total execution time: "
            + f"{total_execution_time}\n")
        global_metadata_file.write(f"Number of datasets: "
            + f"{test_generator._number_of_datasets}\n")
        global_metadata_file.write(f"Tasks: "
            + f"{number_of_test_tasks}")
    vprint(f"\nOverall time spent: {total_execution_time} seconds", VERBOSE)
        
    vprint(f"\n{'-'*45}\nIngestion Program Finished Successfully\n{'#'*45}\n", 
        VERBOSE)

if __name__ == "__main__":	
    app.run(ingestion)