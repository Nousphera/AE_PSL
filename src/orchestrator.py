import os
import json
import itertools
import copy
import traceback
from datetime import datetime
from argparse import Namespace

import json

import traceback
# from qcg.pilotjob.api.manager import LocalManager
# from qcg.pilotjob.api.job import Jobs


from calculate_computation_overhead import compute_flops
from main_AE_PSL_global_evaluation import run_2_stage_mpsl
from main_AE_PSL_local_evaluation import run_2_stage_mpsl_personalized
from utils.orchestrator_argument_utils import (
    build_base_argument_parser,
    expand_argument_parser_with_ae_pretraining_parameters,
    expand_argument_parser_with_distributed_learning_parameters,
    namespace_to_global_args_and_search_space,
    set_env_variables, expand_argument_parser_with_baseline_parameters
)



def setup_orchestrator_parser():
    """
    Constructs the parser by chaining the utility functions provided.
    Adds orchestrator-specific arguments.
    """
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_ae_pretraining_parameters(parser)
    parser = expand_argument_parser_with_distributed_learning_parameters(parser)
    parser = expand_argument_parser_with_baseline_parameters(parser)

    # Orchestrator specific controls
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help='Name/Tag for the experiment campaign folder.')
    parser.add_argument('--experiments_dir', type=str, default='experiment',
                        help='Name/Tag for the experiment campaign folder.')
    parser.add_argument('--resume_from_manifest', type=str, default=None,
                        help='Path to a manifest.json file to resume an interrupted campaign.')
    parser.add_argument('--test_num_workers', type=int, default=5,
                        help='num_workers provided to the test Dataloader. For Split Learning, we differentiate between num_workers for the train Dataloader and test_num_workers for the test Dataloader.')

    return parser


import os
import json
import itertools
from datetime import datetime


def generate_manifest(experiment_name, experiments_dir, global_args, search_space):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    campaign_folder_name = f"{timestamp}_{experiment_name}"
    campaign_dir = os.path.join(experiments_dir, campaign_folder_name)
    os.makedirs(os.path.join(campaign_dir, "logs"), exist_ok=True)

    # 1. Separate seeds from the rest of the search space
    # We use .get() to provide a default of [None] if no seeds are provided
    seeds = search_space.get("random_seed", [None])

    # Create a copy of the search space without the seed for the product
    grid_space = {k: v for k, v in search_space.items() if k != "random_seed"}

    # 2. Sort keys for deterministic ordering of the structural grid
    keys = sorted(grid_space.keys())
    values = [grid_space[k] for k in keys]
    grid_combinations = list(itertools.product(*values))

    # 3. Build jobs: Seeds are the OUTER loop
    # This ensures Job 1...N cover the full grid for Seed A
    # before starting Job N+1 with Seed B.
    jobs = []
    for seed in seeds:
        for combo in grid_combinations:
            job_params = dict(zip(keys, combo))
            if seed is not None:
                job_params["random_seed"] = seed

            jobs.append({
                "params": job_params
            })

    manifest = {
        "experiment_name": experiment_name,
        "created_at": timestamp,
        "global_args": global_args,
        "search_space": search_space,
        "jobs": jobs
    }

    manifest_path = os.path.join(campaign_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)

    print(f"Initialized Campaign: {campaign_dir}")
    print(f"Generated {len(jobs)} jobs ({len(seeds)} seeds x {len(grid_combinations)} configs).")

    return campaign_dir, manifest


def get_job_filename(params, experiment_name):
    """
    Creates a unique, readable filename for a job based on its varying parameters.
    """
    if not params:
        return "run_default"

    # Mapping for shortening common long parameter names
    abbreviations = {
        'learning_rate': 'lr',
        'start_lr': 'lr',
        'batch_size': 'bs',
        'random_seed': 'seed',
        'dataset': 'ds',
        'split_layer': 'split',
        'ae_latent_dim': 'lat',
        'ae_type': 'ae'
    }

    parts = []
    for k in sorted(params.keys()):
        key_str = abbreviations.get(k, k)
        val_str = str(params[k])
        parts.append(f"{key_str}_{val_str}")

    return "run_" + experiment_name + "_" + "_".join(parts)


def run_orchestrator(args: Namespace):


    # 1. Setup Campaign (New or Resume)
    if args.resume_from_manifest:
        if not os.path.exists(args.resume_from_manifest):
            print(f"Error: Manifest file not found at {args.resume_from_manifest}")
            return

        print(f"Resuming campaign from: {args.resume_from_manifest}")
        with open(args.resume_from_manifest, 'r') as f:
            manifest = json.load(f)

        global_args = manifest['global_args']
        jobs = manifest['jobs']
        campaign_dir = os.path.dirname(os.path.abspath(args.resume_from_manifest))

    else:
        # Separate constant arguments from search space lists
        global_args, search_space = namespace_to_global_args_and_search_space(args)

        # Remove orchestrator-specific args from global_args to keep it clean
        global_args.pop('experiment_name', None)
        global_args.pop('experiments_dir', None)
        global_args.pop('resume_from_manifest', None)

        campaign_dir, manifest = generate_manifest(args.experiment_name, args.experiments_dir, global_args, search_space)
        jobs = manifest['jobs']

    # 2. Configure Environment
    # We use the global_args to set env variables (data dirs, etc.)
    set_env_variables(global_args)

    total_jobs = len(jobs)
    print(f"Starting Execution of {total_jobs} jobs...")

    # 3. Execution Loop
    for i, job in enumerate(jobs):
        job_params = job['params']

        # Merge global args with specific job params
        current_run_args = copy.deepcopy(global_args)
        current_run_args.update(job_params)

        # include seed in job params if doing multi process runs via snellius
        if current_run_args['single_seed_per_process']:
            job_params['random_seed'] = current_run_args['random_seed']
            experiment_name = args.experiment_name
        else:
            experiment_name = ""

        # Determine output filename
        filename = get_job_filename(job_params, experiment_name)

        # We use absolute path for save_file_name to override the default behavior 
        # of saving into os.environ['MODEL_WEIGHTS_DIR']
        save_file_path = os.path.abspath(os.path.join(campaign_dir, "logs", filename))
        current_run_args['save_file_name'] = save_file_path
        # os.environ['MODEL_WEIGHTS_DIR'] = campaign_dir
        current_run_args['model_weights_dir'] = campaign_dir

        # Check for existing result (Resume Logic)
        # Note: save_experiment_results appends '.json'
        expected_output_file = save_file_path + ".json"

        if os.path.exists(expected_output_file):
            print(f"[{i + 1}/{total_jobs}] SKIPPING: {filename} (Result exists)")
            continue




        print(f"[{i + 1}/{total_jobs}] RUNNING: {filename}")
        print(f"   Params: {job_params}")

        try:
            # Execute the training stage directly
            # This handles AE pre-training/loading internally

            if current_run_args['profile_flops'] == True:
                compute_flops(current_run_args, job_params, manifest)
                return

            if current_run_args['dataset'] == 'femnist':
                run_2_stage_mpsl_personalized(current_run_args, job_params)
            else:
                run_2_stage_mpsl(current_run_args, job_params)

        except KeyboardInterrupt:
            print("\nOrchestrator interrupted by user.")
            break
        except Exception as e:
            print(f"!!! JOB FAILED: {filename}")
            print(f"Error: {e}")
            traceback.print_exc()
            # We continue to the next job instead of crashing the whole campaign
            # continue

    print("Orchestrator finished.")



if __name__ == "__main__":
    parser = setup_orchestrator_parser()
    args = parser.parse_args()

    run_orchestrator(args)