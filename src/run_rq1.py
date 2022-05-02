import json
import os
import logging
import subprocess

from src.config_parser.config_parser import get_config, merge_parameters

TABULATOR = ">>>"
launch_counter = 0


def launch_script(script):
    global launch_counter
    launch_counter += 1
    logger.info(script)
    subprocess.run(script)


def run():
    config_dir = config["config_dir"]
    for seed in config["seeds"]:
        logger.info(f"{TABULATOR*1} Running seed {seed} ...")
        for project in config["projects"]:
            logger.info(f"{TABULATOR*2} Running project {project} ...")
            for budget in config["budgets"]:
                logger.info(f"{TABULATOR * 3} Running budget {budget} ...")

                if "moeva" in config["attacks"]:
                    logger.info(f"{TABULATOR * 4} Running MoEvA ...")
                    eps_list = {"eps_list": config['eps_list']}
                    eps_list_str = json.dumps(eps_list, separators=(',', ':'))
                    launch_script([
                        "python", "-m", "src.experiments.united.04_moeva",
                        "-c", f"{config_dir}/moeva.yaml",
                        "-c", f"{config_dir}/{project}.yaml",
                        "-p", f"seed={seed}",
                        "-p", f"budget={budget}",
                        "-j", eps_list_str]
                    )

                # Run the rest
                if "pgd" in config["attacks"]:
                    logger.info(f"{TABULATOR * 4} Running pgd ...")
                    for eps in config["eps_list"]:
                        logger.info(f"{TABULATOR * 5} Running eps {eps} ...")

                        for loss_evaluation in config["loss_evaluations"]:
                            logger.info(
                                f"{TABULATOR * 6} Running loss_evaluation {loss_evaluation} ..."
                            )
                            launch_script([
                                "python", f"-m", f"src.experiments.united.01_pgd_united",
                                "-c", f"{config_dir}/pgd.yaml",
                                "-c", f"{config_dir}/{project}.yaml",
                                "-p", f"seed={seed}",
                                "-p", f"budget={budget}",
                                "-p", f"eps={eps}",
                                "-p", f"loss_evaluation={loss_evaluation}"]
                            )


if __name__ == "__main__":
    config = get_config()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    run()
    logger.info(f"{launch_counter} run executed.")
