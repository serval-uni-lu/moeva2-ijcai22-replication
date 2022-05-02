def parse_moeva(metrics):
    config = metrics["config"]
    return [
        {
            "attack_name": config["attack_name"],
            "eps": config["eps_list"][i],
            **metrics["objectives_list"][i],
        }
        for i in range(len(metrics["objectives_list"]))
    ]


def parse_pgd(metrics):
    config = metrics["config"]
    return {
        "attack_name": config["loss_evaluation"],
        "eps": config["eps"],
        **metrics["objectives"],
    }


def parse_metrics(metrics):
    config = metrics["config"]
    parsed = {
        "n_state": config["n_initial_state"],
        "config_hash": metrics["config_hash"],
        "project_name": config["project_name"],
        "budget": config["budget"],
        "time": metrics["time"],
        "model": config["paths"]["model"],
        "reconstruction": config.get("reconstruction", None)
    }
    if metrics["config"]["attack_name"] == "moeva":
        return [{**parsed, **parsed_moeva} for parsed_moeva in parse_moeva(metrics)]
    else:
        return [{**parsed, **parse_pgd(metrics)}]
