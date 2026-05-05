import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass


def _default_wandb_dir():
    autodl_tmp = "/root/autodl-tmp"
    if os.path.isdir(autodl_tmp):
        return os.path.join(autodl_tmp, "wandb")
    return None


def _config_to_dict(cfg):
    if is_dataclass(cfg):
        payload = asdict(cfg)
    else:
        payload = dict(cfg)
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }


def init_wandb_run(cfg):
    if not getattr(cfg, "wandb_enabled", False):
        return None

    import wandb

    wandb_dir = cfg.wandb_dir or _default_wandb_dir()
    if wandb_dir is not None:
        os.makedirs(wandb_dir, exist_ok=True)

    tags = list(cfg.wandb_tags or ())
    if cfg.direction_method not in tags:
        tags.append(cfg.direction_method)
    if cfg.model_alias not in tags:
        tags.append(cfg.model_alias)

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_name,
        group=cfg.wandb_group,
        tags=tags,
        mode=cfg.wandb_mode,
        dir=wandb_dir,
        config=_config_to_dict(cfg),
    )


@contextmanager
def wandb_run_context(cfg):
    run = init_wandb_run(cfg)
    try:
        yield run
    finally:
        if run is not None:
            run.finish()


def wandb_log(payload, step=None):
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    wandb.log(payload, step=step)


def wandb_save(path: str):
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None or not os.path.exists(path):
        return

    wandb.save(path)


def wandb_artifact(name: str, artifact_type: str, paths):
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    artifact = wandb.Artifact(name=name, type=artifact_type)
    for path in paths:
        if os.path.isdir(path):
            artifact.add_dir(path)
        elif os.path.exists(path):
            artifact.add_file(path)
    wandb.log_artifact(artifact)
