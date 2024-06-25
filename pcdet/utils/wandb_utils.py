import os

import wandb

def init_or_resume_wandb_run(wandb_id_file_path,
                             config):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file. 
        
        Returns the config, if it's not None it will also update it first
        
        NOTE:
            Make sure that wandb_id_file_path.parent exists before calling this function
    """
    # if the run_id was previously saved, resume from there
    
    if wandb_id_file_path.exists():
        resume_id = wandb_id_file_path.read_text()
        wandb.init(name=config["wandb"]["run_name"], config=config,
                            project=config["wandb"]["project"],
                            entity=config["wandb"]["entity"],
                            group=config["wandb"]["group"],
                            job_type=config["wandb"]["job_type"],
                            id=resume_id, resume='must', dir=config["wandb"]["dir"])
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        run = wandb.init(name=config["wandb"]["run_name"], config=config,
                            project=config["wandb"]["project"],
                            entity=config["wandb"]["entity"],
                            group=config["wandb"]["group"],
                            job_type=config["wandb"]["job_type"], dir=config["wandb"]["dir"])
        wandb_id_file_path.write_text(str(run.id))
        
def is_wandb_enabled(cfg):
    wandb_enabled = False
    if cfg.get('WANDB'):
        wandb_enabled = cfg.WANDB.ENABLED
    if cfg.get('GLOBAL_RANK'):
        wandb_enabled &= cfg.GLOBAL_RANK == 0
    # TODO Check WANDB_API_KEY validity, try except?
    if os.environ.get('WANDB_API_KEY') is None:
        wandb_enabled = False
    return wandb_enabled


def init(cfg, args, job_type='train_eval', eval_tag=''):
    """
        Initialize wandb by passing in config
    """
    if not is_wandb_enabled(cfg):
        return

    dir = cfg['WANDB'].get('dir', None)
    tag_list = cfg['WANDB'].get('tag', None)
    tags = []
    if tag_list is not None:
        tags += tag_list
    name = cfg.TAG
    if args.extra_tag != 'default':
        name = name + '-' + args.extra_tag
        tags += [args.extra_tag]
    if len(eval_tag):
        name = name + '-' + eval_tag
        tags += [eval_tag]

    wandb.init(name=name,
               config=cfg,
               project=cfg.WANDB.PROJECT,
               entity=cfg.WANDB.ENTITY,
               tags=tags,
               job_type=job_type,
               dir=dir)


def log(cfg, log_dict, step):
    if not is_wandb_enabled(cfg):
        return

    assert isinstance(log_dict, dict)
    assert isinstance(step, int)

    wandb.log(log_dict, step)


def summary(cfg, log_dict, step, highest_metric=-1):
    """
    Wandb summary information
    Args:
        cfg
    """

    if not is_wandb_enabled(cfg):
        return

    assert isinstance(log_dict, dict)
    assert isinstance(step, int)

    metric = log_dict.get(cfg.WANDB.get('SUMMARY_HIGHEST_METRIC'))
    if metric is not None and metric > highest_metric:
        # wandb overwrites summary with last epoch run. Append '_best' to keep highest metric
        for key, value in log_dict.items():
            wandb.run.summary[key + '_best'] = value
        wandb.run.summary['epoch'] = step
        highest_metric = metric

    return highest_metric


def log_and_summary(cfg, log_dict, step, highest_metric=-1):
    log(cfg, log_dict, step)
    return summary(cfg, log_dict, step, highest_metric)
