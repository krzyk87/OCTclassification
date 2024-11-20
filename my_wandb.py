import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


class MyWandbMetricsLogger(WandbMetricsLogger):
    def on_epoch_end(self, epoch, logs=None):
        logs = dict() if logs is None else {f"epoch/{k}": v for k, v in logs.items()}
        logs["epoch/epoch"] = epoch
        lr = self._get_lr()
        if lr is not None:
            logs["epoch/learning_rate"] = lr
        logs["val_loss"] = logs["epoch/val_loss"]
        logs["val_accuracy"] = logs["epoch/val_accuracy"]

        wandb.log(logs)


def get_sweep_config(dataset, model_name, is_test=False):
    sweep_name = f"sweep_{dataset}_{model_name}"
    if is_test:
        sweep_name = sweep_name + "_test"
    sweep_config = {
            "project": f"{dataset}-project",
            "name": sweep_name,
            "method": "grid",
            "metric": {"goal": "minimize", "name": "val_loss"},
            "parameters": {
                # "learning_rate": {"distribution": "uniform", "min": 0.00001, "max": 0.001},
                "learning_rate": {"values": [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001]},
                "batch_size": {"values": [8, 16, 32]},
                "optimizer": {"values": ["Adam", "SGD", "RMS"]},
            },
        }
    return sweep_config
