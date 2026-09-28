import os
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

from evaluate import evaluation_metrics

from .base import Attack
from ..New_FGSM.dos import NewFGSMDoS
from ..New_FGSM.spoof import NewFGSMSpoof

ATTACK_REGISTRY = {
    "dos": NewFGSMDoS,
    "spoof": NewFGSMSpoof,
}


class FGSM(Attack):
    """
    Dispatch key ('evasion_attack' in config.yaml) must match this class
    name case-insensitively -- see attacks/attack_handler/__init__.py and
    src/get_attack.py.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, **kwargs):
        cfg = self.cfg
        attack_mode = (cfg.get('attack_mode') or 'dos').lower()

        AttackClass = ATTACK_REGISTRY.get(attack_mode)
        if AttackClass is None:
            raise ValueError(f"Unknown FGSM attack_mode: {attack_mode}")

        print(f"Selected adversarial attack: FGSM ({attack_mode})")

        log_file_dir = os.path.join(cfg['dir_path'], "..", "datasets", cfg['dataset_name'], "log_files")
        os.makedirs(log_file_dir, exist_ok=True)
        timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")
        log_file = os.path.join(log_file_dir, f"fgsm_{attack_mode}_attack{timestamp}.log")

        attack = AttackClass(cfg)

        with open(log_file, "w") as f:
            with redirect_stdout(f), redirect_stderr(f):
                print("Making call to the attack : ", f"FGSM ({attack_mode})")
                preds, labels, scores, output_path = attack.apply()
                evaluation_metrics(preds, labels, cfg, all_scores=scores)
                return output_path
