import os

from ..Genetic_algorithm.Adversarial_DoS import AdversarialDosAttack
from ..Genetic_algorithm.Adversarial_Fuzzy import AdversarialFuzzyAttack
from ..Genetic_algorithm.Adversarial_Spoof import AdversarialSpoofAttack
from .base import Attack


class GeneticAdvAttack(Attack):
    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self):
        cfg         = self.cfg
        attack_name = cfg['attack_mode'].lower()
        print(f"  Attack         : Genetic ({attack_name.upper()})")

        # Per-mode overrides, same shape as NewBitFlipAttack / New_FGSM: keys under
        # GeneticAdvAttack.<mode> win over the shared ones, so dos and spoof can carry
        # different models, frame budgets and thresholds without editing between runs.
        mode_cfg = cfg.get(attack_name)
        if isinstance(mode_cfg, dict):
            cfg = {**cfg, **mode_cfg}
            print(f"  Mode overrides : {', '.join(sorted(mode_cfg))}")

        project_root = os.path.join(cfg['dir_path'], "..")

        # GeneticAdvAttack.model_path wins when set — the <arch>_<name>.h5 scheme
        # cannot name a checkpoint that does not follow it (e.g. spoof_final_model.h5).
        explicit_model = cfg.get('model_path')
        if explicit_model:
            model_path = (explicit_model if os.path.isabs(explicit_model)
                          else os.path.join(project_root, explicit_model))
        else:
            model_path = os.path.join(
                project_root, "models",
                cfg['test_model'] + "_" + cfg['test_model_name'] + ".h5"
            )

        file_path = os.path.join(
            project_root, "datasets", cfg['dataset_name'],
            "test", cfg['test_dataset_dir'],
            cfg['file_name'][:-4] + "_test_data.npz"
        )
        print("Model Path : ", model_path)
        print("File Path  : ", file_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"IDS model not found: {model_path}\n"
                f"Set adversarial_perturbation.GeneticAdvAttack.model_path to the checkpoint "
                f"(relative to the project root), or make testing.model / testing.model_name "
                f"spell an existing models/<model>_<model_name>.h5."
            )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Attack input not found: {file_path}\n"
                f"Run Stage 1 with feature_extractor: GeneticExtractor and "
                f"GeneticExtractor.output_dir: {cfg['test_dataset_dir']} to build it."
            )

        ATTACK_REGISTRY = {
            "dos":   AdversarialDosAttack,
            "fuzzy": AdversarialFuzzyAttack,
            "spoof": AdversarialSpoofAttack,
        }

        AttackClass = ATTACK_REGISTRY.get(attack_name)
        if AttackClass is None:
            raise ValueError(f"Unknown attack mode: {attack_name}")

        # Defaults match the standalone reference implementation (Spoofing_attack.py
        # main()); the previously hardcoded 20 generations / 0.4 rate cut the search
        # budget to roughly a quarter of it.
        population_size = int(cfg.get('population_size', 100))
        max_generations = int(cfg.get('max_generations', 75))
        mutation_rate   = float(cfg.get('mutation_rate', 0.2))

        print(f"  GA params      : population={population_size}, "
              f"generations={max_generations}, mutation_rate={mutation_rate}")

        attack = AttackClass(
            model_path=model_path,
            file_path=file_path,
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=mutation_rate,
        )

        return attack.apply(cfg)
