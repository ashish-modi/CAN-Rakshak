import os
from datetime import datetime

from .base import Attack
from ..New_Bit_Flip_attack import dos as bit_flip_dos
from ..New_Bit_Flip_attack import spoof as bit_flip_spoof
from ..New_Bit_Flip_attack.traffic_decoder import run as run_decode
from ..New_Bit_Flip_attack.evaluate_attack import run as run_evaluate
from ..New_Bit_Flip_attack.update_labels import run as run_update

ATTACK_REGISTRY = {
    "dos": bit_flip_dos,
    "spoof": bit_flip_spoof,
}


class BitFlipAttack(Attack):
    """
    Dispatch key ('evasion_attack' in config.yaml) must match this class
    name case-insensitively -- see attacks/attack_handler/__init__.py and
    src/get_attack.py.

    attack_mode ('dos' | 'spoof') selects which module in
    attacks/New_Bit_Flip_attack/ runs stage 1 (attack.run); stages 2-4
    (decode/evaluate/update) are attack-mode-agnostic and shared.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, **kwargs):
        cfg    = self.cfg
        rounds = cfg.get('rounds', 1)
        attack_mode = (cfg.get('attack_mode') or 'dos').lower()

        attack_module = ATTACK_REGISTRY.get(attack_mode)
        if attack_module is None:
            raise ValueError(f"Unknown BitFlipAttack attack_mode: {attack_mode}")

        mode_cfg     = cfg.get(attack_mode, {})
        decode_cfg   = cfg.get('decode', {})
        evaluate_cfg = cfg.get('evaluate', {})
        update_cfg   = cfg.get('update', {})

        tracksheet = mode_cfg['original_tracksheet']

        # Derive the dataset directory from the tracksheet path.
        # Tracksheet lives at <dataset_dir>/csv_files/<name>.csv,
        # so two dirname() calls reach <dataset_dir>.
        dataset_dir = os.path.dirname(os.path.dirname(tracksheet))
        test_dataset_dir = mode_cfg.get('test_dataset_dir', '')
        original_test_dir = os.path.join(dataset_dir, "test", test_dataset_dir)

        label_file         = os.path.join(original_test_dir, "labels.txt")
        timestamp          = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Results tree is named after the dispatch key, so it follows a rename of
        # this class instead of drifting from it. Runs made before the
        # NewBitFlipAttack -> BitFlipAttack rename still live under
        # Results/NewBitFlipAttack/ — point adversarial_defense.adv_examples_path
        # at whichever tree the run you want came from.
        attack_result_path = os.path.join(dataset_dir, "Results", type(self).__name__, attack_mode, timestamp)
        os.makedirs(attack_result_path, exist_ok=True)

        output_dir            = os.path.join(attack_result_path, mode_cfg['output_dir'])
        decoded_output_dir    = os.path.join(attack_result_path, decode_cfg['decoded_output_dir'])
        prediction_output_dir = os.path.join(attack_result_path, evaluate_cfg['prediction_output_dir'])
        tracksheet_dir        = os.path.join(attack_result_path, update_cfg['tracksheet_dir'])

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(decoded_output_dir, exist_ok=True)
        os.makedirs(prediction_output_dir, exist_ok=True)
        os.makedirs(tracksheet_dir, exist_ok=True)
        print(f"Selected adversarial attack: BitFlipAttack ({attack_mode})")
        print("orginal dataset dir: ", original_test_dir)
        print("Label file : ", label_file)
        # Round 0 reads the original images; subsequent rounds read the
        # perturbed images that were written to output_dir by the previous round.
        current_test_dir = original_test_dir
        for round_num in range(rounds):
            print(f"\n{'='*60}")
            print(f"  BitFlipAttack ({attack_mode}) — Round {round_num}")
            print(f"{'='*60}")

            # Generates perturbed images and packet_level_data_{round}.csv
            print(f"\n[Stage 1] Running attack (round={round_num}) ...")
            attack_params = {
                "test_data_dir":     current_test_dir,
                "test_label_file":   label_file,
                "packet_level_data": tracksheet,
                "model_path":        cfg['surrogate_model'],
                "output_path":       output_dir,
                "rounds":            round_num,
            }
            if attack_mode == "spoof":
                attack_params["benign_csv"]     = mode_cfg.get("benign_csv", "CAN_DATA/gear_test.csv")
                attack_params["ablation_frac"]  = mode_cfg.get("ablation_frac", 0.25)
                attack_params["exclude_ids"]    = mode_cfg.get("exclude_ids", [])
                attack_params["anchor_id"]  = mode_cfg.get("anchor_id", "43F")

            attack_module.run(attack_params)

            # packet_level_data_{round_num}.csv is written by attack_module.run into output_dir;
            # it contains the columns traffic_decoder requires: timestamp, can_id,
            # original_label, operation_label (and pred_label for rounds > 0).
            packet_level_csv = os.path.join(output_dir, f"packet_level_data_{round_num}.csv")
            traffic_file = os.path.join(decoded_output_dir, f"traffic_{round_num}.txt")
            print(f"\n[Stage 2] Decoding perturbed images (round={round_num}) ...")
            run_decode({
                "rounds":       round_num,
                "input_images": output_dir,
                "csv_file":     packet_level_csv,
                "output_file":  traffic_file,
            })

            # Scores decoded traffic against the target model;
            # writes tracksheets_CH/dos_test_track_{round_num}.csv
            pred_output = os.path.join(prediction_output_dir, f"preds_{round_num}.csv")
            print(f"\n[Stage 3] Evaluating against target model (round={round_num}) ...")
            run_evaluate({
                "rounds":         round_num,
                "model_path":     cfg['target_model'],
                "traffic_path":   traffic_file,
                "tracksheet":     packet_level_csv,
                "output_path":    pred_output,
                "tracksheet_dir": tracksheet_dir,
                "attack_mode":    attack_mode,
            })

            # save_preds writes dos_test_track_{round_num}.csv into tracksheet_dir
            next_tracksheet = os.path.join(tracksheet_dir, f"dos_test_track_{round_num}.csv")

            # Updates the label file using the new tracksheet predictions
            updated_label_file = os.path.join(
                original_test_dir,
                f"labels_round_{round_num + 1}.txt"
            )
            print(f"\n[Stage 4] Updating label file (round={round_num}) ...")
            run_update({
                "tracksheet":         next_tracksheet,
                "label_file":         label_file,
                "updated_label_file": updated_label_file,
            })

            tracksheet = next_tracksheet
            label_file = updated_label_file
            # Perturbed images (perturbed_image_*.png) live in output_dir;
            # all rounds after 0 must read from there.
            current_test_dir = output_dir

            print(f"\n  Round {round_num} complete.")
            print(f"  Next tracksheet : {tracksheet}")
            print(f"  Next label file : {label_file}")
