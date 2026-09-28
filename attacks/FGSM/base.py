import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from attacks.attack_handler.base import Attack

from .attack_utilities import hex_id_to_bits, load_dataset, load_model, stuff_bits
from .mask_utils import (
    find_max_perturbations,
    generate_max_grad_mask,
    generate_mask_modify,
    gradient_perturbation_targeted,
)


class NewFGSMAttack(Attack):
    """
    Shared targeted black-box FGSM driver for CAN adversarial attacks.

    Subclasses select the attack variant by setting two class attributes:
      attack_type        -- 'dos' or 'spoof'
      default_target_id  -- hex CAN arbitration ID used when the config
                             doesn't override it ('000' for dos, '43f' for
                             spoof, matching the gear attack's actual
                             injected arbitration ID in the recorded data).

    Injection always perturbs ID + Data of a brand-new synthetic frame
    placed in an empty (green) row — identical for both variants.
    Modification perturbs an existing recorded frame that matches the
    target ID's bit-stuffed pattern:
      dos   -> ID + Data both perturbed (no projection to an existing ID).
      spoof -> only Data perturbed, ID kept fixed as target_id_bits.
    """

    attack_type = None
    default_target_id = None

    def __init__(self, cfg):
        self.cfg = cfg

    def fgsm_attack_injection(self, image, data_grad, ep):
        sign_data_grad = data_grad.sign()
        mask = generate_max_grad_mask(image, data_grad)

        if mask is None:
            return None

        perturbed_image = image + ep * sign_data_grad * mask
        perturbed_image = gradient_perturbation_targeted(image, perturbed_image, mask, mode='id_data')
        return torch.clamp(perturbed_image, 0, 1)

    def apply_injection(self, pack, test_model, target, data_grad, data_denorm, ep):
        perturbed_data = self.fgsm_attack_injection(data_denorm, data_grad, ep)

        if perturbed_data is None:
            print("No more space to inject")
            output = test_model(data_denorm)
            final_pred = output.max(1, keepdim=True)[1]
            final_score = torch.exp(output)[:, 1]
            return True, pack, final_pred, final_score, data_denorm

        with torch.no_grad():
            output = test_model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]
        final_score = torch.exp(output)[:, 1]

        if final_pred.item() == target.item():
            return True, pack + 1, final_pred, final_score, perturbed_data
        return False, pack, final_pred, final_score, perturbed_data

    def fgsm_attack_modification(self, image, data_grad, ep, target_id_bits, matched_rows, selected_rows_set, bit_pattern):
        perturb_id = (self.attack_type == 'dos')
        sign_data_grad = data_grad.sign()

        mask, matched_rows, selected_rows_set = generate_mask_modify(
            image, data_grad, matched_rows, selected_rows_set, bit_pattern, perturb_id
        )
        perturbed_image = image + ep * sign_data_grad * mask

        if perturb_id:
            perturbed_image = gradient_perturbation_targeted(image, perturbed_image, mask, mode='id_data')
        else:
            perturbed_image = gradient_perturbation_targeted(
                image, perturbed_image, mask, mode='data_only', target_id_bits=target_id_bits
            )

        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image, matched_rows, selected_rows_set

    def apply_modification(self, pack, test_model, target, data_grad, data_denorm, ep, target_id_bits, matched_rows, selected_rows_set, bit_pattern):
        perturbed_data, matched_rows, selected_rows_set = self.fgsm_attack_modification(
            data_denorm, data_grad, ep, target_id_bits, matched_rows, selected_rows_set, bit_pattern
        )

        with torch.no_grad():
            output = test_model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]
        final_score = torch.exp(output)[:, 1]

        if final_pred.item() == target.item():
            return True, pack + 1, final_pred, final_score, perturbed_data, matched_rows, selected_rows_set
        return False, pack, final_pred, final_score, perturbed_data, matched_rows, selected_rows_set

    def Attack_procedure(self, model, test_model, device, test_loader, ep, max_injection_perturbations, target_id_bits):
        all_preds = []
        all_labels = []
        all_scores = []
        n_image = 1

        middle_bits = "0001000"
        bit_pattern = stuff_bits('0' + target_id_bits + middle_bits)
        rgb_pattern = [(0.0, 0.0, 0.0) if bit == '0' else (1.0, 1.0, 1.0) for bit in bit_pattern]
        pattern_length = len(rgb_pattern)

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            current_target = target[0] if target.dim() > 0 else target

            initial_output = model(data)
            final_pred = initial_output.max(1, keepdim=True)[1]
            injection_count = 0
            modification_count = 0

            if current_target == 1:
                print("\nImage no:", n_image, "(Attack image)")
                pack = 1

                data.requires_grad = True
                model.eval()

                initial_output = model(data)
                loss = F.nll_loss(initial_output, target)

                model.zero_grad()
                loss.backward()
                data_grad = data.grad.data

                data_denorm = data
                continue_perturbation = True
                matched_rows = None
                selected_rows_set = None
                perturbation_type = "injection"
                _, max_modification_perturbations = find_max_perturbations(
                    data_denorm, pattern_length, rgb_pattern, matched_rows, ifprint=False
                )
                print("max_modification_perturbations", max_modification_perturbations)

                while continue_perturbation:
                    perturbed_data = data_denorm.clone().detach().to(device)
                    perturbed_data.requires_grad = True
                    model.eval()

                    if perturbation_type == "injection" and injection_count < max_injection_perturbations:
                        continue_perturbation, pack, final_pred, final_score, data_denorm = self.apply_injection(
                            pack, test_model, target, data_grad, perturbed_data, ep
                        )
                        injection_count += 1
                        if continue_perturbation and modification_count < max_modification_perturbations:
                            perturbation_type = "modification"
                    elif perturbation_type == "modification" and modification_count < max_modification_perturbations:
                        continue_perturbation, pack, final_pred, final_score, data_denorm, matched_rows, selected_rows_set = self.apply_modification(
                            pack, test_model, target, data_grad, perturbed_data, ep,
                            target_id_bits, matched_rows, selected_rows_set, bit_pattern
                        )
                        modification_count += 1
                        if continue_perturbation and injection_count < max_injection_perturbations:
                            perturbation_type = "injection"
                    else:
                        if injection_count >= max_injection_perturbations and modification_count >= max_modification_perturbations:
                            continue_perturbation = False
                        elif injection_count < max_injection_perturbations:
                            perturbation_type = "injection"
                        elif modification_count < max_modification_perturbations:
                            perturbation_type = "modification"
            else:
                data.requires_grad = True
                test_model.eval()
                initial_output = test_model(data)
                final_pred = initial_output.max(1, keepdim=True)[1]
                final_score = torch.exp(initial_output)[:, 1]
                print(f"Image {n_image}: Benign Image (Skipping Perturbation)")

            print(f"Final perturbations: Injection={injection_count}, Modification={modification_count}")
            print(f"Image {n_image}, Truth Labels {target.item()}, Final Pred {final_pred.cpu().numpy()}")

            n_image += 1
            all_preds.extend(final_pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_scores.extend(final_score.detach().cpu().numpy())

        return np.array(all_preds).squeeze(), np.array(all_labels), np.array(all_scores).squeeze()

    def apply(self, **kwargs):
        cfg = self.cfg
        type_cfg = cfg.get(self.attack_type, {})

        target_id = type_cfg.get('target_id') or cfg.get('target_id') or self.default_target_id
        target_id_bits = hex_id_to_bits(target_id)
        print(f"Attack type: {self.attack_type}, Target ID: 0x{target_id} -> {target_id_bits}")

        dataset_path = os.path.join(cfg['dir_path'], "..", "datasets", cfg['dataset_name'])
        test_dataset_dir_name = type_cfg.get('test_dataset_dir') or cfg.get('test_dataset_dir')
        test_dataset_dir = os.path.join(dataset_path, "test", test_dataset_dir_name)
        test_label_file = os.path.join(test_dataset_dir, "labels.txt")

        surrogate_model_path = cfg['surrogate_model']
        target_model_path = cfg.get('target_model') or surrogate_model_path

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        _, test_loader = load_dataset(test_dataset_dir, test_label_file, is_train=False)
        print("loaded test dataset")

        model, test_model = load_model(surrogate_model_path, target_model_path, device)

        epsilon = cfg.get('epsilon', 1)
        max_injection_perturbations = cfg.get('max_injection_limit', 30)

        timestamp = datetime.now().strftime("_%Y_%m_%d_%H%M%S")
        output_path = os.path.join(dataset_path, "adversarial_images", f"FGSM_{self.attack_type}{timestamp}")
        os.makedirs(output_path, exist_ok=True)

        st = time.time()
        print("Start time:", st)
        print(f"attack_type={self.attack_type}, target_id=0x{target_id}, max_injections={max_injection_perturbations}")

        preds, labels, scores = self.Attack_procedure(
            model, test_model, device, test_loader, epsilon,
            max_injection_perturbations, target_id_bits
        )

        et = time.time()
        print("End time:", et, "Execution Time:", et - st)

        return preds, labels, scores, output_path
