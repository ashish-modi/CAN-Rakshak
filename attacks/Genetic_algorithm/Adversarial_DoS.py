#!/usr/bin/env python3
"""
Adversarial DoS Attack Generator using Genetic Algorithm

This script implements a genetic algorithm-based approach to generate adversarial attacks
against deep learning intrusion detection systems for vehicle CAN networks. The attack
focuses on DoS (Denial of Service) attack scenarios by modifying dummy rows in CAN frames
to evade detection while maintaining attack characteristics.

The genetic algorithm evolves adversarial examples through:
- Population-based search with crossover and mutation
- Fitness evaluation based on IDS confidence scores  
- Elitist selection to preserve best candidates
- Multi-generation evolution until successful evasion

Usage:
    python3 adversarial_dos_attack.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
from attacks.attack_handler.base import GeneticAttack

class AdversarialDosAttack(GeneticAttack):

    # Columns of a frame row that must stay 0 so the perturbed message keeps the
    # highest arbitration priority and therefore remains a DoS — the constraint of
    # Choi & Kim (WISA'21) Sec. 3.3, "the 3 bits are used to represent its priority
    # ... attackers can only modify the other remaining bits".
    #
    # GeneticExtractor writes each row as format(can_id, '029b'), i.e. the ID
    # right-aligned in 29 columns, so this trace's 11-bit IDs occupy columns 18-28
    # and columns 0-17 are never set anywhere in the data. The three priority bits
    # are thus the top three bits of that 11-bit field: columns 18, 19, 20. Guarding
    # columns 0-2 instead (the previous behaviour) protected padding that is already
    # zero while leaving the real priority bits free to be set.
    #
    # If GeneticExtractor is ever switched to the paper's own layout (Fig. 2:
    # 11-bit base identifier followed by an 18-bit extension), this becomes (0, 1, 2).
    #
    # CHANGED: kept as the 29-column default, but no longer read directly — see
    # self.priority_cols in __init__, which derives the same three columns for
    # whatever frame width the .npz carries (29 or 11).
    PRIORITY_COLS = (18, 19, 20)

    # Width of a standard CAN arbitration ID. The frame encoding right-aligns the
    # ID in the row, so this is what locates the priority bits in either layout.
    ID_BITS = 11

    def __init__(self, model_path, file_path, population_size=100, max_generations=75, mutation_rate=0.1):
        """
        Initialize the adversarial DoS attack generator.
        
        Args:
            model_path: Path to the trained IDS model (H5 format)
            population_size: Number of individuals in each genetic algorithm generation
            max_generations: Maximum number of generations to evolve before giving up
            mutation_rate: Probability of mutation occurring for each individual
        """
        
        super().__init__(model_path, file_path, population_size, max_generations, mutation_rate)
        self.original_dummy_rows = []
        self.num_cols        = self.x_test.shape[2]

        # ADDED: packets per frame, read from the data rather than hardcoded —
        # 29 for CANShield frames, 32 for a MULSAM window.
        self.num_rows        = self.x_test.shape[1]

        # ── ORIGINAL ─────────────────────────────────────────────────────────
        # self.mutable_cols    = [c for c in range(self.num_cols)
        #                         if c not in self.PRIORITY_COLS]
        # ─────────────────────────────────────────────────────────────────────
        # CHANGED: the priority columns are now located instead of hardcoded.
        # The ID is right-aligned in the row, so its top three bits — the
        # arbitration priority the DoS must keep at zero — start at
        # (num_cols - ID_BITS):
        #   num_cols 29 -> (18, 19, 20)  identical to PRIORITY_COLS above
        #   num_cols 11 -> ( 0,  1,  2)  the 11-bit MULSAM layout
        # Left at (18, 19, 20) on an 11-column frame this would have guarded
        # columns that do not exist while leaving the real priority bits free,
        # so the GA could drop the message's priority and stop being a DoS.
        id_start = max(0, self.num_cols - self.ID_BITS)
        self.priority_cols = tuple(range(id_start, id_start + 3))

        self.mutable_cols    = [c for c in range(self.num_cols)
                                if c not in self.priority_cols]

    def find_dummy_rows(self, frame):
        """
        Identify rows in a CAN frame that are completely zero (dummy rows).
        
        These dummy rows represent unused CAN message slots and are the only
        parts of the frame we can modify without destroying the attack payload.
        
        Args:
            frame: 29x29x1 numpy array representing a CAN frame
            
        Returns:
            List of row indices that contain all zeros
        """
        # ── ORIGINAL ─────────────────────────────────────────────────────────
        # return [i for i in range(29) if np.all(frame[i,:,0] == 0)]
        # ─────────────────────────────────────────────────────────────────────
        # CHANGED: 29 was the CANShield frame height. A MULSAM window is 32
        # packets, so rows 29-31 were never inspected — their dummy rows would
        # be invisible to the GA and left unperturbed.
        return [i for i in range(self.num_rows) if np.all(frame[i,:,0] == 0)]

    def mutate(self, frame):
        """
        Apply mutation to a frame by modifying only the original dummy rows.

        Mutation process, per Choi & Kim (WISA'21) Sec. 3.2 — "the Mutate operation
        selects a random bit and flips it with some probability":
        1. With probability = mutation_rate, pick one random original dummy row
        2. Flip one random bit in it, excluding PRIORITY_COLS so the message keeps
           the highest arbitration priority and stays a DoS (Sec. 3.3)

        The row is deliberately NOT cleared first. Clearing it — the previous
        behaviour — reset the row on every mutation, so each dummy row was pinned to
        exactly one set bit for the whole run and the reachable set was only the
        power-of-two arbitration IDs. Nothing in the paper's Mutate does this, and it
        is what left the fitness landscape flat: every candidate scored ~0.997,
        identical to an untouched frame, so selection had no gradient to follow.

        Flipping rather than setting is also the paper's wording, and it is what lets
        a bit be taken back out again once crossover has combined two parents.

        Args:
            frame: 29x29x1 numpy array to mutate

        Returns:
            Mutated copy of the input frame
        """
        mutated = frame.copy()

        if random.random() < self.mutation_rate and self.original_dummy_rows:
            row_idx     = random.choice(self.original_dummy_rows)
            bit_to_flip = random.choice(self.mutable_cols)

            mutated[row_idx, bit_to_flip, 0] = 1.0 - mutated[row_idx, bit_to_flip, 0]

        return mutated

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent frames to create offspring.
        
        Crossover strategy, per Choi & Kim (WISA'21) Sec. 3.2 — "generate z by
        choosing each bit in z from either x or y with equal probability":
        - Start with parent1 as the base (child inherits most characteristics)
        - Within each original dummy row, take every bit independently from either
          parent with probability 0.5
        - Only touch rows that were originally dummy rows (preserves attack payload)

        Inheritance is per BIT, not per row. Row-wise inheritance — the previous
        behaviour — can only ever copy a dummy row wholesale, so two parents each
        carrying one set bit in row r can never produce a child carrying both. Per-bit
        inheritance is the mechanism by which set bits accumulate across generations,
        which is what produces the multi-bit dummy rows the paper reports (Fig. 5 for
        DoS, Fig. 7 for spoofing).

        Args:
            parent1: First parent frame (29x29x1 numpy array)
            parent2: Second parent frame (29x29x1 numpy array)

        Returns:
            Child frame combining characteristics from both parents
        """
        child = parent1.copy()

        for i in self.original_dummy_rows:
            from_parent2 = np.random.rand(self.num_cols) < 0.5
            child[i, from_parent2, 0] = parent2[i, from_parent2, 0]

        return child

    def generate_adversarial_attack(self, dummy_row_threshold=10, max_frames=7000):
        """
        Main genetic algorithm to generate adversarial DoS attacks.
        
        Process:
        1. Load and filter frames based on dummy row count
        2. Create balanced dataset (70% attack, 30% benign)
        3. For each attack frame with sufficient dummy rows:
           - Initialize genetic algorithm population
           - Evolve through generations using crossover/mutation
           - Stop when successful evasion achieved (confidence < 0.5)
        4. Return complete adversarial dataset with labels
        
        Args:
            dummy_row_threshold: Minimum dummy rows required to process a frame
            max_frames: Maximum total frames to include in final dataset
            
        Returns:
            tuple: (final_test, y_test, orig_frame, generations_needed, frame_indices)
                - final_test: Adversarial frames ready for evaluation
                - y_test: Corresponding labels for the frames
                - orig_frame: Original frames before adversarial modification
                - generations_needed: List of generations required per attack frame
                - frame_indices: Source index in self.x_test of each returned frame,
                  which is what lets the decoder map perturbations back to CAN rows
        """
        final_test = []      # Adversarial/benign frames for evaluation
        orig_frame = []      # Original frames for comparison
        generations_needed = []  # Track genetic algorithm performance
        frame_indices = []   # Source index in x_test of each returned frame
        success_counter = 0
        attack_count = int(max_frames * 0.7)
        benign_count = max_frames - attack_count

        print(f"Using {attack_count} attack frames and {benign_count} benign frames")

        benign_indices = np.where(self.y_test == 0)[0][:benign_count]

        if len(benign_indices) < benign_count:
            benign_count = len(benign_indices)
            print(f"Warning: Only {benign_count} benign frames available")

        for i in benign_indices:
            final_test.append(self.x_test[i])
            orig_frame.append(self.x_test[i])
            frame_indices.append(int(i))

        print("Starting genetic algorithm for adversarial attack generation...")

        attack_frames_processed = 0
        attack_indices = np.where(self.y_test == 1)[0]  # Find all attack frames

        for i in attack_indices:
            if attack_frames_processed >= attack_count:
                break

            dummy_rows = self.find_dummy_rows(self.x_test[i])

            if len(dummy_rows) <= dummy_row_threshold:
                continue  # Skip this frame
                final_test.append(self.x_test[i])
                orig_frame.append(self.x_test[i])
                attack_frames_processed += 1
            else:
                self.original_dummy_rows = dummy_rows.copy()
                frame_copy = self.x_test[i].copy()

                for row_idx in dummy_rows:
                    # Avoid the priority bits — see PRIORITY_COLS. randint(3, 28)
                    # here guarded columns 0-2, which are never set in this encoding
                    # anyway, and could set 18-20 and drop the message's priority.
                    bit_to_flip = random.choice(self.mutable_cols)
                    frame_copy[row_idx, bit_to_flip, 0] = 1

                mut_rate = self.mutation_rate  # Save current mutation rate
                self.mutation_rate = 1.0       # Force mutation for population diversity
                population = [self.mutate(frame_copy.copy()) for _ in range(self.population_size)]
                self.mutation_rate = mut_rate  # Restore original mutation rate

                success_generation = -1  # Track when successful attack was found
                success_counter = 0
                for generation in range(self.max_generations):

                    # Batch predict all individuals at once
                    scores = self.calculate_confidence_batch(population)
                    scores = np.nan_to_num(scores, nan=1.0)

                    # Check for successful evasion
                    success_idx = np.where(scores < 0.5)[0]
                    if len(success_idx) > 0:
                        winner = success_idx[0]
                        success_counter += 1
                        best_score = float(scores[winner])
                        final_test.append(population[winner])
                        orig_frame.append(self.x_test[i])
                        success_generation = generation + 1
                        break

                    inv_scores = 1.0 - scores

                    total = inv_scores.sum()
                    num_positive = np.count_nonzero(inv_scores)

                    if total <= 1e-12 or num_positive < 2:
                        selection_probs = np.ones_like(inv_scores) / len(inv_scores)
                    else:
                        selection_probs = inv_scores / total

                    indices = np.arange(len(population))
                    new_pop = []

                    # Algorithm 1 fills POP_{i+1} with Psize offspring and carries no
                    # elite over, and it selects x and y with two independent draws
                    # ("Select x ...; Select y ..."), so the same individual may be
                    # picked twice. Keeping an elite and forcing replace=False were
                    # both additions to the paper.
                    while len(new_pop) < self.population_size:
                        p1_idx, p2_idx = np.random.choice(
                            indices, size=2, p=selection_probs
                        )

                        child = self.crossover(
                            population[p1_idx],
                            population[p2_idx]
                        )

                        child = self.mutate(child)
                        new_pop.append(child)

                    population = new_pop
                    # print("Success counter : ", success_counter)

                if success_generation == -1:
                    # `population` was replaced on the final iteration, so the last
                    # computed `scores` describe the previous generation — re-score
                    # the survivors rather than indexing them with stale fitness.
                    scores = np.nan_to_num(self.calculate_confidence_batch(population), nan=1.0)
                    best_idx = int(np.argmin(scores))
                    best_score = float(scores[best_idx])
                    final_test.append(population[best_idx])
                    orig_frame.append(self.x_test[i])
                    success_generation = self.max_generations

                generations_needed.append(success_generation)
                frame_indices.append(int(i))
                attack_frames_processed += 1

                # One line per frame — flushed so `tail -f` tracks a redirected run.
                outcome = (f"evaded in generation {success_generation}"
                           if success_counter else
                           f"NOT evaded in {success_generation} generations, "
                           f"best score {best_score:.4f}")
                print(f"Attack frame {attack_frames_processed}/{attack_count} "
                      f"(frame {i}): {outcome}", flush=True)

        y_final = np.zeros(len(final_test))
        y_final[benign_count:] = 1  # Attack samples start after benign samples

        return (np.array(final_test), y_final, np.array(orig_frame),
                generations_needed, np.array(frame_indices))

    def save_adversarial_attack(self, frame, output_file="adversarial_dos_attack.npy"):
        """
        Save a single adversarial attack frame to disk.
        
        Args:
            frame: 29x29x1 numpy array to save
            output_file: Output filename (NPY format)
        """
        np.save(output_file, frame)
        print(f"Adversarial attack saved to {output_file}")

    def apply(self, cfg):
        """
        Main function that orchestrates the complete adversarial DoS attack generation and evaluation process.

        Process:
        1. Load pre-trained IDS model for RPM/spoofing attacks
        2. Generate adversarial DoS attack dataset (or load if exists)
        3. Evaluate model performance on adversarial examples
        4. Compute detailed performance metrics
        5. Create visualizations of misclassified examples
        6. Run parameter optimization experiments
        7. Generate performance analysis plots
        """
        dir_path     = cfg['dir_path']
        dataset_name = cfg['dataset_name']

        max_frames          = int(cfg.get('max_frames', 7000))
        dummy_row_threshold = int(cfg.get('dummy_row_threshold', 10))

        # Scoped to the input file AND frame count: the previous fixed filename was
        # overwritten by every run and could not distinguish datasets.
        attack_file = os.path.join(
            dir_path, "..", "datasets", dataset_name,
            f"adversarial_DoS_attack_{cfg['file_name'][:-4]}_n{max_frames}.npz"
        )
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(dir_path, "..", "datasets", dataset_name, "Results", "attack_results", f"DoS_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        final_test = y_test = x_test = frame_indices = None

        if os.path.exists(attack_file):
            print("Adversarial attack already exists. Loading from file...")
            try:
                data          = np.load(attack_file)
                final_test    = data['final_test']
                y_test        = data['y_test']
                x_test        = data['x_test']
                frame_indices = data['frame_indices'] if 'frame_indices' in data else None
            except Exception as e:
                print(f"Error loading {attack_file}: {e} — regenerating.")
                final_test = None

        if final_test is None:
            print(f"Generating new adversarial DoS attack dataset "
                  f"(max_frames={max_frames}, dummy_row_threshold={dummy_row_threshold})...")
            final_test, y_test, x_test, _, frame_indices = self.generate_adversarial_attack(
                dummy_row_threshold=dummy_row_threshold, max_frames=max_frames
            )
            np.savez(attack_file,
                     final_test=final_test,
                     y_test=y_test,
                     x_test=x_test,
                     frame_indices=frame_indices)
            print(f"Adversarial attack generated and saved to {attack_file}")

        if cfg.get('write_perturbed_csv', True):
            self.write_perturbed_traffic(cfg, final_test, y_test, frame_indices,
                                         results_dir, label='dos')

        print("  Evaluating model on adversarial dataset...")
        # ── ORIGINAL ─────────────────────────────────────────────────────────
        # test_loss, test_accuracy = self.model.evaluate(final_test, y_test, verbose=0)
        # ─────────────────────────────────────────────────────────────────────
        # CHANGED: GeneticAttack.evaluate wraps both frameworks — it forwards to
        # Model.evaluate(verbose=0) for .h5 targets (same call, same silence) and
        # computes sparse categorical cross-entropy + accuracy for .pth ones.
        test_loss, test_accuracy = self.evaluate(final_test, y_test)

        with open(os.path.join(results_dir, "adversarial_DoS_test.txt"), "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy:.4f}\n")

        # ── ORIGINAL ─────────────────────────────────────────────────────────
        # y_pred_prob = self.model.predict(final_test, verbose=0)
        # ─────────────────────────────────────────────────────────────────────
        # CHANGED: framework-agnostic scoring. For a .pth target this also
        # applies the softmax the bare nn.Linear head does not.
        y_pred_prob = self.predict_proba(final_test)
        y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions

        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()  # True Negative, False Positive, False Negative, True Positive

        FNR = round(FN / (TP + FN), 4) if (TP + FN) > 0 else 0.0        # False Negative Rate
        ER = round((FP + FN) / (TN + FP + FN + TP), 4)                   # Error Rate
        precision = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0  # Precision
        recall = round(TP / (TP + FN), 4) if (TP + FN) > 0 else 0.0     # Recall (Sensitivity)
        f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0  # F1 Score
        ASR = round(FN / (TP + FN), 4) if (TP + FN) > 0 else 0.0        # Attack Success Rate: adversarial attacks misclassified as Normal
        self.plot_confusion_matrix(cm, classes=['Normal', 'Attack'], suffix="adv_DoS_attack", normalize=False,
                            title='Confusion Matrix',
                            filename=os.path.join(results_dir, "confusion_matrix_adv_DoS_attack.png"))

        report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])

        with open(os.path.join(results_dir, "evaluation_metrics_DoS_adv.txt"), "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write(f"False Negative Rate (FNR): {FNR:.4f}\n")
            f.write(f"Error Rate (ER): {ER:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        print(f"\n  Accuracy       : {test_accuracy:.4f}")
        print(f"  Precision      : {precision:.4f}")
        print(f"  Recall         : {recall:.4f}")
        print(f"  F1 Score       : {f1:.4f}")
        print(f"  FNR            : {FNR:.4f}")
        print(f"  Error Rate     : {ER:.4f}")
        print(f"  Attack Success rate : {ASR:.4f}")
        print(f"\n{report}")

        cnt = 0
        for i in range(len(y_test)):
            if cnt == 5:  # Limit to 5 examples to avoid clutter
                break

            if y_test[i] != y_pred[i] and y_test[i] == 1:
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.imshow(x_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
                plt.title(f"Original DoS Attack (True: {y_test[i]})")

                plt.subplot(1, 2, 2)
                plt.imshow(final_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
                plt.title(f"Adversarial DoS Attack (Predicted: {y_pred[i]})")

                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"DoS_attack_comparison_{cnt}.png"))
                plt.close()
                cnt += 1

