#!/usr/bin/env python3
"""
Adversarial Spoofing Attack Generator using Genetic Algorithm

This script implements a genetic algorithm-based approach to generate adversarial attacks
against deep learning intrusion detection systems for vehicle CAN networks. The attack
focuses on RPM/Spoofing attack scenarios by modifying ECU-controlled dummy rows in CAN frames
to evade detection while maintaining attack characteristics.

Key characteristics of Spoofing attacks:
- Uses ECU control information to identify modifiable dummy rows
- Only modifies rows where ECU control value = 1 (transmitter controlled)
- Similar to DoS attacks but with ECU-specific constraints
- Preserves RPM attack payload in non-dummy rows

The genetic algorithm evolves adversarial examples through:
- Population-based search with crossover and mutation
- Fitness evaluation based on IDS confidence scores  
- Elitist selection to preserve best candidates
- Multi-generation evolution until successful evasion

Usage:
    python3 adversarial_spoofing_attack.py
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

class AdversarialSpoofAttack(GeneticAttack):
    def __init__(self, model_path, file_path, population_size=100, max_generations=75, mutation_rate=0.1):
        """
        Initialize the adversarial Spoofing attack generator.
        
        Args:
            model_path: Path to the trained IDS model (H5 format)
            population_size: Number of individuals in each genetic algorithm generation
            max_generations: Maximum number of generations to evolve before giving up
            mutation_rate: Probability of mutation occurring for each individual
        """

        super().__init__(model_path, file_path, population_size, max_generations, mutation_rate)
        self.original_dummy_rows = []
        self.ecu_control = self.data['ecu_control']  # ECU control values for each frame

        # ADDED: frame geometry read from the data instead of hardcoded, so the
        # same class handles the 29x29 CANShield frames and the 32x11 MULSAM
        # windows. AdversarialDosAttack already derives num_cols this way.
        self.num_rows = self.x_test.shape[1]   # packets per frame  (29 | 32)
        self.num_cols = self.x_test.shape[2]   # ID bits per packet (29 | 11)

    def find_dummy_rows(self, frame_idx):
        """
        Find and return indices of ECU-controlled dummy rows for a specific frame.
        
        For spoofing attacks, dummy rows are identified by ECU control values:
        - ECU control value = 0: Receiver controlled (cannot modify)
        - ECU control value = 1: Transmitter controlled (can modify)
        
        This is specific to RPM/spoofing attacks where ECU control information
        determines which parts of the CAN frame can be safely modified without
        destroying the attack payload.
        
        Args:
            frame_idx: Index of the frame in the dataset
            
        Returns:
            List of row indices where ECU control value = 1 (modifiable rows)
        """
        if frame_idx >= len(self.ecu_control):
            return []  # Return empty list if index is out of bounds

        # ── ORIGINAL ─────────────────────────────────────────────────────────
        # return [j for j in range(29) if self.ecu_control[frame_idx][j] == 1]
        # ─────────────────────────────────────────────────────────────────────
        # CHANGED: 29 was the CANShield frame height. A MULSAM window is 32
        # packets, so the hardcoded bound silently ignored rows 29-31 — the
        # attacker would never touch the last three messages of any frame.
        return [j for j in range(self.num_rows) if self.ecu_control[frame_idx][j] == 1]

    def mutate(self, frame):
        """
        Apply mutation to a frame by modifying only the original ECU-controlled dummy rows.
        
        Spoofing attack mutation strategy:
        - Only modifies rows that were originally identified as ECU-controlled (value=1)
        - For each dummy row, applies mutation with probability = mutation_rate
        - Sets exactly one random bit to 1 in the selected row
        - Unlike fuzzy attacks, can set any bit position (0-28) including priority bits
        
        This maintains the constraint that only ECU-transmitter controlled portions
        of the frame can be modified, preserving the RPM attack characteristics.
        
        Args:
            frame: 29x29x1 numpy array representing a CAN frame
            
        Returns:
            Mutated copy of the input frame
        """
        mutated = frame.copy()

        for i in self.original_dummy_rows:
            if random.random() < self.mutation_rate:
                # ── ORIGINAL ─────────────────────────────────────────────────
                # bit_to_flip = random.randint(0, 28)
                # ─────────────────────────────────────────────────────────────
                # CHANGED: 28 was the last column of a 29-bit CANShield row. An
                # 11-bit MULSAM row only has columns 0-10, so this raised
                # IndexError on the first mutation. Derived from the data now.
                bit_to_flip = random.randint(0, self.num_cols - 1)
                mutated[i, bit_to_flip, 0] = 1

        return mutated

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent frames to create offspring.
        
        Spoofing attack crossover strategy:
        - Start with parent1 as the base (child inherits most characteristics)
        - For each original ECU-controlled dummy row, randomly choose to inherit from parent2
        - Only swap rows that were originally ECU-controlled (preserves attack payload)
        - 50% chance per row to inherit from parent2
        
        This allows combining successful mutations from different parents
        while maintaining the integrity of the original RPM attack structure
        and ECU control constraints.
        
        Args:
            parent1: First parent frame (29x29x1 numpy array)
            parent2: Second parent frame (29x29x1 numpy array)
            
        Returns:
            Child frame combining characteristics from both parents
        """
        child = parent1.copy()
        
        for i in self.original_dummy_rows:
            if random.random() < 0.5:  # 50% chance to inherit from parent2
                child[i] = parent2[i]
                
        return child

    # ── ORIGINAL ─────────────────────────────────────────────────────────────
    # def calculate_confidence(self, frame):
    #     """
    #     Calculate the IDS confidence score for classifying a frame as an attack.
    #
    #     The IDS model outputs probabilities for [normal, attack] classes.
    #     We return the attack confidence (index 1) since our goal is to
    #     minimize this score below 0.5 to achieve misclassification.
    #
    #     Args:
    #         frame: 29x29x1 numpy array representing a CAN frame
    #
    #     Returns:
    #         Float between 0-1 representing attack confidence score
    #     """
    #     frame_batch = np.expand_dims(frame, 0)
    #
    #     prediction = self.model.predict(frame_batch, verbose=0)
    #
    #     return prediction[0][1]
    # ─────────────────────────────────────────────────────────────────────────
    # CHANGED: removed. This override was a byte-for-byte copy of
    # GeneticAttack.calculate_confidence except that it called self.model.predict
    # directly — a Keras-only call that shadowed the framework dispatch added to
    # the base class, so a .pth target would fail here. The inherited version
    # does the same thing through predict_proba and works for both frameworks.

    def _next_generation(self, population, scores):
        """
        Build the next population: elitist carry-over plus fitness-proportionate
        crossover and mutation.

        Fitness is 1 - attack_confidence, so lower-confidence (better evading)
        individuals are more likely to be selected as parents. If every
        individual scores 1.0 there is nothing to discriminate on, so selection
        falls back to uniform.

        Args:
            population: Current list of frames
            scores: Attack confidence per individual, same order as population

        Returns:
            New population of size self.population_size
        """
        inv_scores = 1.0 - scores
        total      = inv_scores.sum()

        if total <= 1e-12 or np.count_nonzero(inv_scores) < 2:
            selection_probs = np.ones_like(inv_scores) / len(inv_scores)
        else:
            selection_probs = inv_scores / total

        indices = np.arange(len(population))
        new_pop = [population[int(np.argmin(scores))]]  # elite

        while len(new_pop) < self.population_size:
            p1_idx, p2_idx = np.random.choice(
                indices, size=2, p=selection_probs, replace=False
            )
            child = self.crossover(population[p1_idx], population[p2_idx])
            new_pop.append(self.mutate(child))

        return new_pop

    def _evolve(self, frame, dummy_rows):
        """
        Run the genetic search on a single frame until evasion or generation cap.

        Shared by generate_adversarial_attack and both parameter experiments so
        the three call sites cannot drift apart.

        Deliberately silent: callers log one line per frame instead. Printing per
        generation puts tens of thousands of lines in the log for a normal run.

        Args:
            frame: 29x29x1 numpy array to perturb
            dummy_rows: Row indices the mutation/crossover operators may touch

        Returns:
            tuple: (best_frame, generations_used, best_score, evaded)
        """
        self.original_dummy_rows = list(dummy_rows)

        mut_rate = self.mutation_rate
        self.mutation_rate = 1  # Force mutation so the initial population is diverse
        population = [self.mutate(frame.copy()) for _ in range(self.population_size)]
        self.mutation_rate = mut_rate

        for generation in range(self.max_generations):
            scores = np.nan_to_num(self.calculate_confidence_batch(population), nan=1.0)

            success_idx = np.where(scores < 0.5)[0]
            if len(success_idx) > 0:
                winner = int(success_idx[0])
                return population[winner], generation + 1, float(scores[winner]), True

            population = self._next_generation(population, scores)

        # The loop replaced `population` on its final iteration, so the last
        # computed `scores` belong to the previous generation — score the
        # surviving population instead of indexing it with stale fitness.
        scores   = np.nan_to_num(self.calculate_confidence_batch(population), nan=1.0)
        best_idx = int(np.argmin(scores))

        return population[best_idx], self.max_generations, float(scores[best_idx]), False

    def _suitable_attack_frames(self, threshold):
        """
        Indices of attack frames with at least `threshold` ECU-controlled rows.

        Args:
            threshold: Minimum number of modifiable rows

        Returns:
            numpy array of frame indices
        """
        n      = min(len(self.x_test), len(self.ecu_control))
        counts = (self.ecu_control[:n] == 1).sum(axis=1)

        return np.where((self.y_test[:n] == 1) & (counts >= threshold))[0]

    def generate_adversarial_attack(self, dummy_row_threshold=1, max_frames=100):
        """
        Main genetic algorithm to generate adversarial Spoofing attacks.

        Process:
        1. Create balanced dataset (70% attack, 30% benign)
        2. Add benign frames directly (no modification needed)
        3. For each attack frame with sufficient ECU-controlled dummy rows:
           - Identify ECU-controlled dummy rows using ECU control data
           - Initialize genetic algorithm population
           - Evolve through generations using crossover/mutation
           - Stop when successful evasion achieved (confidence < 0.5)
        4. Return complete adversarial dataset with labels
        
        Args:
            dummy_row_threshold: Minimum ECU-controlled rows required to process a frame
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
        attack_count = int(max_frames * 0.7)
        benign_count = max_frames - attack_count
        
        print(f"Using {attack_count} attack frames and {benign_count} benign frames")
        
        benign_indices = np.where(self.y_test == 0)[0][:benign_count]
        if len(benign_indices) < benign_count:
            benign_count = len(benign_indices)
            print(f"Warning: Only {benign_count} benign frames available")
        
        attack_indices = np.where(self.y_test == 1)[0]
        if len(attack_indices) < attack_count:
            attack_count = len(attack_indices)
            print(f"Warning: Only {attack_count} attack frames available")
        
        orig_frame = []          # Original frames for comparison
        final_test = []          # Adversarial/benign frames for evaluation
        generations_needed = []  # Track genetic algorithm performance
        frame_indices = []       # Source index in x_test of each returned frame

        for i in benign_indices:
            final_test.append(self.x_test[i])
            orig_frame.append(self.x_test[i])
            frame_indices.append(int(i))


        print("Starting genetic algorithm for adversarial attack generation...")
        
        attack_frames_processed = 0
        for i in attack_indices:
            if attack_frames_processed >= attack_count:
                break
                
            dummy_rows = self.find_dummy_rows(i)

            if len(dummy_rows) < dummy_row_threshold:
                continue  # Not enough modifiable rows — skip this frame entirely

            best, generations, best_score, evaded = self._evolve(self.x_test[i], dummy_rows)

            # One line per frame — flushed so `tail -f` tracks a redirected run.
            outcome = (f"evaded in generation {generations}" if evaded
                       else f"NOT evaded in {generations} generations, best score {best_score:.4f}")
            print(f"Attack frame {attack_frames_processed+1}/{attack_count} "
                  f"(frame {i}): {outcome}", flush=True)

            final_test.append(best)
            orig_frame.append(self.x_test[i])
            generations_needed.append(generations)
            frame_indices.append(int(i))
            attack_frames_processed += 1

        y_final = np.zeros(len(final_test))
        y_final[benign_count:] = 1  # Attack samples start after benign samples

        return (np.array(final_test), y_final, np.array(orig_frame),
                generations_needed, np.array(frame_indices))

    def run_dummy_row_experiment(self, max_frames=20, mutation_rate=0.3,
                                 thresholds=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20)):
        """
        Test the effect of the ECU-controlled dummy row threshold on attack success.

        Tests different minimum ECU-controlled row requirements and measures
        performance. Higher thresholds mean:
        - More ECU-controlled modification space available (easier attacks)
        - Fewer eligible frames (reduced dataset size)

        Runs against this instance's already-loaded model and dataset — building a
        fresh AdversarialSpoofAttack per threshold would reload the model and the
        full test npz ten times over.

        Args:
            max_frames: Number of frames to test per threshold
            mutation_rate: Fixed mutation rate, held constant for fair comparison
            thresholds: Dummy row thresholds to sweep

        Returns:
            Dictionary mapping dummy row thresholds to average generations needed
        """
        results    = {}
        saved_rate = self.mutation_rate
        self.mutation_rate = mutation_rate

        try:
            for threshold in thresholds:
                print(f"\n--- Testing dummy row threshold: {threshold} ---")

                suitable_frames = self._suitable_attack_frames(threshold)
                print(f"Found {len(suitable_frames)} frames with dummy rows >= {threshold}")

                if len(suitable_frames) == 0:
                    results[threshold] = 0
                    print(f"Dummy row threshold {threshold}: No suitable frames found")
                    continue

                suitable_frames = suitable_frames[:max_frames]

                generations_list = []
                for idx, frame_idx in enumerate(suitable_frames):
                    print(f"Processing frame {idx+1}/{len(suitable_frames)}")
                    _, generations, _, _ = self._evolve(
                        self.x_test[frame_idx], self.find_dummy_rows(frame_idx)
                    )
                    generations_list.append(generations)

                results[threshold] = float(np.mean(generations_list))
                print(f"Dummy row threshold {threshold}: "
                      f"Average generations = {results[threshold]:.2f}")
        finally:
            self.mutation_rate = saved_rate

        return results

    def run_mutation_rate_experiment(self, max_frames=20, dummy_row_threshold=1,
                                     mutation_rates=(0.1, 0.2, 0.3, 0.4, 0.5)):
        """
        Test the effect of the mutation rate on attack success.

        Holds the frame set fixed and sweeps the mutation probability, so the
        resulting curve isolates mutation pressure from frame difficulty.

        Args:
            max_frames: Number of frames to test per mutation rate
            dummy_row_threshold: Minimum ECU-controlled rows a frame must have
            mutation_rates: Mutation rates to sweep

        Returns:
            Dictionary mapping mutation rate to average generations needed
        """
        suitable_frames = self._suitable_attack_frames(dummy_row_threshold)[:max_frames]
        print(f"Found {len(suitable_frames)} attack frames with dummy rows "
              f">= {dummy_row_threshold}")

        if len(suitable_frames) == 0:
            print("No suitable frames found — skipping mutation rate experiment")
            return {}

        results    = {}
        saved_rate = self.mutation_rate

        try:
            for rate in mutation_rates:
                print(f"\n--- Testing mutation rate: {rate} ---")
                self.mutation_rate = rate

                generations_list = []
                for idx, frame_idx in enumerate(suitable_frames):
                    print(f"Processing frame {idx+1}/{len(suitable_frames)}")
                    _, generations, _, _ = self._evolve(
                        self.x_test[frame_idx], self.find_dummy_rows(frame_idx)
                    )
                    generations_list.append(generations)

                results[rate] = float(np.mean(generations_list))
                print(f"Mutation rate {rate}: Average generations = {results[rate]:.2f}")
        finally:
            self.mutation_rate = saved_rate

        return results

    def plot_dummy_row_results(self, results, filename='spoofing_dummy_rows_vs_generations.png'):
        """
        Create line plot showing the relationship between ECU-controlled dummy row threshold and attack performance.

        Shows how the amount of ECU-controlled modification space affects attack difficulty.

        Args:
            results: Dictionary mapping dummy row thresholds to average generations needed
            filename: Output path for the PNG
        """
        thresholds = sorted(results)
        avgs = [results[t] for t in thresholds]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, avgs, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Dummy Row Threshold')
        plt.ylabel('Average Number of Generations')
        plt.title('Effect of Dummy Row Threshold on Spoofing Adversarial Attack Generations')
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.savefig(filename)
        plt.close()

        print(f"Dummy row experiment plot saved as '{filename}'")

    def apply(self, cfg):
        """
        Main function that orchestrates the complete adversarial spoofing attack generation and evaluation process.

        Process:
        1. Load pre-trained IDS model for RPM/spoofing attacks
        2. Generate adversarial spoofing attack dataset (or load if exists)
        3. Evaluate model performance on adversarial examples
        4. Compute detailed performance metrics
        5. Create visualizations of misclassified examples
        6. Run parameter optimization experiments
        7. Generate performance analysis plots
        """
        dir_path     = cfg['dir_path']
        dataset_name = cfg['dataset_name']

        max_frames          = int(cfg.get('max_frames', 100))
        dummy_row_threshold = int(cfg.get('dummy_row_threshold', 1))

        # Scoped to the input file AND frame count: a single fixed filename would
        # silently serve a cached result generated from a different dataset or a
        # different max_frames.
        attack_file = os.path.join(
            dir_path, "..", "datasets", dataset_name,
            f"adversarial_spoofing_attack_{cfg['file_name'][:-4]}_n{max_frames}.npz"
        )
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(dir_path, "..", "datasets", dataset_name, "Results", "attack_results", f"Spoof_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # This method runs on an instance the handler already built with the
        # configured model, dataset and GA hyper-parameters — reuse it rather than
        # constructing a second attack against a different model.
        final_test = y_test = x_test = frame_indices = None

        if os.path.exists(attack_file):
            print("Adversarial attack already exists. Loading from file...")
            try:
                data          = np.load(attack_file)
                final_test    = data['final_test']  # Adversarial/benign frames
                y_test        = data['y_test']      # True labels
                x_test        = data['x_test']      # Original frames
                frame_indices = data['frame_indices'] if 'frame_indices' in data else None
            except Exception as e:
                print(f"Error loading {attack_file}: {e} — regenerating.")
                final_test = None

        if final_test is None:
            print(f"Generating new adversarial spoofing attack dataset "
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
                                        results_dir, label='spoof')

        print("Evaluating model performance on adversarial spoofing attack dataset...")
        # ── ORIGINAL ─────────────────────────────────────────────────────────
        # # verbose=0: the Keras progress bar writes ANSI escapes and \r, which turn a
        # # redirected log into unreadable single-line noise.
        # test_loss, test_accuracy = self.model.evaluate(final_test, y_test, verbose=0)
        # ─────────────────────────────────────────────────────────────────────
        # CHANGED: GeneticAttack.evaluate wraps both frameworks — it forwards to
        # Model.evaluate(verbose=0) for .h5 targets (same call, same silence) and
        # computes sparse categorical cross-entropy + accuracy for .pth ones.
        test_loss, test_accuracy = self.evaluate(final_test, y_test)
        print(f"  Test loss {test_loss:.4f}, accuracy {test_accuracy:.4f}")

        with open(os.path.join(results_dir, "adversarial_spoof_test.txt"), "w") as f:
            f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy:.4f}\n")

        print("Computing detailed performance metrics...")

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

        # Raw counts, not normalized fractions, so the figure reads the same as the
        # matrix written into evaluation_metrics_spoof_adv.txt.
        self.plot_confusion_matrix(cm, classes=['Normal', 'Attack'], suffix="adv_spoof_attack", normalize=False,
                            title='Confusion Matrix',
                            filename=os.path.join(results_dir, "confusion_matrix_adv_spoof_attack.png"))

        report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])

        with open(os.path.join(results_dir, "evaluation_metrics_spoof_adv.txt"), "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write(f"False Negative Rate (FNR): {FNR:.4f}\n")
            f.write(f"Error Rate (ER): {ER:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        print(f"Results for spoof adversarial attack (test set) saved")

        print("Creating visualizations of misclassified spoofing attack examples...")

        cnt = 0
        for i in range(len(y_test)):
            if cnt == 5:  # Limit to 5 examples to avoid clutter
                break

            if y_test[i] != y_pred[i] and y_test[i] == 1:
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.imshow(x_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
                plt.title(f"Original Spoof Attack (True: {y_test[i]})")

                plt.subplot(1, 2, 2)
                plt.imshow(final_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
                plt.title(f"Adversarial Spoof Attack (Predicted: {y_pred[i]})")

                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"spoof_attack_comparison_{cnt}.png"))
                plt.close()
                print(f"Attack comparison saved as spoof_attack_comparison_{cnt}.png")
                cnt += 1
        
        # Both sweeps re-run the full GA per frame per setting, so they cost far
        # more than the attack itself — opt in via GeneticAdvAttack.run_experiments.
        if not cfg.get('run_experiments', False):
            print("\nSkipping parameter experiments "
                  "(set GeneticAdvAttack.run_experiments: true to enable).")
            return

        print("\n===== RUNNING MUTATION RATE EXPERIMENT =====")
        mutation_results = self.run_mutation_rate_experiment(max_frames=20)
        if mutation_results:
            self.plot_mutation_rate_results(
                mutation_results,
                filename=os.path.join(results_dir, "spoofing_mutation_rate_vs_generations.png"),
            )

        print("\n===== RUNNING DUMMY ROW THRESHOLD EXPERIMENT =====")
        dummy_row_results = self.run_dummy_row_experiment(max_frames=20)
        self.plot_dummy_row_results(
            dummy_row_results,
            filename=os.path.join(results_dir, "spoofing_dummy_rows_vs_generations.png"),
        )

