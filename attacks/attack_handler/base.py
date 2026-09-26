import inspect
import os

# ── ORIGINAL ─────────────────────────────────────────────────────────────────
# from common_imports import abc, np, pd, load_model, plt, itertools, tf
# ─────────────────────────────────────────────────────────────────────────────
# CHANGED: torch is needed to target PyTorch checkpoints (MULSAM,
# models/mulsam_*_target.pth) with the genetic attacks. common_imports already
# guards the import behind TORCH_AVAILABLE, so this adds no hard dependency for
# the Keras-only paths.
from common_imports import abc, np, pd, load_model, plt, itertools, tf, torch

# Keras 3 refuses to deserialize a `Lambda` layer wrapping a Python lambda unless
# safe_mode is off, and every .h5 in models/ contains one. Keras 2 has no such
# restriction and no such keyword, so only pass it where it exists — the repo has
# to run under both (can_rakshak venv ships Keras 2.10).
_LOAD_MODEL_KWARGS = (
    {'safe_mode': False}
    if 'safe_mode' in inspect.signature(load_model).parameters
    else {}
)


class Attack(abc.ABC):


    @abc.abstractmethod
    def apply(self, **kwargs):
        """
        Core attack logic. The meaning of parameters is up to subclass: 
        could be frames, dataframes, model, training data, etc.
        """
        pass


class EvasionAttack(Attack):
    @abc.abstractmethod
    def apply(self, frames: list[dict], labels: np.ndarray | None = None, **kwargs) -> list[dict]:
        """
        Perturb actual CAN frames or features to cause model misclassification.
        """

class GeneticAttack(Attack, abc.ABC):

    # Input geometry of a PyTorch MULSAM target: 32-packet windows of 11-bit
    # arbitration IDs. Only used to build the network before load_state_dict —
    # the frames themselves come from the .npz.
    TORCH_INPUT_SIZE  = 11
    TORCH_HIDDEN_SIZE = 24
    TORCH_NUM_CLASSES = 2

    # Chunk size for torch inference. The GA scores one population (100) at a
    # time, but apply() scores the whole adversarial set in one call, which can
    # be thousands of frames — chunking keeps that off the GPU memory ceiling.
    TORCH_BATCH_SIZE = 1024

    def __init__(self, model_path, file_path, population_size=100, max_generations=75, mutation_rate=0.1):

        # ── ORIGINAL (Keras only) ────────────────────────────────────────────
        # self.model = load_model(model_path, compile=False, **_LOAD_MODEL_KWARGS)
        #
        # self.model.compile(
        #     optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
        #     loss='sparse_categorical_crossentropy',
        #     metrics=['accuracy']
        # )
        # ─────────────────────────────────────────────────────────────────────
        # CHANGED: the genetic attacks now also target PyTorch checkpoints,
        # which load_model cannot read. Dispatch on the file extension so every
        # existing .h5 attack keeps exactly the behaviour above.
        self.framework = 'torch' if str(model_path).endswith(('.pth', '.pt')) else 'keras'

        if self.framework == 'torch':
            # Imported here, not at module scope: attacks/attack_handler/__init__
            # imports this module eagerly, and ids.mulsam pulls in torch +
            # torchvision, which the Keras-only runs should not have to load.
            # MULSAMNet carries the same submodule names (attention, lstm_time,
            # lstm_depth, fc) as the class the .pth was saved from, so the
            # state_dict loads without any key remapping.
            from ids.mulsam import MULSAMNet

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model  = MULSAMNet(
                input_size  = self.TORCH_INPUT_SIZE,
                hidden_size = self.TORCH_HIDDEN_SIZE,
                num_classes = self.TORCH_NUM_CLASSES,
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()   # the target is a frozen black box — never trained here

            print(f"  Target model   : PyTorch MULSAM on {self.device}")
        else:
            self.device = None
            self.model  = load_model(model_path, compile=False, **_LOAD_MODEL_KWARGS)

            self.model.compile(
                optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

        self.data = np.load(file_path)
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.x_test = self.data['x_test']
        self.y_test = self.data['y_test']  
        

    def predict_proba(self, frames):
        """
        Class probabilities for a batch of frames, whichever framework backs the
        target model.

        Added so the attacks have one scoring entry point instead of calling
        self.model.predict directly — a Keras-only call that a .pth target
        cannot answer.

        Args:
            frames: (N, rows, bits, 1) array of frames

        Returns:
            (N, num_classes) array of probabilities
        """
        batch = np.asarray(frames, dtype=np.float32)

        if self.framework == 'keras':
            return self.model.predict(batch, verbose=0)

        # MULSAMNet.forward runs its own unsqueeze(1) before the Conv2d
        # attention block, so it expects (N, 32, 11). Feeding it the trailing
        # channel axis the frames are stored with would make that a 5-D tensor
        # and Conv2d would reject it. The channel is dropped HERE and nowhere
        # else, so mutate/crossover/decode.py keep indexing [row, col, 0].
        if batch.ndim == 4:
            batch = batch[..., 0]

        probs = []
        with torch.no_grad():
            for lo in range(0, len(batch), self.TORCH_BATCH_SIZE):
                chunk  = torch.from_numpy(batch[lo:lo + self.TORCH_BATCH_SIZE]).to(self.device)
                # The net returns raw logits (a bare nn.Linear), so softmax is
                # ours to apply — Keras models here already end in softmax.
                probs.append(torch.softmax(self.model(chunk), dim=1).cpu().numpy())

        return np.concatenate(probs, axis=0)

    def evaluate(self, frames, labels):
        """
        (loss, accuracy) for a batch, mirroring Keras Model.evaluate.

        Added so apply() does not have to branch on the framework. The torch
        branch computes sparse categorical cross-entropy, matching the loss the
        Keras path is compiled with above.

        Args:
            frames: (N, rows, bits, 1) array of frames
            labels: (N,) integer class labels

        Returns:
            tuple: (loss, accuracy)
        """
        if self.framework == 'keras':
            return self.model.evaluate(frames, labels, verbose=0)

        probs = self.predict_proba(frames)
        y     = np.asarray(labels).astype(int)

        # Clipped before the log: a confidently wrong prediction can round to
        # exactly 0 in float32 and turn the mean into inf.
        picked = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)

        return float(-np.log(picked).mean()), float((probs.argmax(axis=1) == y).mean())

    def calculate_confidence(self, frame):

        # ── ORIGINAL (Keras only) ────────────────────────────────────────────
        # frame_batch = np.expand_dims(frame, 0)
        # prediction = self.model.predict(frame_batch, verbose=0)
        #
        # return prediction[0][1]
        # ─────────────────────────────────────────────────────────────────────
        # CHANGED: routed through predict_proba so a .pth target is scored the
        # same way as a .h5 one.
        frame_batch = np.expand_dims(frame, 0)
        prediction = self.predict_proba(frame_batch)

        return prediction[0][1]

    def calculate_confidence_batch(self, population):
        """Predict confidence scores for an entire population in one call."""
        # ── ORIGINAL (Keras only) ────────────────────────────────────────────
        # batch = np.array(population)
        # predictions = self.model.predict(batch, verbose=0)
        # return predictions[:, 1]
        # ─────────────────────────────────────────────────────────────────────
        # CHANGED: routed through predict_proba. This is the GA's inner loop —
        # called once per generation per frame — so it stays a single batched
        # call, no per-individual Python loop.
        predictions = self.predict_proba(population)
        return predictions[:, 1]


    def plot_confusion_matrix(self, cm, classes, suffix, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, filename=None):
        """
        Create and save a confusion matrix visualization with color coding and annotations.
        
        Args:
            cm: 2x2 confusion matrix array
            classes: List of class names ['Normal', 'Attack']
            suffix: String identifier for filename generation
            normalize: Whether to show percentages instead of raw counts
            title: Plot title
            cmap: Matplotlib colormap for visualization
            filename: Optional custom filename (auto-generated if None)
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'  # Format: decimals for percentages, integers for counts
        thresh = cm.max() / 2.  # Threshold for text color (white on dark, black on light)
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if filename is None:
            filename = f"confusion_matrix_{suffix}.png"
        plt.savefig(filename)
        plt.close()

    def write_perturbed_traffic(self, cfg, final_test, y_test, frame_indices, results_dir,
                                label):
        """
        Decode perturbed frames back into a CAN CSV usable by other IDSs.

        Shared by every genetic attack mode — the frame encoding and the
        GeneticExtractor provenance are identical across dos / fuzzy / spoof, so
        only the output filename differs.

        Writes into modified_dataset/ so the result can be fed straight back through
        the pipeline by pointing file_name at it, plus a copy of the frame labels
        next to the run's other results.

        perturbed_csv_mode selects the output shape:
          full   — the whole source trace with perturbed IDs spliced in, same line
                   count as the input, surrounding traffic kept as context
          subset — only the frames the attack evaluated, matching the .npz and the
                   reported metrics one-to-one

        Args:
            cfg: Attack config dict
            final_test: Perturbed/benign frames
            y_test: Per-frame labels aligned with final_test
            frame_indices: Source frame index per entry of final_test
            results_dir: This run's results directory
            label: Attack tag used in the output filename, e.g. "spoof" or "dos"
        """
        # Imported here rather than at module scope: attacks/attack_handler/__init__
        # eagerly imports this module, and a top-level import of a sibling attack
        # package would run during that partially-initialised state.
        from attacks.Genetic_algorithm.decode import write_perturbed_csv, write_perturbed_trace

        if frame_indices is None:
            print("Cannot write perturbed CSV: this adversarial .npz predates frame_indices. "
                  "Delete it and re-run to regenerate.")
            return

        if 'row_index' not in self.data:
            print("Cannot write perturbed CSV: the attack input .npz has no row_index "
                  "provenance. Re-run Stage 1 with feature_extractor: GeneticExtractor.")
            return

        mode = cfg.get('perturbed_csv_mode', 'full').lower()
        if mode not in ('full', 'subset'):
            raise ValueError(f"perturbed_csv_mode must be 'full' or 'subset', got {mode!r}")

        stem       = cfg['file_name'][:-4]
        out_csv    = os.path.join(cfg['dir_path'], "..", "datasets", cfg['dataset_name'],
                                  "modified_dataset", f"{stem}_adv_{label}.csv")
        labels_csv = os.path.join(results_dir, f"{stem}_adv_{label}_labels.csv")

        print(f"Decoding perturbed frames back to CAN traffic ({mode} trace)...")

        common = dict(
            frames=final_test,
            frame_indices=frame_indices,
            row_index=self.data['row_index'],
            source_csv=str(self.data['source_csv']),
            output_csv=out_csv,
            labels_csv=labels_csv,
        )

        if mode == 'full':
            # Labels cover every frame in the trace, not just the attacked ones.
            stats = write_perturbed_trace(labels=self.y_test, **common)
        else:
            stats = write_perturbed_csv(labels=y_test, **common)

        print(f"  Perturbed CSV  : {stats['output_csv']}")
        print(f"  Frame labels   : {stats['labels_csv']}")
        print(f"  Lines          : {stats['rows']} "
              f"({stats['changed_ids']} CAN IDs rewritten across "
              f"{stats['frames']} attacked frames)")
        print(f"  Next           : set file_name: {os.path.basename(out_csv)} to run "
              f"another IDS over it")

    def plot_mutation_rate_results(self, results, filename="mutation_rate_vs_generations.png"):
        """
        Plot average generations-to-evasion against mutation rate.

        Args:
            results: Dictionary mapping mutation rate to average generations needed
            filename: Output path for the PNG
        """
        rates = sorted(results)
        avgs  = [results[r] for r in rates]

        plt.figure(figsize=(10, 6))
        plt.plot(rates, avgs, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Mutation Rate')
        plt.ylabel('Average Number of Generations')
        plt.title('Effect of Mutation Rate on Adversarial Attack Generations')
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.savefig(filename)
        plt.close()

        print(f"Mutation rate experiment plot saved as '{filename}'")

    @abc.abstractmethod
    def mutate(self, frame):
        raise NotImplementedError

    @abc.abstractmethod
    def crossover(self, parent1, parent2):
        raise NotImplementedError

    @abc.abstractmethod
    def generate_adversarial_attack(self):
        raise NotImplementedError

