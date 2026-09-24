from common_imports import (
    os, np, logging, joblib,
    accuracy_score, StandardScaler,
    Dense, Input, Sequential, SparseCategoricalCrossentropy, EarlyStopping,
)
from ids.base import IDS
from ids.frame_features import FEATURE_NAMES, csv_path, load_frames
import absl.logging
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("absl").setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)


class MLP(IDS):
    """Feed-forward classifier over per-frame (CAN ID, 8 payload bytes)."""

    HIDDEN_UNITS = 128
    NUM_CLASSES = 2
    BATCH_SIZE = 8192
    PATIENCE = 5
    VALIDATION_SPLIT = 0.2
    SEED = 0

    def __init__(self):
        self.scaler = None
        self.mlp = Sequential()
        self.mlp.add(Input(shape=(len(FEATURE_NAMES),)))
        self.mlp.add(Dense(self.HIDDEN_UNITS, activation='relu'))
        self.mlp.add(Dense(self.HIDDEN_UNITS, activation='relu'))
        self.mlp.add(Dense(self.NUM_CLASSES, activation='softmax'))

    
    def train(self, train_dataset_dir=None, X_train=None, Y_train=None,
              cfg=None, **kwargs):
        """Fit on every frame of modified_dataset/<file_name>.

        `train_dataset_dir` (the feature-split folder Stage 4 passes every IDS)
        is unused: this classifier reads raw frames, not extracted features.
        """
        cfg = cfg or {}
        if X_train is None or Y_train is None:
            path = csv_path(cfg)
            print(f"MLP — loading training data from {path}")
            X_train, Y_train = load_frames(path)

        print(f"  frames: {len(X_train)}  attack: {int(Y_train.sum())}  "
              f"benign: {int((Y_train == 0).sum())}")
        if len(np.unique(Y_train)) < 2:
            raise ValueError(
                "[MLP] training data has a single class -- a supervised "
                "classifier needs both benign and attack frames."
            )

        # The scaler is fitted on training data only and persisted with the
        # weights: test frames must be transformed by the SAME statistics, or
        # the net sees inputs on a scale it never trained on.
        self.scaler = StandardScaler().fit(X_train)
        X = self.scaler.transform(X_train).astype("float32")
        Y = np.asarray(Y_train).astype("int32")

        
        tf.random.set_seed(self.SEED)

        self.mlp.compile(
            optimizer='adam',
            loss=SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'],
        )
        self.es = EarlyStopping(monitor='val_loss', patience=self.PATIENCE,
                                restore_best_weights=True)

        epochs = cfg.get('epochs', 10) or 10
        print(f"Training MLP ({epochs} epochs, batch={self.BATCH_SIZE})")
        self.mlp_hist = self.mlp.fit(
            X, Y,
            epochs=epochs,
            validation_split=self.VALIDATION_SPLIT,
            callbacks=[self.es],
            batch_size=self.BATCH_SIZE,
        )
        print(f"  training accuracy: "
              f"{accuracy_score(Y, self.predict(X_train)):.4f}")

    # ------------------------------------------------------------------
    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs):
        """Score modified_dataset/<file_name> -> (preds, labels), per frame."""
        cfg = cfg or {}
        if X_test is None or Y_test is None:
            path = csv_path(cfg)
            print(f"MLP — loading test data from {path}")
            X_test, Y_test = load_frames(path)

        print(f"  frames: {len(X_test)}  attack: {int(Y_test.sum())}  "
              f"benign: {int((Y_test == 0).sum())}")
        preds = self.predict(X_test)
        return preds, np.asarray(Y_test).astype("int32")

    def predict(self, X_test=None, **kwargs):
        """Class indices for raw (unscaled) per-frame features."""
        if X_test is None:
            raise ValueError("[MLP] predict() needs an input")
        X = np.asarray(X_test)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.mlp.predict(X.astype("float32"),
                                batch_size=self.BATCH_SIZE,
                                verbose=0).argmax(axis=1)

    def save(self, path):
        # Weights and scaler go in one file: the driver hands every IDS a
        # single model path, and a net restored without its scaler would score
        # standardised-scale inputs against raw byte values.
        joblib.dump({"mlp": self.mlp, "scaler": self.scaler}, path)
        print(f"  MLP model saved to {path}")

    def load(self, path):
        state = joblib.load(path)
        if isinstance(state, dict):
            self.mlp = state["mlp"]
            self.scaler = state.get("scaler")
        else:            # a bare model saved before the scaler was persisted
            self.mlp = state
            self.scaler = None
            print("  [MLP] WARNING: checkpoint has no scaler; inputs will be "
                  "fed unscaled and predictions are unreliable")
        print(f"  MLP model loaded from {path}")

    def extract_features(self):
        pass
