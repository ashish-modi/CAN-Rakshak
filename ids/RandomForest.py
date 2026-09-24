from common_imports import accuracy_score, joblib, np
from sklearn.ensemble import RandomForestClassifier
from ids.base import IDS
from ids.frame_features import FEATURE_NAMES, csv_path, load_frames


class RandomForest(IDS):
    """Random-forest classifier over per-frame (CAN ID, 8 payload bytes)."""

    N_ESTIMATORS = 100
    MAX_DEPTH = 4
    RANDOM_STATE = 0

    def __init__(self):
        self.rf = RandomForestClassifier(
            n_estimators=self.N_ESTIMATORS,
            max_depth=self.MAX_DEPTH,
            random_state=self.RANDOM_STATE,
            n_jobs=-1,
        )


    def train(self, train_dataset_dir=None, X_train=None, Y_train=None,
              cfg=None, **kwargs):
        """Fit on every frame of modified_dataset/<file_name>.

        `train_dataset_dir` (the feature-split folder Stage 4 passes every IDS)
        is unused: this classifier reads raw frames, not extracted features.
        """
        cfg = cfg or {}
        if X_train is None or Y_train is None:
            path = csv_path(cfg)
            print(f"RandomForest — loading training data from {path}")
            X_train, Y_train = load_frames(path)

        print(f"  frames: {len(X_train)}  attack: {int(Y_train.sum())}  "
              f"benign: {int((Y_train == 0).sum())}")
        if len(np.unique(Y_train)) < 2:
            raise ValueError(
                "[RandomForest] training data has a single class -- a "
                "supervised classifier needs both benign and attack frames."
            )

        print(f"Training RandomForest "
              f"({self.N_ESTIMATORS} trees, max_depth={self.MAX_DEPTH})")
        self.rf.fit(X_train, Y_train)

        # Which features the forest actually used. On a capture where the
        # attack uses its own CAN ID this is dominated by can_id, which is the
        # signal that the task was trivially separable (see module docstring).
        order = np.argsort(self.rf.feature_importances_)[::-1]
        print("  feature importances:")
        for i in order:
            if self.rf.feature_importances_[i] > 0:
                print(f"    {FEATURE_NAMES[i]:<7} "
                      f"{self.rf.feature_importances_[i]:.4f}")
        print(f"  training accuracy: "
              f"{accuracy_score(Y_train, self.rf.predict(X_train)):.4f}")

    
    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs):
        """Score modified_dataset/<file_name> -> (preds, labels), per frame."""
        cfg = cfg or {}
        if X_test is None or Y_test is None:
            path = csv_path(cfg)
            print(f"RandomForest — loading test data from {path}")
            X_test, Y_test = load_frames(path)

        print(f"  frames: {len(X_test)}  attack: {int(Y_test.sum())}  "
              f"benign: {int((Y_test == 0).sum())}")
        preds = self.predict(X_test)
        return preds, Y_test

    def predict(self, X_test=None, **kwargs):
        if X_test is None:
            raise ValueError("[RandomForest] predict() needs an input")
        return self.rf.predict(np.asarray(X_test))

    def save(self, path):
        joblib.dump(self.rf, path)
        print(f"  RandomForest model saved to {path}")

    def load(self, path):
        self.rf = joblib.load(path)
        print(f"  RandomForest model loaded from {path}")

    def extract_features(self):
        pass
