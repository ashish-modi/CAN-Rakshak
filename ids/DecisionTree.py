from common_imports import accuracy_score, joblib, np
from sklearn.tree import DecisionTreeClassifier, export_text
from ids.base import IDS
from ids.frame_features import FEATURE_NAMES, csv_path, load_frames


class DecisionTree(IDS):
    """Decision-tree classifier over per-frame (CAN ID, 8 payload bytes)."""

    MAX_DEPTH = 4
    RANDOM_STATE = 0

    def __init__(self):
        self.dt = DecisionTreeClassifier(max_depth=self.MAX_DEPTH,
                                          random_state=self.RANDOM_STATE)

    # ------------------------------------------------------------------
    def train(self, train_dataset_dir=None, X_train=None, Y_train=None,
              cfg=None, **kwargs):
        
        cfg = cfg or {}
        if X_train is None or Y_train is None:
            path = csv_path(cfg)
            print(f"DecisionTree — loading training data from {path}")
            X_train, Y_train = load_frames(path)

        print(f"  frames: {len(X_train)}  attack: {int(Y_train.sum())}  "
              f"benign: {int((Y_train == 0).sum())}")
        if len(np.unique(Y_train)) < 2:
            raise ValueError(
                "[DecisionTree] training data has a single class -- a "
                "supervised classifier needs both benign and attack frames."
            )

        print(f"Training DecisionTree (max_depth={self.MAX_DEPTH})")
        self.dt.fit(X_train, Y_train)

        print(f"  depth reached: {self.dt.get_depth()}  "
              f"leaves: {self.dt.get_n_leaves()}")
        order = np.argsort(self.dt.feature_importances_)[::-1]
        print("  feature importances:")
        for i in order:
            if self.dt.feature_importances_[i] > 0:
                print(f"    {FEATURE_NAMES[i]:<7} "
                      f"{self.dt.feature_importances_[i]:.4f}")

        # The learned rule itself. A one-split tree here means the label was a
        # deterministic function of a single feature (see module docstring).
        print("  learned rule:")
        for line in export_text(self.dt, feature_names=FEATURE_NAMES,
                                 max_depth=self.MAX_DEPTH).splitlines():
            print(f"    {line}")
        print(f"  training accuracy: "
              f"{accuracy_score(Y_train, self.dt.predict(X_train)):.4f}")

    
    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs):
        """Score modified_dataset/<file_name> -> (preds, labels), per frame."""
        cfg = cfg or {}
        if X_test is None or Y_test is None:
            path = csv_path(cfg)
            print(f"DecisionTree — loading test data from {path}")
            X_test, Y_test = load_frames(path)

        print(f"  frames: {len(X_test)}  attack: {int(Y_test.sum())}  "
              f"benign: {int((Y_test == 0).sum())}")
        preds = self.predict(X_test)
        return preds, Y_test

    def predict(self, X_test=None, **kwargs):
        if X_test is None:
            raise ValueError("[DecisionTree] predict() needs an input")
        return self.dt.predict(np.asarray(X_test))

    def save(self, path):
        joblib.dump(self.dt, path)
        print(f"  DecisionTree model saved to {path}")

    def load(self, path):
        self.dt = joblib.load(path)
        print(f"  DecisionTree model loaded from {path}")

    def extract_features(self):
        pass
