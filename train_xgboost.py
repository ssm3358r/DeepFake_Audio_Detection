import argparse
import os
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from audio_features import extract_rich_features_from_signal


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost deepfake audio detector")
    parser.add_argument("--dataset", default=".", help="Dataset root directory")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--model-out", default="deepfake_model_xgb.pkl", help="Output model path")
    parser.add_argument("--scaler-out", default="scaler_xgb.pkl", help="Output scaler path")
    parser.add_argument(
        "--cache-file",
        default="features_cache_rich.npz",
        help="Path to cached extracted features",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuilding features even if cache exists",
    )
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of boosting rounds")
    parser.add_argument("--max-depth", type=int, default=5, help="Tree max depth")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Early stopping rounds on validation set",
    )
    return parser.parse_args()


def list_folders(dataset_root: Path):
    folders = [p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    folders = [p for p in folders if p.name not in {"__pycache__"}]
    return sorted(folders, key=lambda p: p.name.lower())


def collect_wav_files(folder: Path):
    return sorted(folder.glob("*.wav"))


def load_feature(file_path: Path, sr: int):
    audio, _ = librosa.load(file_path.as_posix(), sr=sr, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if audio.size == 0:
        return None
    return extract_rich_features_from_signal(audio, sr)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    folders = list_folders(dataset_root)
    if not folders:
        raise RuntimeError("No class folders found in dataset root.")

    cache_path = Path(args.cache_file)
    if cache_path.exists() and not args.rebuild_cache:
        print(f"Loading cached features from {cache_path} ...")
        cached = np.load(cache_path)
        X = cached["X"].astype(np.float32)
        y = cached["y"].astype(np.int64)
        skipped = int(cached["skipped"])
    else:
        X, y = [], []
        skipped = 0
        class_counts = {}

        print("Scanning folders...")
        for folder in folders:
            wav_files = collect_wav_files(folder)
            if not wav_files:
                continue

            label = 0 if folder.name == "real_samples" else 1
            class_counts[folder.name] = len(wav_files)

            print(f"- {folder.name}: {len(wav_files)} files -> label {label}")
            for wav_file in wav_files:
                try:
                    feat = load_feature(wav_file, args.sr)
                    if feat is None:
                        skipped += 1
                        continue
                    X.append(feat)
                    y.append(label)
                except Exception:
                    skipped += 1

        if not X:
            raise RuntimeError("No training samples were extracted. Check dataset/audio format.")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        np.savez_compressed(cache_path, X=X, y=y, skipped=skipped)
        print(f"Saved feature cache to {cache_path}")

    print("\nData summary")
    print(f"Total samples: {len(y)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Skipped files: {skipped}")
    print(f"Real samples: {(y == 0).sum()}")
    print(f"Fake samples: {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Install with: pip install xgboost"
        ) from exc

    pos_count = max((y_train == 1).sum(), 1)
    neg_count = max((y_train == 0).sum(), 1)
    scale_pos_weight = neg_count / pos_count

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        max_depth=args.max_depth,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.random_state,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\nEvaluation")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(model, args.model_out)
    joblib.dump(scaler, args.scaler_out)

    print("\nSaved artifacts")
    print(f"Model: {args.model_out}")
    print(f"Scaler: {args.scaler_out}")


if __name__ == "__main__":
    main()
