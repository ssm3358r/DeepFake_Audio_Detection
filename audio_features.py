import numpy as np
import librosa

N_MFCC = 40
BASIC_FEATURE_DIM = 80
RICH_FEATURE_DIM = 274


def extract_basic_features_from_signal(audio: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.hstack((mfcc_mean, mfcc_std)).astype(np.float32)


def extract_rich_features_from_signal(audio: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    rms = librosa.feature.rms(y=audio)

    features = np.hstack(
        (
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
            np.mean(delta2, axis=1),
            np.std(delta2, axis=1),
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),
            [
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(zcr),
                np.std(zcr),
                np.mean(rms),
                np.std(rms),
            ],
        )
    )

    return features.astype(np.float32)


def resolve_feature_mode_from_dim(feature_dim: int) -> str:
    if feature_dim == BASIC_FEATURE_DIM:
        return "basic"
    if feature_dim == RICH_FEATURE_DIM:
        return "rich"
    raise ValueError(
        f"Unsupported feature dimension {feature_dim}. Expected {BASIC_FEATURE_DIM} or {RICH_FEATURE_DIM}."
    )


def extract_features_for_mode(audio: np.ndarray, sr: int, mode: str) -> np.ndarray:
    if mode == "basic":
        return extract_basic_features_from_signal(audio, sr)
    if mode == "rich":
        return extract_rich_features_from_signal(audio, sr)
    raise ValueError(f"Unknown feature mode: {mode}")
