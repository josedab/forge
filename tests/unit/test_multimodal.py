"""Tests for multi-modal feature fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.multimodal import (
    ImageFeatureExtractor,
    MultiModalFusion,
    SignalFeatureExtractor,
)


@pytest.fixture
def image_df():
    np.random.seed(42)
    images = [np.random.rand(64).astype(np.float32) for _ in range(50)]
    return pd.DataFrame({
        "id": range(50),
        "price": np.random.randn(50),
        "thumbnail": images,
    })


@pytest.fixture
def signal_df():
    np.random.seed(42)
    signals = [np.sin(np.linspace(0, 2 * np.pi * f, 256)) for f in range(1, 51)]
    return pd.DataFrame({
        "id": range(50),
        "value": np.random.randn(50),
        "audio": signals,
    })


@pytest.fixture
def multimodal_df():
    np.random.seed(42)
    images = [np.random.rand(64).astype(np.float32) for _ in range(50)]
    signals = [np.sin(np.linspace(0, 2 * np.pi, 256)) for _ in range(50)]
    return pd.DataFrame({
        "price": np.random.randn(50),
        "category": np.random.choice(["A", "B"], 50),
        "thumbnail": images,
        "audio": signals,
    })


class TestImageFeatureExtractor:
    """Tests for ImageFeatureExtractor."""

    def test_fit_transform_basic(self, image_df):
        ext = ImageFeatureExtractor(columns=["thumbnail"])
        result = ext.fit_transform(image_df)
        assert result.shape[0] == 50
        assert result.shape[1] > image_df.shape[1]

    def test_extracts_named_features(self, image_df):
        ext = ImageFeatureExtractor(
            columns=["thumbnail"],
            features=["mean", "std"],
        )
        result = ext.fit_transform(image_df)
        assert "img_thumbnail_mean" in result.columns
        assert "img_thumbnail_std" in result.columns

    def test_histogram_features(self, image_df):
        ext = ImageFeatureExtractor(
            columns=["thumbnail"],
            features=["histogram"],
            histogram_bins=4,
        )
        result = ext.fit_transform(image_df)
        hist_cols = [c for c in result.columns if "hist_" in c]
        assert len(hist_cols) == 4

    def test_preserves_original(self, image_df):
        ext = ImageFeatureExtractor(columns=["thumbnail"])
        result = ext.fit_transform(image_df)
        assert "price" in result.columns
        assert "id" in result.columns

    def test_custom_prefix(self, image_df):
        ext = ImageFeatureExtractor(
            columns=["thumbnail"], output_prefix="photo"
        )
        result = ext.fit_transform(image_df)
        photo_cols = [c for c in result.columns if c.startswith("photo_")]
        assert len(photo_cols) > 0

    def test_get_feature_names_out(self, image_df):
        ext = ImageFeatureExtractor(columns=["thumbnail"])
        ext.fit(image_df)
        names = ext.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_transform_before_fit_raises(self, image_df):
        ext = ImageFeatureExtractor()
        with pytest.raises(RuntimeError, match="must be fitted"):
            ext.transform(image_df)

    def test_no_image_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ext = ImageFeatureExtractor()
        result = ext.fit_transform(df)
        assert result.shape == df.shape


class TestSignalFeatureExtractor:
    """Tests for SignalFeatureExtractor."""

    def test_fit_transform_basic(self, signal_df):
        ext = SignalFeatureExtractor(columns=["audio"])
        result = ext.fit_transform(signal_df)
        assert result.shape[0] == 50
        assert result.shape[1] > signal_df.shape[1]

    def test_spectral_features(self, signal_df):
        ext = SignalFeatureExtractor(
            columns=["audio"],
            features=["peak_freq", "spectral_entropy"],
        )
        result = ext.fit_transform(signal_df)
        assert "sig_audio_peak_freq" in result.columns
        assert "sig_audio_spectral_entropy" in result.columns

    def test_time_domain_features(self, signal_df):
        ext = SignalFeatureExtractor(
            columns=["audio"],
            features=["mean", "std", "rms", "zero_crossings"],
        )
        result = ext.fit_transform(signal_df)
        assert "sig_audio_rms" in result.columns
        assert "sig_audio_zero_crossings" in result.columns

    def test_different_frequency_signals(self, signal_df):
        ext = SignalFeatureExtractor(
            columns=["audio"], features=["peak_freq"]
        )
        result = ext.fit_transform(signal_df)
        # Different input frequencies should give different peak_freq
        peak_freqs = result["sig_audio_peak_freq"].values
        assert len(np.unique(peak_freqs)) > 1

    def test_preserves_original(self, signal_df):
        ext = SignalFeatureExtractor(columns=["audio"])
        result = ext.fit_transform(signal_df)
        assert "value" in result.columns

    def test_get_feature_names_out(self, signal_df):
        ext = SignalFeatureExtractor(columns=["audio"])
        ext.fit(signal_df)
        names = ext.get_feature_names_out()
        assert all("sig_audio" in n for n in names)


class TestMultiModalFusion:
    """Tests for MultiModalFusion."""

    def test_fit_transform_basic(self, multimodal_df):
        fusion = MultiModalFusion(
            image_columns=["thumbnail"],
            signal_columns=["audio"],
            tabular_columns=["price"],
        )
        result = fusion.fit_transform(multimodal_df)
        assert result.shape[0] == 50
        assert result.shape[1] > 1  # More than just price

    def test_includes_all_modalities(self, multimodal_df):
        fusion = MultiModalFusion(
            image_columns=["thumbnail"],
            signal_columns=["audio"],
            tabular_columns=["price"],
        )
        result = fusion.fit_transform(multimodal_df)
        assert "price" in result.columns
        img_cols = [c for c in result.columns if c.startswith("img_")]
        sig_cols = [c for c in result.columns if c.startswith("sig_")]
        assert len(img_cols) > 0
        assert len(sig_cols) > 0

    def test_dimension_reduction(self, multimodal_df):
        fusion = MultiModalFusion(
            image_columns=["thumbnail"],
            signal_columns=["audio"],
            tabular_columns=["price"],
            reduce_dim=5,
            random_state=42,
        )
        result = fusion.fit_transform(multimodal_df)
        pc_cols = [c for c in result.columns if c.startswith("fused_pc_")]
        assert len(pc_cols) == 5

    def test_get_feature_names_out(self, multimodal_df):
        fusion = MultiModalFusion(
            image_columns=["thumbnail"],
            tabular_columns=["price"],
        )
        fusion.fit(multimodal_df)
        names = fusion.get_feature_names_out()
        assert isinstance(names, list)

    def test_transform_before_fit_raises(self, multimodal_df):
        fusion = MultiModalFusion()
        with pytest.raises(RuntimeError, match="must be fitted"):
            fusion.transform(multimodal_df)
