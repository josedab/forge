"""Healthcare domain feature pack.

Generates clinical features from patient data: vitals trends,
lab value derivatives, BMI, age groups, comorbidity indicators.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forge.packs.base import FeaturePack


class HealthcareFeaturePack(FeaturePack):
    """Generate healthcare-specific features from clinical data.

    Auto-detects common medical columns (age, weight, height,
    blood pressure, lab values) and generates derived features.

    Args:
        features: Specific features to generate. None for all.
        prefix: Prefix for generated feature names.

    Example:
        >>> pack = HealthcareFeaturePack()
        >>> features = pack.fit_transform(patient_data)
    """

    domain = "healthcare"

    def available_features(self) -> list[str]:
        return [
            "bmi", "age_group", "bp_category", "pulse_pressure",
            "map_pressure", "bmi_category", "lab_zscore",
            "lab_critical_flag", "age_decades", "is_elderly",
            "risk_score_simple",
        ]

    def _generate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        result: dict[str, pd.Series] = {}
        active = set(self.features) if self.features else set(self.available_features())
        p = self.prefix
        cols_lower = {c.lower(): c for c in X.columns}

        # BMI
        if "bmi" in active:
            weight_col = self._find_col(cols_lower, ["weight", "weight_kg", "wt"])
            height_col = self._find_col(cols_lower, ["height", "height_cm", "ht", "height_m"])
            if weight_col and height_col:
                weight = X[weight_col].astype(float)
                height = X[height_col].astype(float)
                # Detect if height is in cm (>3) or m (<3)
                h_m = np.where(height > 3, height / 100, height)
                bmi: Any = weight / (h_m ** 2)
                result[f"{p}bmi"] = pd.Series(bmi, index=X.index)

                if "bmi_category" in active:
                    cats = pd.cut(
                        pd.Series(bmi, index=X.index),
                        bins=[0, 18.5, 25, 30, 100],
                        labels=["underweight", "normal", "overweight", "obese"],
                    )
                    result[f"{p}bmi_category"] = cats.astype(str)

        # Age features
        age_col = self._find_col(cols_lower, ["age", "patient_age", "age_years"])
        if age_col:
            age = X[age_col].astype(float)
            if "age_group" in active:
                result[f"{p}age_group"] = pd.cut(
                    age, bins=[0, 18, 35, 50, 65, 120],
                    labels=["pediatric", "young_adult", "middle_age", "senior", "elderly"],
                ).astype(str)
            if "age_decades" in active:
                result[f"{p}age_decades"] = (age // 10).astype(int)
            if "is_elderly" in active:
                result[f"{p}is_elderly"] = (age >= 65).astype(int)

        # Blood pressure features
        sys_col = self._find_col(
            cols_lower, ["systolic", "systolic_bp", "sbp", "sys_bp"]
        )
        dia_col = self._find_col(
            cols_lower, ["diastolic", "diastolic_bp", "dbp", "dia_bp"]
        )
        if sys_col and dia_col:
            sys = X[sys_col].astype(float)
            dia = X[dia_col].astype(float)

            if "pulse_pressure" in active:
                result[f"{p}pulse_pressure"] = sys - dia

            if "map_pressure" in active:
                result[f"{p}map_pressure"] = dia + (sys - dia) / 3

            if "bp_category" in active:
                bp_cat = pd.Series("normal", index=X.index)
                bp_cat = bp_cat.where(sys < 120, "elevated")
                bp_cat = bp_cat.where(sys < 130, "high_stage1")
                bp_cat = bp_cat.where(sys < 140, "high_stage2")
                bp_cat = bp_cat.where(sys < 180, "crisis")
                result[f"{p}bp_category"] = bp_cat

        # Lab value z-scores
        if "lab_zscore" in active:
            lab_prefixes = ["lab_", "test_", "result_"]
            for col in X.columns:
                if any(col.lower().startswith(lp) for lp in lab_prefixes):
                    if np.issubdtype(X[col].dtype, np.number):
                        std = X[col].std()
                        if std > 0:
                            result[f"{p}{col}_zscore"] = (X[col] - X[col].mean()) / std

        # Lab critical flags
        if "lab_critical_flag" in active:
            lab_prefixes = ["lab_", "test_", "result_"]
            for col in X.columns:
                if any(col.lower().startswith(lp) for lp in lab_prefixes):
                    if np.issubdtype(X[col].dtype, np.number):
                        q1 = X[col].quantile(0.01)
                        q99 = X[col].quantile(0.99)
                        result[f"{p}{col}_critical"] = (
                            (X[col] < q1) | (X[col] > q99)
                        ).astype(int)

        # Simple risk score (normalized sum of concerning indicators)
        if "risk_score_simple" in active and age_col:
            risk = pd.Series(0.0, index=X.index)
            age = X[age_col].astype(float)
            risk += (age > 65).astype(float)
            if sys_col:
                risk += (X[sys_col].astype(float) > 140).astype(float)
            if f"{p}bmi" in result:
                risk += (result[f"{p}bmi"] > 30).astype(float)
            result[f"{p}risk_score_simple"] = risk

        return pd.DataFrame(result, index=X.index)

    @staticmethod
    def _find_col(
        cols_lower: dict[str, str], candidates: list[str]
    ) -> str | None:
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        return None
