"""Geospatial domain feature pack.

Generates distance features, coordinate transformations,
clustering, and spatial statistics from latitude/longitude data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forge.packs.base import FeaturePack


class GeospatialFeaturePack(FeaturePack):
    """Generate geospatial features from latitude/longitude data.

    Auto-detects lat/lon columns and generates distance, bearing,
    density, and spatial bin features.

    Args:
        features: Specific features to generate. None for all.
        lat_col: Column for latitude. Auto-detected if None.
        lon_col: Column for longitude. Auto-detected if None.
        reference_points: Dict of name -> (lat, lon) for distance calculation.
        n_clusters: Number of spatial clusters for k-means.

    Example:
        >>> pack = GeospatialFeaturePack(
        ...     reference_points={"city_center": (40.7128, -74.0060)}
        ... )
        >>> features = pack.fit_transform(location_data)
    """

    domain = "geospatial"

    EARTH_RADIUS_KM = 6371.0

    def __init__(
        self,
        features: list[str] | None = None,
        lat_col: str | None = None,
        lon_col: str | None = None,
        reference_points: dict[str, tuple[float, float]] | None = None,
        n_clusters: int = 5,
        geohash_precision: int = 5,
        prefix: str | None = None,
    ) -> None:
        super().__init__(features=features, prefix=prefix)
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.reference_points = reference_points or {}
        self.n_clusters = n_clusters
        self.geohash_precision = geohash_precision
        self._cluster_centers: np.ndarray[Any, Any] | None = None

    def available_features(self) -> list[str]:
        return [
            "haversine_distance", "bearing", "lat_sin", "lat_cos",
            "lon_sin", "lon_cos", "spatial_cluster", "lat_bin",
            "lon_bin", "distance_to_centroid", "cartesian_x",
            "cartesian_y", "cartesian_z", "manhattan_distance",
        ]

    def _resolve_latlon(self, X: pd.DataFrame) -> tuple[str | None, str | None]:
        cols_lower = {c.lower(): c for c in X.columns}
        lat = self.lat_col
        lon = self.lon_col

        if not lat:
            for c in ["latitude", "lat", "y"]:
                if c in cols_lower:
                    lat = cols_lower[c]
                    break
        if not lon:
            for c in ["longitude", "lon", "lng", "x"]:
                if c in cols_lower:
                    lon = cols_lower[c]
                    break
        return lat, lon

    def _generate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        lat_col, lon_col = self._resolve_latlon(X)
        if not lat_col or not lon_col:
            return pd.DataFrame(index=X.index)

        lat = X[lat_col].astype(float)
        lon = X[lon_col].astype(float)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        result: dict[str, pd.Series] = {}
        active = set(self.features) if self.features else set(self.available_features())
        p = self.prefix

        # Cyclical encoding
        if "lat_sin" in active:
            result[f"{p}lat_sin"] = np.sin(lat_rad)
        if "lat_cos" in active:
            result[f"{p}lat_cos"] = np.cos(lat_rad)
        if "lon_sin" in active:
            result[f"{p}lon_sin"] = np.sin(lon_rad)
        if "lon_cos" in active:
            result[f"{p}lon_cos"] = np.cos(lon_rad)

        # 3D Cartesian coordinates
        if "cartesian_x" in active:
            result[f"{p}cartesian_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        if "cartesian_y" in active:
            result[f"{p}cartesian_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        if "cartesian_z" in active:
            result[f"{p}cartesian_z"] = np.sin(lat_rad)

        # Binning
        if "lat_bin" in active:
            result[f"{p}lat_bin"] = pd.cut(lat, bins=20, labels=False)
        if "lon_bin" in active:
            result[f"{p}lon_bin"] = pd.cut(lon, bins=20, labels=False)

        # Distances to reference points
        if "haversine_distance" in active or "bearing" in active:
            for name, (ref_lat, ref_lon) in self.reference_points.items():
                if "haversine_distance" in active:
                    dist = self._haversine(lat, lon, ref_lat, ref_lon)
                    result[f"{p}dist_to_{name}_km"] = dist
                if "bearing" in active:
                    brg = self._bearing(lat_rad, lon_rad, ref_lat, ref_lon)  # type: ignore[arg-type]
                    result[f"{p}bearing_to_{name}"] = brg

        # Manhattan distance to centroid
        if "manhattan_distance" in active:
            clat = lat.mean()
            clon = lon.mean()
            result[f"{p}manhattan_to_center"] = (lat - clat).abs() + (lon - clon).abs()

        # Distance to centroid
        if "distance_to_centroid" in active:
            clat = lat.mean()
            clon = lon.mean()
            result[f"{p}dist_to_centroid_km"] = self._haversine(lat, lon, clat, clon)

        # Spatial clustering
        if "spatial_cluster" in active:
            result[f"{p}spatial_cluster"] = self._spatial_cluster(lat, lon)

        return pd.DataFrame(result, index=X.index)

    def _haversine(
        self,
        lat1: pd.Series,
        lon1: pd.Series,
        lat2: float,
        lon2: float,
    ) -> pd.Series:
        """Compute haversine distance in km."""
        lat1_r = np.radians(lat1)
        lon1_r = np.radians(lon1)
        lat2_r = np.radians(lat2)
        lon2_r = np.radians(lon2)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return pd.Series(self.EARTH_RADIUS_KM * c, index=lat1.index)

    def _bearing(
        self,
        lat1_rad: pd.Series,
        lon1_rad: pd.Series,
        lat2: float,
        lon2: float,
    ) -> pd.Series:
        """Compute bearing to a reference point."""
        lat2_r = np.radians(lat2)
        lon2_r = np.radians(lon2)
        dlon = lon2_r - lon1_rad
        x = np.sin(dlon) * np.cos(lat2_r)
        y = np.cos(lat1_rad) * np.sin(lat2_r) - np.sin(lat1_rad) * np.cos(lat2_r) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360  # type: ignore[no-any-return]

    def _spatial_cluster(self, lat: pd.Series, lon: pd.Series) -> pd.Series:
        """Simple k-means spatial clustering."""
        from sklearn.cluster import MiniBatchKMeans

        coords = np.column_stack([lat.fillna(0).to_numpy(), lon.fillna(0).to_numpy()])
        n_clusters = min(self.n_clusters, len(coords))
        if n_clusters < 2:
            return pd.Series(0, index=lat.index)
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init=3)
        labels = km.fit_predict(coords)
        self._cluster_centers = km.cluster_centers_
        return pd.Series(labels, index=lat.index)
