"""Runtime-only diagnostics for NuRec dynamic-track initialization.

Mount this file as ``sitecustomize.py`` ahead of NuRec's Python path. It does
not change the vendor package; it records the exact track subset and point
cloud rows received by ``DynamicTracksInitialization`` before delegating to
the original implementation.
"""

from __future__ import annotations

import logging


LOG = logging.getLogger("closedloopbench.nurec_dynamic_initializer_probe")


def _install_probe() -> None:
    from nre.models.gaussians.initializations import DynamicTracksInitialization

    original = DynamicTracksInitialization.initialize_from_datasource

    def instrumented_initialize(self, datasource, **kwargs):
        cuboid_tracks = kwargs.get("cuboid_tracks")
        track_ids = list(getattr(cuboid_tracks, "tracks_id", []))
        pose_count = len(getattr(cuboid_tracks, "tracks_timestamps_us", []))
        LOG.warning("CLB_DYNAMIC_INIT tracks=%s poses=%s ids=%s", len(track_ids), pose_count, track_ids)

        original_get_track_point_clouds = datasource.get_track_point_clouds

        def instrumented_get_track_point_clouds(*args, **call_kwargs):
            rows = list(original_get_track_point_clouds(*args, **call_kwargs))
            LOG.warning(
                "CLB_DYNAMIC_INIT_ROWS rows=%s nonempty=%s points=%s",
                len(rows),
                sum(row.point_cloud.n_points > 0 for row in rows),
                sum(row.point_cloud.n_points for row in rows),
            )
            return iter(rows)

        datasource.get_track_point_clouds = instrumented_get_track_point_clouds
        try:
            return original(self, datasource, **kwargs)
        finally:
            datasource.get_track_point_clouds = original_get_track_point_clouds

    DynamicTracksInitialization.initialize_from_datasource = instrumented_initialize


try:
    _install_probe()
except Exception as exc:  # pragma: no cover - executed only inside the vendor image.
    LOG.warning("CLB_DYNAMIC_INIT_PROBE_INSTALL_FAILED %r", exc)
