"""Live, 24/7-station-oriented pieces -- as opposed to ``irix.demo``,
which is either fully synthetic or a single-pass run against one
already-known video/webcam source that exits when the source runs out.

``irix.live.camera_source.ReconnectingFrameSource`` and ``irix.live.
station_runner.StationSessionRunner`` are what actually running one of
this repo's stations continuously, across many members' sessions over a
day, needs on top of the single-session pipeline (``irix.pipeline.
rep_session.RepSession``) that already exists.
"""
