"""Type stubs for ``edgefirst.tracker``.

ByteTrack lives here, not on ``edgefirst.decoder``. The decoder wheel's
``decode_tracked`` accepts any object with an ``update`` method.
"""

def version() -> str:
    """Return the HAL version string (matches ``Cargo.toml``)."""

class TrackInfo:
    """Identity and Kalman-smoothed location for one track."""

    def __init__(
        self,
        uuid: str,
        tracked_location: tuple[float, float, float, float],
        count: int,
        created: int,
        last_updated: int,
    ) -> None: ...
    @property
    def uuid(self) -> str: ...
    @property
    def tracked_location(self) -> tuple[float, float, float, float]: ...
    @property
    def count(self) -> int: ...
    @property
    def created(self) -> int: ...
    @property
    def last_updated(self) -> int: ...

class ActiveTrackInfo:
    """A live track plus the last associated detection."""

    def __init__(
        self,
        track_info: TrackInfo,
        bbox: tuple[float, float, float, float],
        score: float,
        label: int,
    ) -> None: ...
    @property
    def info(self) -> TrackInfo: ...
    @property
    def last_box(self) -> tuple[tuple[float, float, float, float], float, int]: ...

class ByteTrack:
    """Kalman-filtered multi-object tracker with stable per-track UUIDs."""

    def __init__(
        self,
        high_conf: float = 0.7,
        iou: float = 0.25,
        update: float = 0.25,
        lifespan_ns: int = 500_000_000,
    ) -> None: ...
    def update(
        self,
        boxes: object,
        scores: object,
        labels: object,
        timestamp_ns: int,
    ) -> list[TrackInfo | None]:
        """Associate detections for this timestamp.

        ``boxes`` is ``(N, 4)`` XYXY, ``scores`` and ``labels`` are length ``N``.
        """

    def get_active_tracks(self) -> list[ActiveTrackInfo]: ...
