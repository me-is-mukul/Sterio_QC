"""Side-by-side stereo geometry.

Computes the left/right crop rectangles from the configured layout and split
ratio, validated against the real frame size. Cropping itself happens inside the
FFmpeg sampling pass, so both views of a pair always come from the same decoded
frame (identical timestamp by construction).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from video_pipeline.core.errors import SamplingError


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    width: int
    height: int

    def ffmpeg_filter(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


@dataclass(frozen=True)
class StereoGeometry:
    layout: str
    frame_width: int
    frame_height: int
    split_ratio: float
    left: Crop
    right: Crop
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


def compute_geometry(frame_width: int, frame_height: int, *, layout: str, split_ratio: float) -> StereoGeometry:
    if layout != "side_by_side":
        raise SamplingError(f"Unsupported stereo layout '{layout}'.")
    if frame_width < 2 or frame_height < 1:
        raise SamplingError(f"Invalid frame size {frame_width}x{frame_height}.")

    warnings: list[str] = []
    exact = frame_width * split_ratio
    left_width = int(round(exact))
    if left_width % 2:  # keep widths even for 4:2:0 chroma
        left_width -= 1
        warnings.append(f"Left width rounded down to even value {left_width}.")
    right_width = frame_width - left_width
    if right_width % 2:
        right_width -= 1
        warnings.append(f"Right width rounded down to even value {right_width}; last column dropped.")
    if left_width <= 0 or right_width <= 0:
        raise SamplingError(f"split_ratio {split_ratio} leaves no room for one of the views.")
    if left_width != right_width:
        warnings.append(f"Views have different widths ({left_width} vs {right_width}); check stereo.split_ratio.")
    if frame_width / frame_height < 2.0 and split_ratio == 0.5:
        warnings.append(
            f"Aspect ratio {frame_width}x{frame_height} is narrow for side-by-side stereo; "
            "each view would be taller than wide. Verify the layout."
        )
    return StereoGeometry(
        layout=layout,
        frame_width=frame_width,
        frame_height=frame_height,
        split_ratio=split_ratio,
        left=Crop(0, 0, left_width, frame_height),
        right=Crop(left_width, 0, right_width, frame_height),
        warnings=tuple(warnings),
    )
