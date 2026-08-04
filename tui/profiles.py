from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FS = 48_000
FEC_MODES = ("none", "conv", "rs", "rs_conv")
MODULATIONS = ("bpsk", "qpsk")


@dataclass(slots=True)
class ModemProfile:
    name: str = "TUI v1 Default"
    protocol: str = "tui_v1"
    fs: int = FS
    fft_size: int = 512
    cp_samples: int = 256
    active_ranges: list[list[int]] = field(default_factory=lambda: [[64, 120], [158, 178]])
    modulation: str = "bpsk"
    fec: str = "conv"
    pilot_spacing: int = 4
    noise_seconds: float = 0.5
    tail_seconds: float = 0.25
    sync_symbols: int = 64
    sync_correlation_symbols: int = 8
    sync_seed: int = 7026
    preamble_symbols: int = 128
    preamble_repeats: int = 2
    preamble_seed: int = 8026
    block_size: int = 512
    header_repeats: int = 3
    pilot_seed: int = 2027
    fec_seed: int = 7027
    timing_anchor_interval: int = 128
    timing_anchor_symbols: int = 8
    timing_anchor_seed: int = 9028
    payload_start_anchor_symbols: int = 8
    payload_start_anchor_seed: int = 10028
    channel_alpha: float = 0.35
    payload_start_anchor_h_alpha: float = 0.5
    anchor_h_alpha: float = 0.5
    anchor_min_score: float = 0.12
    training_anchor_symbols: int = 8
    training_anchor_step: int = 32
    clock_search: int = 128
    payload_search: int = 16
    phase_slope: str = "off"
    slope_window: int = 64
    slope_clip: float = 0.05
    immutable: bool = False

    @property
    def symbol_len(self) -> int:
        return self.fft_size + self.cp_samples

    @property
    def active_bins(self) -> list[int]:
        bins: list[int] = []
        for start, end in self.active_ranges:
            bins.extend(range(start, end + 1))
        return bins

    @property
    def frequency_ranges_hz(self) -> list[list[float]]:
        return [[a * self.fs / self.fft_size, b * self.fs / self.fft_size] for a, b in self.active_ranges]

    @property
    def profile_id(self) -> str:
        air_fields = (
            "protocol", "fs", "fft_size", "cp_samples", "active_ranges",
            "modulation", "fec", "pilot_spacing", "noise_seconds",
            "sync_symbols", "sync_seed", "preamble_symbols", "preamble_repeats",
            "preamble_seed", "block_size", "header_repeats", "pilot_seed",
            "fec_seed", "timing_anchor_interval", "timing_anchor_symbols",
            "timing_anchor_seed", "payload_start_anchor_symbols",
            "payload_start_anchor_seed",
        )
        values = {key: getattr(self, key) for key in air_fields}
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()[:16]

    def validate(self) -> None:
        if self.fs != FS:
            raise ValueError("TUI v1 currently requires 48000 Hz")
        if self.fft_size < 256 or self.fft_size > 4096 or self.fft_size & (self.fft_size - 1):
            raise ValueError("fft_size must be a power of two between 256 and 4096")
        if not 1 <= self.cp_samples <= self.fft_size:
            raise ValueError("cp_samples must be between 1 and fft_size")
        if self.modulation not in MODULATIONS:
            raise ValueError(f"unsupported modulation: {self.modulation}")
        if self.fec not in FEC_MODES:
            raise ValueError(f"unsupported FEC: {self.fec}")
        if not 2 <= self.pilot_spacing <= 32:
            raise ValueError("pilot_spacing must be between 2 and 32")
        if not self.active_ranges:
            raise ValueError("at least one active range is required")
        if any(start > end for start, end in self.active_ranges):
            raise ValueError("active bin range start exceeds end")
        if any(self.active_ranges[i][0] <= self.active_ranges[i - 1][1] for i in range(1, len(self.active_ranges))):
            raise ValueError("active bin ranges must be sorted and non-overlapping")
        bins = self.active_bins
        if len(set(bins)) != len(bins):
            raise ValueError("active bin ranges overlap")
        if min(bins) < 1 or max(bins) >= self.fft_size // 2:
            raise ValueError("active bins must exclude DC and Nyquist")
        if len(bins) < self.pilot_spacing * 2:
            raise ValueError("too few active bins for the selected pilot spacing")
        positive = (
            "sync_symbols", "sync_correlation_symbols", "preamble_symbols",
            "preamble_repeats", "block_size", "header_repeats",
            "timing_anchor_interval", "timing_anchor_symbols",
            "training_anchor_symbols", "training_anchor_step", "clock_search",
            "slope_window",
        )
        for key in positive:
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be >= 1")
        if self.sync_correlation_symbols > self.sync_symbols:
            raise ValueError("sync_correlation_symbols exceeds sync_symbols")
        if self.training_anchor_symbols > self.sync_symbols + self.preamble_symbols * self.preamble_repeats:
            raise ValueError("training_anchor_symbols exceeds complete training")
        if self.block_size > 65535:
            raise ValueError("block_size must fit the 16-bit frame field")
        if self.payload_start_anchor_symbols < 0 or self.payload_search < 0:
            raise ValueError("anchor symbols and search radius must be non-negative")
        if self.noise_seconds < 0 or self.tail_seconds < 0:
            raise ValueError("silence durations must be non-negative")
        for key in ("channel_alpha", "payload_start_anchor_h_alpha", "anchor_h_alpha", "anchor_min_score"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
        if self.phase_slope not in ("off", "slow"):
            raise ValueError("phase_slope must be off or slow")

    def to_dict(self, include_id: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "symbol_len": self.symbol_len,
                "active_bins": self.active_bins,
                "active_frequency_ranges_hz": self.frequency_ranges_hz,
            }
        )
        if include_id:
            result["profile_id"] = self.profile_id
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModemProfile":
        allowed = set(cls.__dataclass_fields__)
        profile = cls(**{key: value for key, value in raw.items() if key in allowed})
        profile.validate()
        expected = raw.get("profile_id")
        if expected and expected != profile.profile_id:
            raise ValueError("profile checksum mismatch")
        return profile

    @classmethod
    def load(cls, path: Path) -> "ModemProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        self.validate()
        path = Path(path)
        if path.exists() and self.immutable:
            raise ValueError("immutable profile cannot be overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def step8_profile() -> ModemProfile:
    return ModemProfile(name="Step8 Compatible", protocol="step8_compatible", immutable=True)


def default_profile() -> ModemProfile:
    return ModemProfile()


def parse_ranges(text: str) -> list[list[int]]:
    ranges: list[list[int]] = []
    for item in text.replace(" ", "").split(","):
        if not item:
            continue
        parts = item.split("-")
        if len(parts) == 1:
            start = end = int(parts[0])
        elif len(parts) == 2:
            start, end = map(int, parts)
        else:
            raise ValueError(f"invalid bin range: {item}")
        if start > end:
            raise ValueError(f"reversed bin range: {item}")
        ranges.append([start, end])
    return ranges


def ensure_builtin_profiles() -> None:
    folder = ROOT / "profiles"
    folder.mkdir(parents=True, exist_ok=True)
    for filename, profile in (
        ("step8_compatible.json", step8_profile()),
        ("tui_v1_default.json", default_profile()),
    ):
        path = folder / filename
        if not path.exists():
            path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
