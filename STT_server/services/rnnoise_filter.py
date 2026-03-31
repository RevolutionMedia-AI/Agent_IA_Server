"""RNNoise/detection wrapper + lightweight fallback denoiser.

This module exposes `RNNoiseFilter` which attempts to use a native
RNNoise binding if available. When not available it can run a simple
spectral-gating denoiser implemented with NumPy. The class operates on
μ-law 8k frames (as used by Twilio) and returns μ-law frames.

The implementation is conservative: if no optional dependencies are
installed the filter is a no-op unless `RNNOISE_FALLBACK_ENABLED` is
True.
"""
from __future__ import annotations

import logging
import math
import audioop
from typing import Optional

import numpy as np

from STT_server.config import RNNOISE_ENABLED, RNNOISE_FALLBACK_ENABLED, TWILIO_SR

log = logging.getLogger("stt_server.rnnoise")


class RNNoiseFilter:
    """Denoiser that works on 20ms μ-law frames (8 kHz).

    Usage:
        f = RNNoiseFilter()
        out_frame = f.process_mulaw_frame(in_frame)
    """

    def __init__(self, sample_rate: int = TWILIO_SR, frame_ms: int = 20):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_len = int(sample_rate * frame_ms / 1000)

        # Try optional RNNoise backends (best-effort). We do not require them.
        self.backend = None
        self._backend_obj = None
        if RNNOISE_ENABLED:
            # Prefer local ctypes wrapper if present (loads system librnnoise)
            try:
                from STT_server.services.rnnoise_ctypes import RNNoise as _RNNoise

                self._backend_obj = _RNNoise()
                self.backend = "rnnoise_ctypes"
                log.info("RNNoise ctypes wrapper loaded")
            except Exception:
                try:
                    # Common Python binding (if installed via pip)
                    from rnnoise import RNNoise as _RNNoise

                    self._backend_obj = _RNNoise()
                    self.backend = "rnnoise_py"
                    log.info("RNNoise Python binding loaded")
                except Exception:
                    # Could try other bindings here; keep it simple.
                    log.debug("RNNoise binding not available; falling back")

        # Safety: native RNNoise implementations expect a fixed frame size
        # (typically 480 samples at 48 kHz). If our pipeline uses a different
        # sample rate (e.g. Twilio 8 kHz) or frame length, do not call the
        # native backend to avoid out-of-bounds reads in native code which can
        # lead to segmentation faults. In that case we fall back to the
        # spectral-gating denoiser below.
        if self._backend_obj is not None:
            try:
                if self.sample_rate != 48000:
                    log.warning(
                        "Native RNNoise backend disabled: unsupported sample_rate=%s — using fallback",
                        self.sample_rate,
                    )
                    self._backend_obj = None
                    self.backend = None
                else:
                    # Expect 480-sample frames at 48k for 10ms frames; ensure
                    # our frame length matches or warn.
                    expected_len = int(480 * (self.frame_ms / 10.0))
                    if self.frame_len != expected_len:
                        log.warning(
                            "Native RNNoise backend frame length mismatch: expected %s got %s — disabling native backend",
                            expected_len,
                            self.frame_len,
                        )
                        self._backend_obj = None
                        self.backend = None
            except Exception:
                # If any runtime introspection fails, be conservative and
                # disable native backend to avoid crashes.
                log.exception("Error checking RNNoise backend compatibility; disabling native backend")
                self._backend_obj = None
                self.backend = None

        # Fallback denoiser state (spectral gating)
        self.noise_mag = None
        self.noise_alpha = 0.98  # EMA for noise spectrum
        self.noise_update_threshold = 200.0  # int16 RMS threshold to consider frame "noise" for update
        self.reduction_factor = 1.0

        # Resampler states for converting between pipeline sample rate and
        # RNNoise's expected 48 kHz processing rate. We keep separate state
        # objects for up/down conversions so audioop.ratecv can maintain
        # continuity across frames.
        self._ratecv_up_state = None
        self._ratecv_down_state = None
    def available(self) -> bool:
        return self.backend is not None or RNNOISE_FALLBACK_ENABLED

    def process_mulaw_frame(self, mulaw_bytes: bytes) -> bytes:
        """Accepts a single μ-law frame (raw bytes) and returns processed μ-law bytes.

        This function is robust: on any internal error it returns the original frame.
        """
        try:
            # Convert μ-law (8-bit) -> PCM16 (2 bytes per sample)
            pcm = audioop.ulaw2lin(mulaw_bytes, 2)
            # Ensure correct length
            if len(pcm) != self.frame_len * 2:
                # If chunking differs, attempt to process whatever we have.
                pass

            if self._backend_obj is not None:
                # Try RNNoise Python binding path (best-effort API assumption)
                try:
                        # If the pipeline sample rate is not 48 kHz, resample up to
                        # 48 kHz, split into 480-sample blocks, process each block
                        # with the native RNNoise backend, then resample down to
                        # the original sample rate.
                        if self.sample_rate != 48000:
                            # Upsample PCM to 48k
                            up_bytes, self._ratecv_up_state = audioop.ratecv(
                                pcm, 2, 1, self.sample_rate, 48000, self._ratecv_up_state
                            )

                            up_samples = np.frombuffer(up_bytes, dtype=np.int16)
                            if up_samples.size == 0:
                                raise RuntimeError("resampler returned no samples")

                            out_blocks = []
                            # Process in 480-sample chunks expected by RNNoise
                            for i in range(0, up_samples.size, 480):
                                block = up_samples[i : i + 480]
                                if block.size == 0:
                                    continue
                                if block.size < 480:
                                    # Pad last block to 480 samples
                                    pad = np.zeros(480 - block.size, dtype=np.int16)
                                    block = np.concatenate([block, pad])

                                arr = block.astype(np.float32) / 32768.0
                                outf = self._backend_obj.filter(arr)
                                outf = np.asarray(outf, dtype=np.float32)
                                # Ensure output block length is 480
                                if outf.size != 480:
                                    if outf.size < 480:
                                        outf = np.pad(outf, (0, 480 - outf.size), mode="constant")
                                    else:
                                        outf = outf[:480]

                                out_int16 = np.clip(outf * 32768.0, -32768, 32767).astype(np.int16)
                                out_blocks.append(out_int16)

                            if not out_blocks:
                                raise RuntimeError("no output blocks produced by RNNoise processing")

                            out_resampled = np.concatenate(out_blocks)
                            out_resampled_bytes = out_resampled.tobytes()

                            # Downsample processed 48k audio back to original rate
                            down_bytes, self._ratecv_down_state = audioop.ratecv(
                                out_resampled_bytes, 2, 1, 48000, self.sample_rate, self._ratecv_down_state
                            )

                            # Ensure output length matches input length (pad/trim)
                            if len(down_bytes) < len(pcm):
                                down_bytes = down_bytes + (b"\x00" * (len(pcm) - len(down_bytes)))
                            elif len(down_bytes) > len(pcm):
                                down_bytes = down_bytes[: len(pcm)]

                            return audioop.lin2ulaw(down_bytes, 2)

                        else:
                            # 48 kHz pipeline — direct processing
                            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                            out = self._backend_obj.filter(arr)
                            out_int16 = np.clip((out * 32768.0), -32768, 32767).astype(np.int16)
                            out_bytes = out_int16.tobytes()
                            return audioop.lin2ulaw(out_bytes, 2)
                except Exception:
                    log.exception("RNNoise backend processing failed; falling back to spectral gate")

            # Fallback spectral gating
            if RNNOISE_FALLBACK_ENABLED:
                return self._spectral_gate_mulaw(pcm)

            # No processing available; return original
            return mulaw_bytes

        except Exception:
            log.exception("Error processing mulaw frame in RNNoiseFilter")
            return mulaw_bytes

    def _spectral_gate_mulaw(self, pcm_bytes: bytes) -> bytes:
        # Convert to numpy float32
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            if samples.size == 0:
                return audioop.lin2ulaw(pcm_bytes, 2)

            # Window to reduce FFT artifacts
            win = np.hanning(samples.size)
            frame = samples * win

            # FFT
            spec = np.fft.rfft(frame)
            mag = np.abs(spec)
            phase = np.angle(spec)

            # Estimate noise spectrum from low-energy frames
            rms = math.sqrt(float((samples.astype(np.float32) ** 2).mean()))
            if self.noise_mag is None:
                # Initialize with current magnitude scaled low
                self.noise_mag = mag * 0.5
            else:
                if rms < self.noise_update_threshold:
                    # Update noise estimate
                    self.noise_mag = self.noise_alpha * self.noise_mag + (1.0 - self.noise_alpha) * mag

            # Subtract noise magnitude
            new_mag = mag - (self.reduction_factor * self.noise_mag)
            new_mag = np.maximum(new_mag, 0.0)

            # Reconstruct and inverse FFT
            new_spec = new_mag * np.exp(1j * phase)
            new_frame = np.fft.irfft(new_spec)

            # Undo window using simple normalization (avoid amplifying)
            denom = np.where(win == 0, 1.0, win)
            new_samples = new_frame / denom

            # Cast back to int16 safely
            out = np.clip(new_samples, -32768.0, 32767.0).astype(np.int16)
            out_bytes = out.tobytes()
            return audioop.lin2ulaw(out_bytes, 2)
        except Exception:
            log.exception("Spectral gate failed; returning original pcm")
            return audioop.lin2ulaw(pcm_bytes, 2)


__all__ = ["RNNoiseFilter"]
