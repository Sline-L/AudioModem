from pathlib import Path
import wave

import numpy as np

p = Path(__file__).resolve().parent
N, C = 1024, 32
bins = np.arange(1, 512)
h = np.fft.fft(np.loadtxt(p / "channel.csv"), N)[bins]
(p / "recovered").mkdir(exist_ok=True)

for f in sorted((p / "data").glob("*.wav")):
    with wave.open(str(f), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32768
    y = x[: x.size // (N + C) * (N + C)].reshape(-1, N + C)[:, C:]
    q = np.fft.fft(y, axis=1)[:, bins] / h
    for bits in ((q.imag < 0, q.real < 0), (q.real < 0, q.imag < 0)):
        try:
            b = np.packbits(np.column_stack(bits).astype(np.uint8).ravel(), bitorder="big").tobytes()
            i, j = b.index(0), b.index(0, b.index(0) + 1)
            name = b[:i].decode()
            size = int(b[i + 1 : j])
            (p / "recovered" / Path(name).name).write_bytes(b[j + 1 : j + 1 + size])
            break
        except Exception:
            pass
