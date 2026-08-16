import numpy as np

import lib.ldpc.py.ldpc as ldpc
from modem_n2 import LDPC_PTYPE, LDPC_RATE, LDPC_STANDARD, LDPC_Z


def main():
    code = ldpc.code(LDPC_STANDARD, LDPC_RATE, LDPC_Z, LDPC_PTYPE)
    rng = np.random.default_rng(2026)
    for info in (np.zeros(code.K, dtype=int), rng.integers(0, 2, code.K)):
        encoded = code.encode(info)
        app, iterations = code.decode(10.0 * (0.5 - encoded), "sumprod2")
        decoded = (app < 0.0).astype(int)
        errors = int(np.count_nonzero(decoded != encoded))
        print(
            f"{code.standard} rate={code.rate} z={code.z} "
            f"K={code.K} N={code.N} iterations={iterations} errors={errors}"
        )
        if errors:
            raise SystemExit("LDPC smoke test failed")


if __name__ == "__main__":
    main()
