#!/usr/bin/env python3
"""
Q8 Neural Network — tehnika: Quantum Amplitude Estimation (QAE)
(čisto kvantno, bez klasičnog treniranja i bez hibrida).

Arhitektura (kanonski Brassard-Høyer-Mosca-Tapp):
  - Priprema stanja A = StatePreparation(amp) nad state registrom (amp iz CELOG CSV-a).
  - „Dobra“ podskupina = TOP-M stanja po verovatnoći; S_f = dijagonalni ±1 oracle (-1 na marked).
  - Grover operator Q = A · S_0 · A† · S_f,  S_0 = 2|0⟩⟨0| - I.
  - QPE sa t pomoćnih qubit-a:
        H^⊗t na ancillama; za j = 0..t-1 primeni controlled-Q^(2^j) sa ancilla[j] kao kontrolom;
        inverzni QFT na ancillama.
  - Iz zajedničkog Statevector-a uzima se marginal nad state registrom → bias_39 → NEXT.

Sve deterministički: seed=39; amp i marked iz CELOG CSV-a.
Deterministička grid-optimizacija (nq, t, M) po meri cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import Diagonal, QFT, StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_T = (3, 4, 5)
GRID_M = (3, 5, 7, 10)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


# =========================
# Amplitude-encoding iz celog CSV-a
# =========================
def amplitude_input(H: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    f = freq_vector(H)
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


def top_m_indices(amp: np.ndarray, m: int) -> List[int]:
    p = amp ** 2
    order = np.argsort(-p, kind="stable")
    return [int(i) for i in order[:m]]


# =========================
# S_f, S_0, Q = A S_0 A† S_f
# =========================
def build_Sf(nq: int, marked: List[int]) -> QuantumCircuit:
    diag = np.ones(2 ** nq, dtype=complex)
    for k in marked:
        if 0 <= k < 2 ** nq:
            diag[k] = -1.0 + 0j
    qc = QuantumCircuit(nq, name="S_f")
    qc.compose(Diagonal(diag.tolist()), range(nq), inplace=True)
    return qc


def build_S0(nq: int) -> QuantumCircuit:
    """S_0 = 2|0⟩⟨0| − I  (fazni flip svega sem |0⟩)."""
    diag = -np.ones(2 ** nq, dtype=complex)
    diag[0] = 1.0 + 0j
    qc = QuantumCircuit(nq, name="S_0")
    qc.compose(Diagonal(diag.tolist()), range(nq), inplace=True)
    return qc


def build_Q(nq: int, amp: np.ndarray, marked: List[int]) -> QuantumCircuit:
    """Q = A · S_0 · A† · S_f"""
    A = StatePreparation(amp.tolist())
    A_dg = A.inverse()
    S_f = build_Sf(nq, marked)
    S_0 = build_S0(nq)

    qc = QuantumCircuit(nq, name="Q")
    qc.compose(S_f, range(nq), inplace=True)
    qc.append(A_dg, range(nq))
    qc.compose(S_0, range(nq), inplace=True)
    qc.append(A, range(nq))
    return qc


# =========================
# QAE (kanonski QPE nad Q)
# =========================
def qae_joint_probs(nq: int, t: int, amp: np.ndarray, marked: List[int]) -> np.ndarray:
    """
    Vraća |ψ_joint|^2 dužine 2^(nq+t), redosled bit-ova: kao Qiskit-ov little-endian,
    sa state registrom na nižim pozicijama i ancilla registrom na višim.
    """
    state = QuantumRegister(nq, name="s")
    anc = QuantumRegister(t, name="a")
    qc = QuantumCircuit(state, anc)

    # A na state registru
    A = StatePreparation(amp.tolist())
    qc.append(A, state)

    # H^⊗t na ancillama
    qc.h(anc)

    # Controlled Q^(2^j)
    Q = build_Q(nq, amp, marked).to_gate()
    for j in range(t):
        power = 1 << j
        cQ_pow = Q.power(power).control(1)
        qc.append(cQ_pow, [anc[j]] + list(state))

    # Inverzni QFT na ancillama
    iqft = QFT(num_qubits=t, do_swaps=True, inverse=True)
    qc.append(iqft, anc)

    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2
    s = float(p.sum())
    return p / s if s > 0 else p


def marginal_state_probs(joint: np.ndarray, nq: int, t: int) -> np.ndarray:
    """Margina preko ancilla registra: vrati dim=2^nq verovatnoće state registra."""
    dim_s = 2 ** nq
    dim_a = 2 ** t
    mat = joint.reshape(dim_a, dim_s)  # [ancilla][state] (Qiskit little-endian za sv.data)
    return mat.sum(axis=0)


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


def qae_state_probs(H: np.ndarray, nq: int, t: int, M: int) -> np.ndarray:
    amp = amplitude_input(H, nq)
    marked = top_m_indices(amp, M)
    joint = qae_joint_probs(nq, t, amp, marked)
    p = marginal_state_probs(joint, nq, t)
    s = float(p.sum())
    return p / s if s > 0 else p


# =========================
# Determ. grid-optimizacija (nq, t, M) po meri cos(bias, freq_csv)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    f_csv_n = f_csv / float(f_csv.sum() or 1.0)
    best = None
    for nq in GRID_NQ:
        for t in GRID_T:
            for M in GRID_M:
                try:
                    p = qae_state_probs(H, nq, t, M)
                    b = bias_39(p)
                    score = cosine(b, f_csv_n)
                except Exception:
                    continue
                key = (score, -nq, -t, -M)
                if best is None or key > best[0]:
                    best = (key, dict(nq=nq, t=t, M=M, score=score))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q8 NN (QAE — Quantum Amplitude Estimation): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| t (ancilla):", best["t"],
        "| M (marked):", best["M"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    p = qae_state_probs(H, best["nq"], best["t"], best["M"])
    pred = pick_next_combination(p)
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q8 NN (QAE — Quantum Amplitude Estimation): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 5 | t (ancilla): 3 | M (marked): 10 | cos(bias, freq_csv): 0.853051
predikcija NEXT: (7, 19, 22, 24, 27, 28, 31)
"""



"""
Q8_NeuralNetwork_Amplitude.py — tehnika: QAE (Quantum Amplitude Estimation, BHMT kanonski)

Učita CEO CSV. Napravi amplitude-enkodovan ulaz dužine 2^nq (blok-frekvencije, L2-normalizovano).
Definiše „dobru“ podskupinu: TOP-M bazna stanja po verovatnoći (|amp|²).
Izgradi Grover-operator Q = A · S₀ · A† · S_f:
A = StatePreparation(amp), A† = njen inverz,
S_f = Diagonal(±1) oracle na marked,
S₀ = 2|0⟩⟨0| - I.
Pokrene kanonski QPE (Brassard-Høyer-Mosca-Tapp, 1998):
t ancilla qubit-a u |+⟩^⊗t,
za j = 0..t-1 kontrolisano Q^{2^j} sa ancilla[j],
inverzni QFT na ancillama.
Uzme marginal nad state registrom iz zajedničkog Statevector-a → bias_39 → NEXT.
Deterministička grid-optimizacija (nq, t, M) po meri cos(bias, freq_csv).

Tehnike:
QAE sa kontrolisanim Grover-operatorom i QFT-om (referentna formulacija).
Q = A S₀ A† S_f ulazi sa CSV-derivisanim A — „data-aware“ Grover.
Egzaktan zajednički Statevector simulator (nema uzorkovanja).
QPE rezolucija t ancilla qubit-a.

Prednosti:
Najstrukturisanija tehnika u seriji: pravi QPE + pravi Grover operator nad pripremljenim stanjem.
Kombinuje ideju state-prep-a (kao QCNN/QRC/QAM) sa amplitude amplifikacijom (kao Grover/KWS) u jedan čistiji sklop.
Marginal nad state registrom posle QPE nosi deo informacije o fazi θ → daje drugačiju distribuciju od običnog Grover-a.
Deterministički, bez klasičnog optimizera.

Nedostaci:
Najskuplji u seriji: ukupno nq + t qubit-a, plus kontrolisani Q^{2^j} — zbog toga je grid sužen (nq ≤ 6, t ≤ 5).
StatePreparation i .control().power() daju guste unitarne matrice u Qiskit-ovoj dekompoziciji — praktično tipično samo za simulator, ne za realni hardver.
Suštinski izlaz QAE-a je a² = sin²(θ) (jedna skalarna veličina) — ovde se umesto toga koristi marginal nad state registrom, što je razumna, ali ne kanonski-interpretabilna veličina.
Kao i ranije, mera optimizacije cos(bias, freq_csv) je pristrana ka reprodukciji marginale.
mod-39 readout i dalje meša stanja.
"""
