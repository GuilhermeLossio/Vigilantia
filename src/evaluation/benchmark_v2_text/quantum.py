"""Actual Qiskit Statevector and Aer; this module has no provider/cloud imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from src.evaluation.benchmark_v2.core import digest, kernel, write_json


def feature_map(encoding):
    if encoding == "z":
        return z_feature_map(4, reps=1, alpha=2.0)
    if encoding == "zz":
        return zz_feature_map(4, reps=1, alpha=2.0, entanglement="linear")
    raise ValueError("Only four-qubit Z and linear ZZ maps are supported.")


def states(values, encoding, transform_hash, cache: Path | None = None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("Statevector requires finite rows of four angles.")
    identity = hashlib.sha256(
        json.dumps(
            {
                "map": encoding,
                "reps": 1,
                "alpha": 2.0,
                "transform": transform_hash,
                "shape": values.shape,
            },
            sort_keys=True,
        ).encode()
        + values.tobytes()
    ).hexdigest()
    path = None if cache is None else cache / f"{identity}.npy"
    if path is not None and path.exists() and path.with_suffix(".json").exists():
        meta = json.loads(path.with_suffix(".json").read_text())
        if digest(path) != meta["sha256"]:
            raise ValueError("Statevector cache was modified.")
        return np.load(path, allow_pickle=False)
    circuit = feature_map(encoding)
    unique, inverse = np.unique(values, axis=0, return_inverse=True)
    # Every unique state is obtained from a Qiskit circuit, not the old NumPy formula.
    output = np.asarray(
        [Statevector.from_instruction(circuit.assign_parameters(row)).data for row in unique]
    )[inverse]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, output)
        write_json(path.with_suffix(".json"), {"sha256": digest(path), "identity": identity})
    return output


def pair_template(encoding):
    circuit = feature_map(encoding)
    left, right = ParameterVector("left", 4), ParameterVector("right", 4)
    pair = circuit.assign_parameters(dict(zip(circuit.parameters, left, strict=True))).compose(
        circuit.assign_parameters(dict(zip(circuit.parameters, right, strict=True))).inverse()
    )
    pair.measure_all()
    return pair, left, right


def measured_kernel(left, right, encoding, shots, seed, batch_size, directory):
    """Row-major pairs, simulator seeds per batch, resumable raw counts with checksums."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    if any(a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all() for a in (left, right)):
        raise ValueError("Aer requires finite rows of four angles.")
    if shots < 1 or batch_size < 1:
        raise ValueError("Shots and batch size must be positive.")
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "encoding": encoding,
        "shots": shots,
        "seed": seed,
        "batch_size": batch_size,
        "left": hashlib.sha256(left.tobytes()).hexdigest(),
        "right": hashlib.sha256(right.tobytes()).hexdigest(),
    }
    manifest = directory / "identity.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Aer measurement identity changed.")
    write_json(manifest, identity)
    simulator = AerSimulator(
        method="statevector", max_parallel_threads=1, max_parallel_experiments=1
    )
    template, lp, rp = pair_template(encoding)
    compiled = transpile(template, simulator, optimization_level=1, seed_transpiler=seed)
    total = len(left) * len(right)
    estimates = np.empty(total)
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        path = directory / f"counts-{start:06d}.json"
        if path.exists() and path.with_suffix(".sha256").exists():
            saved = json.loads(path.read_text())
            if digest(path) != path.with_suffix(".sha256").read_text():
                raise ValueError("Aer counts checkpoint was modified.")
            records = saved["pairs"]
        else:
            indices = np.arange(start, stop)
            rows, columns = indices // len(right), indices % len(right)
            binds = {
                **{p: left[rows, j].tolist() for j, p in enumerate(lp)},
                **{p: right[columns, j].tolist() for j, p in enumerate(rp)},
            }
            result = simulator.run(
                compiled, parameter_binds=[binds], shots=shots, seed_simulator=seed + start
            ).result()
            records = [
                {"left": int(i), "right": int(j), "counts": result.get_counts(k)}
                for k, (i, j) in enumerate(zip(rows, columns, strict=True))
            ]
            write_json(path, {"start": start, "stop": stop, "seed": seed + start, "pairs": records})
            path.with_suffix(".sha256").write_text(digest(path))
        if len(records) != stop - start:
            raise ValueError("Incomplete Aer batch.")
        for offset, record in enumerate(records):
            index = start + offset
            if (record["left"], record["right"]) != (index // len(right), index % len(right)):
                raise ValueError("Aer pair order changed.")
            if sum(record["counts"].values()) != shots:
                raise ValueError("Aer shot count mismatch.")
            estimates[index] = record["counts"].get("0000", 0) / shots
    matrix = estimates.reshape(len(left), len(right))
    np.save(directory / "kernel.npy", matrix)
    return matrix


def cross_kernel(values, reference, encoding, transform_hash, cache):
    return kernel(states(values, encoding, transform_hash, cache), reference)
