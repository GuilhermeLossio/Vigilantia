# Four-Qubit Quantum Kernel

**Status: implemented and executed.** `src/quantum/kernel.py` builds this circuit, and the first benchmark run measured it on `ibm_fez`; see [hardware runs](hardware-runs.md). The circuit is a fixed data-encoding map, not a variational classifier with trainable gate angles. The external validation uses only its exact simulation.

![Planned quantum kernel circuit](quantum-circuit.svg)

## Encoding map

Use Qiskit's functional constructor `zz_feature_map(feature_dimension=4, reps=1, entanglement="linear", alpha=2.0)`. The [official API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.zz_feature_map) defines the phase functions and supported entanglement patterns.

The input is the ordered vector of four numerical features transformed to `[0, pi/2]` by the training-fitted preprocessing described in the [benchmark protocol](benchmark-protocol.md). The frozen features are `host_entropy`, `host_label_count`, `host_digit_ratio`, and `host_hyphen_count`, one per logical qubit in that order. The range stops at pi/2 because each qubit receives a phase of `2 * x`; a range up to pi would make the smallest and largest values produce the same single-qubit phase.

Starting from `|0000>`:

1. Apply a Hadamard gate to each qubit.
2. Apply `P(2*x[i])` to qubit `i`.
3. For each pair `(0,1)`, `(1,2)`, `(2,3)`, apply `CX(i,j)`, `P(2*(pi-x[i])*(pi-x[j]))` on `j`, then `CX(i,j)`.

This defines `U(x)`. One logical feature map contains four H gates, seven phase gates, and six CX gates before transpilation or optimization. The SVG expands the logical map; the inverse block reverses the gate order and signs of phase angles. Physical gate counts and connectivity depend on the selected backend and transpilation.

## Fidelity measurement

Prepare `U(x)|0000>`, then apply `U(y).inverse()` and measure all four qubits. The ideal kernel is:

`K(x,y) = |<0000| U(y)^dagger U(x) |0000>|^2`.

Estimate it as `counts["0000"] / actual_shots`, with **1,024 requested shots** for each pair. Use independent parameter vectors for `x` and `y`; accidentally sharing parameters would collapse the comparison. The all-zero event is unchanged by bitstring endianness, but persist the measurement-register mapping.

The unoptimized pair circuit has eight H gates, fourteen phase gates, and twelve CX gates plus four measurements. These are logical counts, not hardware cost estimates. Do not add an ansatz, optimizer, ancillas, or a SWAP test in the first version.

## IBM Quantum execution

Use `QiskitRuntimeService` and a sampler from `qiskit_ibm_runtime` (`SamplerV2`, or the client-side `Sampler` that replaces it from version 0.50), authenticated against the intended Open Plan instance. Run in job mode with ISA-compatible circuits transpiled for one available backend. Backend selection and remaining quota must be recorded before any submission; no specific backend is promised by this design.

The example environment names are a proposed application contract. Future code must explicitly read and pass the token, instance, and backend to the service and sampler; copying `.env.example` does not authenticate or execute anything. Never publish credentials or saved-account files.

Local statevector/Aer checks precede hardware submission. Check normalization, ideal diagonal values near 1, symmetry, and agreement between direct state overlap and the compute-uncompute circuit for fixed inputs. Validate actual parameter binding and register parsing with the pinned package versions. The same exact statevector computation produces the training kernel and the ideal test kernel. These simulations do not satisfy the hardware classification requirement.

## Integration and limitations

The [benchmark protocol](benchmark-protocol.md) defines the hybrid design, the 336 hardware evaluations, shots, calibration, baselines, and evidence retention. The classifier is a classical SVM fitted to simulated similarities; the hardware measures the similarities used to classify the test set. One shallow linear layer keeps the initial experiment small but is not evidence of improved classification or quantum advantage.

Training and benchmark jobs run outside HTTP requests. New live quantum predictions require kernels against reference training examples and can incur queue delays. The local API will serve the classical model for live inputs and may expose saved QPU test predictions as explicitly labeled replay until a separate asynchronous inference design is implemented. A replay must identify its sample, model, and source run; it is not live QPU inference.

See the [README](../README.md), the [architecture](../Architecture.md), and the [service guide](service.md).
