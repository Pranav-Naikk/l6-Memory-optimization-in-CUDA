# Experiment results evidence

**Status:** Successful execution captured on 17 November 2026 using an NVIDIA GeForce RTX 5060 Ti (compute capability 12.0).

Measured output from the captured run:

| Measurement | Result |
|---|---:|
| Uncoalesced transpose, N=4096 | 0.790 ms |
| Uncoalesced effective bandwidth | 169.8 GB/s |
| Coalesced transpose, N=4096 | 0.412 ms |
| Coalesced effective bandwidth | 325.8 GB/s |
| Coalesced transpose speedup | 1.92× |
| Naive matrix multiplication, N=1024 | 1.421 ms |
| Naive matrix multiplication throughput | 1510.8 GFLOPS |
| Transpose verification | PASS for both kernels |
| Matrix multiplication verification | PASS |

The original terminal capture is stored as `execution_output.png` in this directory and is embedded in the repository's root README.

**Repeatability note:** This screenshot documents one captured run. The lab manual asks students to run the program at least three times and record the repeated measurements. Additional runs should be appended to the results if a repeated-run average is required.
