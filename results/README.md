# Experiment results evidence

The executable has not yet been built on the target Windows PC because NVCC reported that the Microsoft C++ host compiler (`cl.exe`) is not available in PATH.

After installing the Visual Studio C++ Build Tools, compiling, and successfully running the program, save an **unedited screenshot of the actual PowerShell output** in this directory as:

`execution_output.png`

The screenshot should include:
- GPU model and compute capability.
- Uncoalesced transpose time, bandwidth, and PASS/FAIL.
- Coalesced transpose time, bandwidth, and PASS/FAIL.
- Calculated transpose speedup.
- Naive matrix multiplication time, GFLOPS, and PASS/FAIL.

Only actual values printed by the executable should be entered into the README results table. Do not substitute the illustrative sample output from the lab manual.
