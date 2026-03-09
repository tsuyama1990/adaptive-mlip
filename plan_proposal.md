1. **[ARCHITECTURE] Refactor `BasePolicy.generate` interface to use `PolicyContext`**
   - Create a Pydantic model `PolicyContext` in `src/pyacemaker/domain_models/structure.py` to encapsulate `engine`, `potential`, `thresholds`, `cutout_config`, and `loop_strategy`.
   - Update the `BasePolicy.generate` method signature in `src/pyacemaker/core/base.py` to use `context: PolicyContext | None = None`.
   - Update all policy implementations (`SafeBasePolicy`, `ColdStartPolicy`, `MDMicroBurstPolicy`, `NormalModePolicy`, `CompositePolicy`, `DefectPolicy`, `RattlePolicy`, `StrainPolicy`) in `src/pyacemaker/core/policy.py` to match the new signature and extract parameters from the context.
   - Update the callers of `policy.generate` in `src/pyacemaker/core/generator.py` to package parameters into a `PolicyContext` and pass it.

2. **[SECURITY] Change `ThreadPoolExecutor` to `ProcessPoolExecutor` in `DFTManager`**
   - In `src/pyacemaker/core/oracle.py`, modify `DFTManager._compute_single` to use `concurrent.futures.ProcessPoolExecutor` instead of `ThreadPoolExecutor` to execute CPU-bound DFT calculations. This avoids GIL bottlenecks and thread pool exhaustion vulnerabilities.

3. **[SECURITY] Implement strict validation in `MACEManager.__init__`**
   - In `src/pyacemaker/core/oracle.py`, modify `MACEManager.__init__` to strictly validate `model_path` existence and prevent path traversal using `os.path.realpath` and checking `is_file()` (canonicalizing path as per memory best practices).

4. **[ARCHITECTURE] Decouple `TieredOracle` from concrete implementations**
   - In `src/pyacemaker/core/oracle.py`, change `TieredOracle.__init__` to accept `BaseOracle` interface parameters instead of the concrete `MACEManager` and `DFTManager` classes, fulfilling the dependency inversion principle.

5. **[SECURITY] Implement robust command execution validation in `run_command`**
   - In `src/pyacemaker/utils/process.py`, enhance `run_command` to explicitly use `shutil.which(cmd[0])` before executing to ensure the command exists and is an executable (as per memory).
   - Also, we will strengthen the shell injection check and add tests demonstrating shell injections (like `&&`, `||`, `$(command)`) are blocked (and testing that `shell=True` if added by a user by accident would be rejected, though `run_command` currently hardcodes `shell=False`).

6. **Pre-commit Steps**
   - Ensure proper testing, verification, review, and reflection are done by calling the pre-commit checks and running the local test suite using `pytest`.
