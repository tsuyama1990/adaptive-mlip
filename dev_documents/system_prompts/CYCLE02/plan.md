1. **Define Pydantic Models for Compiler (Schema-First):**
   - We will create `src/pyacemaker/domain_models/compiler.py` to contain the `SemanticCompiler` logic. The compiler doesn't define schemas directly, but relies on existing schemas like `IntentRequest` (from `scenario.py` or `gui_schema.py`) and maps to `PyAceConfig` and nested objects (`MDConfig`, `TrainingConfig`, `WorkflowConfig`, etc).
   - We will define the custom exception `CompilerError` in `src/pyacemaker/core/exceptions.py`.

2. **Test-Driven Development (TDD) for Compiler:**
   - Create `tests/unit/test_compiler.py` with mock `IntentRequest`s.
   - Tests will assert correct topological sorting, valid translation of nodes, logic error detection (e.g., active learning before structure), and correct injection of default parameters (`soft_start_steps`, `integration_timestep`, `dft_convergence_threshold`, etc.).

3. **Implementation of Semantic Compiler (`compiler.py`):**
   - Implement `CompilerError`.
   - Implement `SemanticCompiler` class in `src/pyacemaker/domain_models/compiler.py` (or `src/pyacemaker/core/compiler.py` depending on structure). It will feature a `compile` method.
   - topological sort algorithm using Kahn's algorithm or similar to handle dependencies, raising `CompilerError` on cycles or unsupported structures (e.g. branching parallel active learning loops).
   - Match/case factory to process nodes `INITIAL_STRUCTURE`, `MACE_TRAINING`, `ACTIVE_LEARNING_LOOP`.
   - "Intelligent defaults" mapping functions based on `target_material` to pre-fill parameters like mass, timestep, EAM potential placeholders, etc.
   - Construct and return a complete `PyAceConfig` object.

4. **Integration with API Gateway:**
   - We assume there is an API route or we need to define it. (The spec mentions `POST /api/v1/intent/compile` in `api/routes/workflows.py`).
   - Create/modify `src/pyacemaker/api/routes/workflows.py` to add `compile_intent` endpoint. It receives `IntentRequest`, runs `SemanticCompiler.compile(intent)`, catches `CompilerError` to raise `HTTPException(400)`, and returns the `PyAceConfig` as a dict.
   - Create `tests/integration/test_api_endpoints.py` using `fastapi.testclient.TestClient` to verify the endpoint behaves properly, checking JSON payload schemas and `400 Bad Request` structure.

5. **UAT (User Acceptance Testing):**
   - Modify/create `tests/uat/test_cycle02_uat.py` handling Scenario A, B, and C as described in `UAT.md`. This will use the FastAPI test client or pure function calls to ensure the pipeline is complete.

6. **Refinement, Linting, Pre-commit:**
   - Run `uv run ruff check .`, `uv run ruff format .`, `uv run mypy .`.
   - Run `pytest` locally and update `test_execution_log.txt`.
   - Ensure "Pre Commit Steps" instruction are met.
