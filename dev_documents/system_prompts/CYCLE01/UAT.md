# Cycle 01: Establish Foundation API and Schema Structure - UAT

## Test Scenarios
The primary user acceptance testing scenarios for Cycle 01 are inherently focused on proving the absolute robustness and flawless validation of the Intent-Driven GUI's new API backend without triggering the deeper active learning execution loops. The UAT simulates the precise, high-volume interactions expected from the React frontend, validating the absolute integrity of the `extra="forbid"` schema protections. The Marimo notebook exclusively simulates the `POST` payload injection processes, proving that non-expert engineers can send simplified semantic requests successfully. This ensures the foundation is rock solid before we add complex compilation logic in the subsequent cycles. We must guarantee that no malformed data can ever reach the core orchestration engine.

* **SCENARIO-01-A [Priority: High] - Successful Intent Payload Processing:** The UAT must simulate a user completing the simplified visual configuration. The interface successfully injects a fully valid JSON payload containing a Directed Acyclic Graph (DAG) representing a logical flow (e.g., Initial Structure -> Active Learning) and a semantic slider value (e.g., Accuracy: 8) to the REST API. This test must prove that the API accepts the minimal required information without demanding any complex LAMMPS configurations or advanced hyperparameter tuning from the user, fulfilling the primary goal of the Intent-Driven framework. The scenario involves the simulated user explicitly selecting a base metallic system—for example, a Platinum crystal—and setting a moderate desire for accuracy over speed. The simulated frontend constructs a JSON payload consisting of an `INITIAL_STRUCTURE` node with the structure's identifiers, connected via a directed edge to an `ACTIVE_LEARNING_LOOP` node. Crucially, the payload contains absolutely no information regarding the learning rate of the MLIP, the integration time step of the molecular dynamics simulation, or the specific DFT convergence criteria required by the Quantum ESPRESSO backend. The user simply submits this minimal representation of their physical intent. The backend system receives this payload at the `/api/v1/intent/compile` endpoint, parses the JSON, validates the DAG structure, and returns a successful HTTP 200 OK status code. This confirms the system successfully abstracted the complexities of active learning and correctly parsed the fundamental configuration logic. The returned payload mirrors the valid input, demonstrating the system successfully stored and acknowledged the configuration state without errors or the need for the user to understand the underlying Python `WorkflowConfig` mechanics. This first scenario acts as the baseline confirmation that the FastAPI endpoints are correctly bound, the CORS policies allow the React application to connect, and the foundational Pydantic schemas correctly marshal valid JSON into strongly typed Python objects without raising unnecessary validation exceptions. It establishes the "happy path" that all subsequent cycles will build upon.

* **SCENARIO-01-B [Priority: High] - Strict Rejection of Advanced Parameters:** The UAT simulates a malicious or highly confused user attempting to manually explicitly insert internal core engine parameters (e.g., `lammps_command_string`, `learning_rate_min`, or `threshold_call_dft`) directly into the Intent payload, specifically bypassing the intended visual slider abstraction. The API must and immediately respond with an explicitly detailed 422 Unprocessable Entity error, proving the effectiveness of the Anti-Corruption Layer and the `extra="forbid"` rule. The system must protect itself from configuration pollution. In this scenario, the user connects to the API endpoint and constructs a payload that resembles a valid DAG structure. However, they intentionally embed unsupported keys directly into the payload object—for example, manually injecting a `"fix 1 all nve"` command into the active learning node data. This action simulates a user attempting to bypass the visual interface to execute custom, potentially dangerous simulation logic. Upon submitting the request via `POST /api/v1/intent/compile`, the Pydantic schema validation layer intercepts the payload before it ever touches the core application logic. The validation layer recognizes that the `lammps_command_string` key is forbidden by the schema definition. The system immediately rejects the request, generating an HTTP 422 Unprocessable Entity status code. The response body contains a precise error array, clearly identifying the invalid keys and the exact nested path within the JSON object where the violation occurred. The simulated frontend receives this error, demonstrating that the backend is robustly secured against configuration injection attacks and user error, proving the integrity of the Intent-Driven design philosophy. This scenario is crucial for demonstrating that the system adheres to the principle of "fail-fast," refusing to process any data that falls outside the defined boundaries of the GUI schema, thereby preventing unpredictable behavior downstream in the physical simulation engines.

* **SCENARIO-01-C [Priority: Medium] - Validation of DAG Node Enumerations:** The UAT will simulate the frontend sending a DAG structure containing an invalid or unrecognized node type (e.g., `{ "type": "MAGIC_NODE" }`). The API must reject this payload, proving that the enumeration constraints defined in the Pydantic schemas are actively enforced and that the system will only process known, safe workflow states. In this scenario, the simulated user is experimenting with the interface and attempts to construct a workflow utilizing a node type that has not been implemented or supported by the backend compiler. They submit a payload containing a node with the `type` attribute set to `"QUANTUM_MAGIC_NODE"`. The FastAPI endpoint receives the payload and begins the validation process. The Pydantic model enforcing the `NodeType` enumeration evaluates the attribute and determines it is not a member of the allowed set of node definitions (e.g., `INITIAL_STRUCTURE`, `MACE_TRAINING`, `ACTIVE_LEARNING_LOOP`). The validation immediately fails, generating an HTTP 422 Unprocessable Entity status code. The detailed error response specifically highlights the invalid enumeration value and provides a list of the acceptable node types to the user. This scenario confirms that the backend cannot be forced into undefined states by unpredictable frontend behavior, guaranteeing that only physically meaningful simulation sequences are accepted for compilation and execution. Furthermore, we will test the case where a user submits a valid node type but omits a required field within that node's specific data structure (e.g., submitting an `INITIAL_STRUCTURE` node but forgetting to provide the `chemical_symbol`). The API must similarly reject this payload with a 422 error, explicitly stating that the `chemical_symbol` field is required, ensuring the backend always receives complete and contextually valid information.

## Behavior Definitions
These behavior definitions map to the exact User Test Scenarios, specifically defining the deterministic Gherkin-style assertions expected from the PyAceMaker GUI backend. This formalization ensures that the validation logic is testable, repeatable, and clearly understood by both developers and stakeholders reviewing the architectural implementation. By clearly defining the expected inputs and the guaranteed outputs, we establish a robust contract for the API layer that will govern all future feature development.

**Feature: Intent-Driven API Payload Validation and Anti-Corruption Layer**

**Scenario 1: Successful Validation of Simplified Intent Request**
**GIVEN** the PyAceMaker FastAPI backend is fully running in development mode without access to the actual physics engines
**AND** the simulated user intentionally avoids configuring complex thresholds, relying on the visual Intent slider abstraction
**WHEN** the simulated React frontend explicitly sends a precise `POST /api/v1/intent/compile` request containing a perfectly valid, minimal JSON payload:
```json
{
 "accuracy_speed_slider": 5,
 "target_material": "Pt",
 "nodes": [
 {
 "id": "node_001",
 "type": "INITIAL_STRUCTURE",
 "data": { "chemical_symbol": "Pt", "lattice_constant": 3.92 }
 },
 {
 "id": "node_002",
 "type": "ACTIVE_LEARNING_LOOP",
 "data": {}
 }
 ],
 "edges": [
 {"source": "node_001", "target": "node_002"}
 ]
}
```
**THEN** the API gateway responds immediately with an HTTP 200 OK status code
**AND** explicitly returns a JSON response confirming the safe acceptance and structural integrity of the request, containing a `status` of `"success"` and a `node_count` of 2
**AND** the underlying core orchestration logic is not invoked, confirming the endpoint successfully handled the unmarshalling and validation independently.

**Scenario 2: Absolute Rejection of Unauthorized Engine Directives (Config Injection)**
**GIVEN** the PyAceMaker FastAPI backend is fully running and bound to a specific port waiting for connections
**AND** the user specifically attempts to manually inject highly advanced core internal parameters directly into the graphical JSON payload
**WHEN** the simulated frontend explicitly sends a `POST /api/v1/intent/compile` request containing forbidden configuration keys:
```json
{
 "accuracy_speed_slider": 5,
 "target_material": "Pt",
 "lammps_command_string": "fix 1 all nve",
 "learning_rate": 0.001,
 "nodes": []
}
```
**THEN** the strict Pydantic `extra="forbid"` configuration absolutely triggers the internal validation failure mechanism
**AND** the API gateway responds immediately with an HTTP 422 Unprocessable Entity status code
**AND** the detailed validation exception array returned in the JSON body explicitly clearly identifies the forbidden `lammps_command_string` and `learning_rate` fields, proving the system actively prevents internal state corruption by unauthorized configuration injections.

**Scenario 3: Rejection of Out-of-Bounds Intent Parameters and Invalid Types**
**GIVEN** the PyAceMaker FastAPI backend is active and waiting for connections
**AND** the user attempts to bypass the 1-10 boundary logic of the visual slider or provide an invalid string where an integer is expected
**WHEN** the simulated frontend explicitly sends a `POST /api/v1/intent/compile` request containing an out-of-bounds or mis-typed value:
```json
{
 "accuracy_speed_slider": 15,
 "target_material": "Pt",
 "nodes": []
}
```
**THEN** the Pydantic numerical constraints intercept the request during the unmarshalling phase
**AND** the API gateway immediately returns an HTTP 422 Unprocessable Entity status code
**AND** the response body clearly indicates that the slider value must be less than or equal to 10
**AND** if the payload instead contained `"accuracy_speed_slider": "high"`, the gateway similarly rejects it, returning a 422 error explicitly stating that an integer is required, confirming the strict type-casting rules.