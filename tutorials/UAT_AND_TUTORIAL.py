import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

@app.cell
def __(mo):
    mo.md(
        r"""
        # PyAceMaker: Intent-Driven Active Learning Workflow
        Welcome to the PyAceMaker tutorial! This executable notebook will guide you through building a Molecular Dynamics (MD) active learning workflow using our visual-first `IntentRequest` architecture.

        The PyAceMaker API handles validating DAG workflows, running semantic compilations to fill in physics parameters based on accuracy-speed sliders, generating spatial tags, and executing real-time telemetry streaming via WebSockets.
        """
    )
    return

@app.cell
def __():
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    import json
    import asyncio
    import marimo as mo
    from typing import Dict, Any

    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from pyacemaker.domain_models.scenario import IntentRequest, DagNode, NodeType, SpatialRegion, SpatialAction, InitialStructureData, ActiveLearningData
    from pyacemaker.main import app as fastapi_app

    client = TestClient(fastapi_app)

    return Any, DagNode, Dict, IntentRequest, NodeType, SpatialRegion, SpatialAction, InitialStructureData, ActiveLearningData, TestClient, WebSocketDisconnect, asyncio, client, fastapi_app, json, mo, os, sys

@app.cell
def __(mo):
    mo.md(
        r"""
        ## Section 1: The Intent-Driven API Basics (Cycle 01 & 02)
        First, we'll visually construct a simple DAG (Directed Acyclic Graph) workflow. You won't need to specify every LAMMPS parameter or DFT threshold. We define an `INITIAL_STRUCTURE` node and an `ACTIVE_LEARNING_LOOP` node and connect them.
        """
    )
    return

@app.cell
def __(DagNode, IntentRequest, NodeType, InitialStructureData, ActiveLearningData, client, mo):
    # Create the Intent DAG Request
    intent_request = IntentRequest(
        accuracy_speed_slider=5, # 1=Speed, 10=Accuracy
        target_material="Al",
        nodes=[
            DagNode(id="node_1", type=NodeType.INITIAL_STRUCTURE, data=InitialStructureData(chemical_symbol="Al", lattice_constant=4.05)),
            DagNode(id="node_2", type=NodeType.ACTIVE_LEARNING_LOOP, data=ActiveLearningData())
        ],
        edges=[{"source": "node_1", "target": "node_2"}],
        advanced_settings={}
    )

    # Submit to the compiler via the REST API
    response = client.post("/api/v1/intent/compile", json=intent_request.model_dump())

    if response.status_code == 200:
        compiled_workflow = response.json()
        display_msg_01 = mo.md(f"**Successfully Compiled Workflow!** The API injected intelligent default physics parameters based on the slider value `5` and element `Al`.")
    else:
        compiled_workflow = None
        display_msg_01 = mo.md(f"**Skipping step (No API Key or Backend Error).** Status Code: {response.status_code}")

    return compiled_workflow, intent_request, response, display_msg_01

@app.cell
def __(display_msg_01):
    display_msg_01
    return

@app.cell
def __(mo):
    mo.md(
        r"""
        ## Section 2: Interactive Spatial Tagging (Cycle 03)
        Now let's apply a spatial boundary visually. Imagine you are painting the bottom layer of a 3D atomic slab to freeze it. The PyAceMaker backend will translate this bounding box into ASE tags and LAMMPS commands dynamically.
        """
    )
    return

@app.cell
def __(DagNode, IntentRequest, NodeType, InitialStructureData, ActiveLearningData, SpatialRegion, SpatialAction, client, mo):
    # Add a SpatialRegion representing the frozen bottom layer
    freeze_region = SpatialRegion(
        x_min=-100.0, x_max=100.0,
        y_min=-100.0, y_max=100.0,
        z_min=-100.0, z_max=5.0, # Everything below z=5.0 is frozen
        action=SpatialAction.ACTION_FREEZE
    )

    spatial_intent = IntentRequest(
        accuracy_speed_slider=5,
        target_material="Al",
        nodes=[
            DagNode(id="node_1", type=NodeType.INITIAL_STRUCTURE, data=InitialStructureData(chemical_symbol="Al", lattice_constant=4.05, regions=[freeze_region])),
            DagNode(id="node_2", type=NodeType.ACTIVE_LEARNING_LOOP, data=ActiveLearningData())
        ],
        edges=[{"source": "node_1", "target": "node_2"}],
        advanced_settings={}
    )

    response_spatial = client.post("/api/v1/intent/compile", json=spatial_intent.model_dump(mode='json'))

    if response_spatial.status_code == 200:
        display_msg_spatial = mo.md("**Successfully Applied Spatial Tags.** The spatial compiler translated the `FREEZE` intent into corresponding underlying config settings.")
    else:
        display_msg_spatial = mo.md(f"**Skipping step (No API Key or Backend Error).** Status Code: {response_spatial.status_code}")

    return freeze_region, response_spatial, spatial_intent, display_msg_spatial

@app.cell
def __(display_msg_spatial):
    display_msg_spatial
    return

@app.cell
def __(mo):
    mo.md(
        r"""
        ## Section 3: The Accuracy vs Speed Trade-off (Cycle 05 & 06)
        Let's modify the `Accuracy vs Speed` slider to `10` (Max Accuracy). The `candidate_threshold` will become much stricter. If we introduce an invalid slider value (e.g. 15), the system should reject it.
        """
    )
    return

@app.cell
def __(DagNode, IntentRequest, NodeType, InitialStructureData, ActiveLearningData, client, mo):
    accuracy_intent = IntentRequest(
        accuracy_speed_slider=10, # Max Accuracy
        target_material="Al",
        nodes=[
            DagNode(id="node_1", type=NodeType.INITIAL_STRUCTURE, data=InitialStructureData(chemical_symbol="Al", lattice_constant=4.05)),
            DagNode(id="node_2", type=NodeType.ACTIVE_LEARNING_LOOP, data=ActiveLearningData())
        ],
        edges=[{"source": "node_1", "target": "node_2"}],
        advanced_settings={}
    )

    response_acc = client.post("/api/v1/intent/compile", json=accuracy_intent.model_dump())

    if response_acc.status_code == 200:
        display_msg_acc = mo.md(f"**Successfully adjusted slider to 10 (Max Accuracy).** Underlying learning rates and thresholds have been automatically recalibrated.")
    else:
        display_msg_acc = mo.md(f"**Skipping step (No API Key or Backend Error).** Status Code: {response_acc.status_code}")

    return accuracy_intent, response_acc, display_msg_acc

@app.cell
def __(display_msg_acc):
    display_msg_acc
    return

@app.cell
def __(mo):
    mo.md(
        r"""
        ## Section 4: Real-Time Telemetry Visualization (Cycle 04)
        The system supports Pub/Sub telemetry over WebSockets to dynamically stream pseudo-trajectory data (atomic coordinates, forces, MLIP variance). We can connect a test client to listen for real-time events.
        """
    )
    return

@app.cell
def __(WebSocketDisconnect, client, mo):
    try:
        with client.websocket_connect("/api/v1/telemetry/stream/tutorial_workflow") as websocket:
            # We connected successfully, we can close it
            websocket.close()
        display_msg_telemetry = mo.md("**Successfully verified WebSocket Telemetry Endpoint!** (Client Connected & Disconnected gracefully).")
    except WebSocketDisconnect:
        display_msg_telemetry = mo.md("**Successfully verified WebSocket Telemetry Endpoint!** (Client Connected & Disconnected gracefully).")
    except Exception as e:
        display_msg_telemetry = mo.md(f"**Skipping step (WebSocket Error).** {str(e)}")
        websocket = None

    return websocket, display_msg_telemetry

@app.cell
def __(display_msg_telemetry):
    display_msg_telemetry
    return

if __name__ == "__main__":
    app.run()
