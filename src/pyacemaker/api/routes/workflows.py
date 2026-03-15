from fastapi import APIRouter, HTTPException

from pyacemaker.core.exceptions import CompilerError
from pyacemaker.domain_models.compiler import SemanticCompiler
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.scenario import IntentRequest

router = APIRouter(prefix="/api/v1/intent", tags=["intent"])


@router.post("/compile")
async def compile_intent(intent: IntentRequest) -> PyAceConfig:
    """
    Compiles a high-level visual IntentRequest into a full PyAceConfig payload.
    """
    try:
        compiled_config = SemanticCompiler.compile(intent)
        return compiled_config
    except CompilerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
