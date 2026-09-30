"""FastAPI routes for the local evaluation dashboard."""

from fastapi import APIRouter, HTTPException

from app.evaluations.schemas import EvaluationCreate
from app.evaluations.service import evaluation_service


router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("")
def create_evaluation(request: EvaluationCreate):
    """Start one background evaluation using validated options."""

    try:
        return evaluation_service.create(request.model_dump())
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("")
def list_evaluations():
    """List saved runs without large chunk evidence."""

    return {"evaluations": evaluation_service.list()}


@router.get("/{run_id}")
def get_evaluation(run_id: str):
    """Return progress, logs, partial metrics, and question results."""

    state = evaluation_service.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Evaluation was not found")
    return state


@router.post("/{run_id}/cancel")
def cancel_evaluation(run_id: str):
    """Ask a running job to stop after its current provider call."""

    try:
        return evaluation_service.cancel(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Evaluation was not found") from error


@router.post("/{run_id}/resume")
def resume_evaluation(run_id: str):
    """Continue a paused or cancelled job from its checkpoint."""

    try:
        return evaluation_service.resume(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Evaluation was not found") from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
