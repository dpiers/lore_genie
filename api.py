
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import SET_CONFIGS, monte_carlo_histogram_for_card, inspect_rarity

app = FastAPI(title="Lorcana Odds API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_ROOT = Path(__file__).parent / "web"
app.mount("/web", StaticFiles(directory=WEB_ROOT, html=True), name="web")


@app.get("/", include_in_schema=False)
def root():
    index = WEB_ROOT / "mc_visualizer.html"
    if index.exists():
        return FileResponse(index)
    return RedirectResponse(url="/docs")


class SetInfo(BaseModel):
    id: int
    name: str
    rarities: List[str]


@app.get("/sets", response_model=List[SetInfo])
def get_sets() -> List[SetInfo]:
    result: List[SetInfo] = []
    for n, cfg in SET_CONFIGS.items():
        rarities = list(cfg.rarity_card_counts.keys())
        result.append(SetInfo(id=n, name=cfg.name, rarities=rarities))
    return result


class MCBatchRequest(BaseModel):
    set_number: int
    rarity: str
    packs: int
    trials: int
    max_k: Optional[int] = 10


class MCBatchResponse(BaseModel):
    set_number: int
    set_name: str
    rarity: str
    packs: int
    lambda_: float
    num_trials: int
    max_k: int
    counts: List[int]
    overflow: int


@app.post("/mc-batch", response_model=MCBatchResponse)
def mc_batch(req: MCBatchRequest) -> MCBatchResponse:
    cfg = SET_CONFIGS[req.set_number]
    hist = monte_carlo_histogram_for_card(
        cfg,
        req.rarity,
        req.packs,
        req.trials,
        max_k=req.max_k or 10,
    )
    return MCBatchResponse(
        set_number=req.set_number,
        set_name=cfg.name,
        rarity=req.rarity,
        packs=req.packs,
        lambda_=hist["lambda"],
        num_trials=hist["num_trials"],
        max_k=hist["max_k"],
        counts=hist["counts"],
        overflow=hist["overflow"],
    )


class InspectRequest(BaseModel):
    set_number: int
    rarity: str
    packs: int
    k_values: Optional[List[int]] = None


@app.post("/inspect-rarity")
def api_inspect_rarity(req: InspectRequest) -> Dict:
    k_values = tuple(req.k_values) if req.k_values else (1, 4)
    return inspect_rarity(req.set_number, req.rarity, req.packs, k_values=k_values)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
