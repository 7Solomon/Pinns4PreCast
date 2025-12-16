import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import core, graphs, monitoring
import src.node_system.registry

app = FastAPI(title="PINNs Node Editor API", version="1.0")

# Setup global directories
os.makedirs("content/tree_templates", exist_ok=True)
os.makedirs("content/runs", exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(core.router)
app.include_router(graphs.router)
app.include_router(monitoring.router)
