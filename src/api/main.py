from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import uvicorn
import logging

from .schemas import *
from .routes import optimization, simulation, monitoring

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Train Traffic Control API",
    description="API for optimizing train traffic control using AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(optimization.router, prefix="/optimization", tags=["Optimization"])
app.include_router(simulation.router, prefix="/simulation", tags=["Simulation"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])

@app.get("/")
async def root():
    return {
        "message": "AI Train Traffic Control System API",
        "version": "1.0.0",
        "endpoints": {
            "optimization": "/optimization",
            "simulation": "/simulation",
            "monitoring": "/monitoring"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
