from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class FieldStateSchema(BaseModel):
    id: int = Field(..., description="Field index (0-4)")
    state: str = Field(..., description="Field state: EMPTY, GROWING, MATURE")
    crop: Optional[str] = Field(None, description="Crop name if planted, else null")
    progress: int = Field(0, description="Growth progress in days")


class FarmStateSchema(BaseModel):
    day: int = Field(..., description="Current day of the season (1-30)")
    cash: float = Field(..., description="Current liquid cash available")
    debt: float = Field(..., description="Current outstanding debt")
    fields: List[FieldStateSchema] = Field(..., description="List of 5 fields")
    inventory: Dict[str, int] = Field(..., description="Harvested crops inventory count")
    market_prices: Dict[str, float] = Field(..., description="Current selling prices for crops")
    risk_forecasts: Dict[str, str] = Field(..., description="Risk forecasts (LOW, MEDIUM, HIGH)")


class ResetRequest(BaseModel):
    seed: Optional[int] = Field(None, description="Optional random seed for environment reproducibility")
    session_id: Optional[str] = Field(None, description="Optional existing session ID to reset; if null, a new session is created")


class ResetResponse(BaseModel):
    session_id: str = Field(..., description="Unique UUID identifying the farm session")
    state: FarmStateSchema = Field(..., description="Initial farm state")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional reset information")


class FarmStateResponse(BaseModel):
    session_id: str = Field(..., description="Unique UUID identifying the farm session")
    state: FarmStateSchema = Field(..., description="Current farm state")


class FarmMetricsResponse(BaseModel):
    session_id: str = Field(..., description="Unique UUID identifying the farm session")
    current_day: int = Field(..., description="Current day of the season")
    cash: float = Field(..., description="Liquid cash")
    debt: float = Field(..., description="Outstanding debt")
    inventory_value: float = Field(..., description="Value of current inventory at market prices")
    total_wealth: float = Field(..., description="Total wealth (cash + inventory value)")
    profit: float = Field(..., description="Profit relative to starting cash")
    survived: bool = Field(..., description="Whether farm is solvent")
    season_over: bool = Field(..., description="Whether season has reached maximum days")
    fields_planted: int = Field(..., description="Number of fields currently planted")
    fields_mature: int = Field(..., description="Number of fields ready for harvest")
