from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from enum import Enum

class TrainType(str, Enum):
    EXPRESS = "express"
    PASSENGER = "passenger"
    SUBURBAN = "suburban"
    FREIGHT = "freight"
    MAINTENANCE = "maintenance"

class TrainPriority(int, Enum):
    HIGHEST = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    LOWEST = 5

class TrainStatus(str, Enum):
    SCHEDULED = "scheduled"
    DELAYED = "delayed"
    CANCELLED = "cancelled"
    RUNNING = "running"
    COMPLETED = "completed"

class Train(BaseModel):
    id: str
    train_number: str
    name: str
    type: TrainType
    priority: TrainPriority
    max_speed_kmph: float
    length_m: float
    weight_tonnes: float
    
    # Route information
    source_station_id: str
    destination_station_id: str
    route_station_ids: List[str]
    
    # Timing information
    scheduled_departure: datetime
    scheduled_arrival: datetime
    actual_departure: Optional[datetime] = None
    actual_arrival: Optional[datetime] = None
    
    # Current state
    status: TrainStatus = TrainStatus.SCHEDULED
    current_section: Optional[str] = None
    next_station_id: Optional[str] = None
    delay_minutes: int = 0
    
    def update_position(self, section_id: str, next_station_id: str):
        self.current_section = section_id
        self.next_station_id = next_station_id
    
    def calculate_delay(self) -> int:
        if self.actual_departure and self.scheduled_departure:
            delay = (self.actual_departure - self.scheduled_departure).total_seconds() / 60
            return max(0, int(delay))
        return 0
