from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum

class StationType(str, Enum):
    JUNCTION = "junction"
    TERMINAL = "terminal"
    CROSSING = "crossing"
    PASSING = "passing"

class Platform(BaseModel):
    id: str
    length_m: float
    can_accommodate: List[str]  # train types
    is_occupied: bool = False
    occupied_by: Optional[str] = None

class Station(BaseModel):
    id: str
    name: str
    type: StationType
    platforms: List[Platform]
    crossing_capacity: int = 2
    is_operational: bool = True
    coordinates: Optional[Dict[str, float]] = None
    
    def get_available_platforms(self, train_type: str) -> List[Platform]:
        return [p for p in self.platforms 
                if not p.is_occupied and train_type in p.can_accommodate]
    
    def occupy_platform(self, platform_id: str, train_id: str) -> bool:
        for platform in self.platforms:
            if platform.id == platform_id and not platform.is_occupied:
                platform.is_occupied = True
                platform.occupied_by = train_id
                return True
        return False
    
    def release_platform(self, train_id: str) -> bool:
        for platform in self.platforms:
            if platform.occupied_by == train_id:
                platform.is_occupied = False
                platform.occupied_by = None
                return True
        return False
