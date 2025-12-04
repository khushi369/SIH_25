import simpy
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class DiscreteEventSimulator:
    def __init__(self, network_config: Dict):
        self.env = simpy.Environment()
        self.network = network_config
        self.trains = {}
        self.track_segments = {}
        self.stations = {}
        self.events = []
        
        # Initialize resources
        self._initialize_resources()
        
    def _initialize_resources(self):
        """Initialize SimPy resources (tracks, platforms)"""
        # Track segments as resources (only one train at a time)
        for seg_id, segment in self.network['track_segments'].items():
            self.track_segments[seg_id] = simpy.Resource(self.env, capacity=1)
        
        # Station platforms as resources
        for station_id, station in self.network['stations'].items():
            self.stations[station_id] = {
                'resource': simpy.Resource(self.env, capacity=station['platform_count']),
                'crossing_capacity': station.get('crossing_capacity', 1)
            }
    
    def add_train(self, train_id: str, schedule: Dict, 
                  priority: int = 3, speed_variation: float = 0.1):
        """Add a train to the simulation"""
        self.trains[train_id] = {
            'schedule': schedule,
            'priority': priority,
            'speed_variation': speed_variation,
            'current_position': None,
            'delay': 0,
            'events': []
        }
        
        # Start train process
        self.env.process(self._train_process(train_id))
    
    def _train_process(self, train_id):
        """Simulate a train's journey"""
        train = self.trains[train_id]
        schedule = train['schedule']
        
        route = schedule['route']  # List of (station_id, planned_arrival, planned_departure)
        
        for i, (station_id, planned_arr, planned_dep) in enumerate(route):
            # Arrive at station
            arrival_time = self._simulate_arrival(train_id, station_id, i, route)
            
            # Dwell at station
            dwell_time = self._simulate_dwell(train_id, station_id, planned_dep - planned_arr)
            
            # Depart from station
            departure_time = arrival_time + dwell_time
            
            # Record event
            self.events.append({
                'train_id': train_id,
                'station_id': station_id,
                'event_type': 'departure',
                'time': departure_time,
                'planned_time': planned_dep,
                'delay': departure_time - planned_dep
            })
            
            # If not last station, proceed to next
            if i < len(route) - 1:
                next_station_id = route[i+1][0]
                travel_time = self._simulate_travel(
                    train_id, station_id, next_station_id, 
                    route[i+1][1] - planned_dep
                )
        
        # Train completed journey
        logger.info(f"Train {train_id} completed journey")
    
    def _simulate_arrival(self, train_id, station_id, segment_index, route):
        """Simulate arrival at a station"""
        if segment_index == 0:
            # First station - start from schedule
            planned_arrival = route[segment_index][1]
            actual_arrival = self._apply_delay(planned_arrival, train_id)
        else:
            # Arrive from previous segment
            prev_station = route[segment_index-1][0]
            
            # Request track resource
            segment_id = f"{prev_station}-{station_id}"
            with self.track_segments[segment_id].request() as req:
                yield req  # Wait for track to be available
                
                # Calculate travel time with variation
                planned_travel = route[segment_index][1] - route[segment_index-1][2]
                actual_travel = self._apply_variation(planned_travel, 
                                                      self.trains[train_id]['speed_variation'])
                
                yield self.env.timeout(actual_travel.total_seconds() / 60)  # Convert to minutes
        
        # Record arrival event
        actual_time = self.env.now
        planned_time = route[segment_index][1]
        delay = actual_time - planned_time
        
        self.events.append({
            'train_id': train_id,
            'station_id': station_id,
            'event_type': 'arrival',
            'time': actual_time,
            'planned_time': planned_time,
            'delay': delay
        })
        
        self.trains[train_id]['current_position'] = station_id
        self.trains[train_id]['delay'] = delay
        
        return actual_time
    
    def _simulate_dwell(self, train_id, station_id, planned_dwell):
        """Simulate dwell time at station"""
        # Request platform resource
        with self.stations[station_id]['resource'].request() as req:
            yield req  # Wait for platform
            
            # Apply dwell time variation
            actual_dwell = self._apply_variation(planned_dwell, 0.2)  # 20% variation
            
            # Random chance of extended dwell (e.g., for maintenance, loading)
            if random.random() < 0.05:  # 5% chance
                actual_dwell += timedelta(minutes=random.randint(5, 15))
            
            yield self.env.timeout(actual_dwell.total_seconds() / 60)
            
            return actual_dwell
    
    def _simulate_travel(self, train_id, from_station, to_station, planned_travel):
        """Simulate travel between stations"""
        segment_id = f"{from_station}-{to_station}"
        
        with self.track_segments[segment_id].request() as req:
            yield req
            
            # Apply speed variation and potential delays
            actual_travel = self._apply_variation(planned_travel, 0.15)
            
            # Random delays (e.g., signal failure, weather)
            if random.random() < 0.03:  # 3% chance of delay
                actual_travel += timedelta(minutes=random.randint(5, 30))
            
            yield self.env.timeout(actual_travel.total_seconds() / 60)
            
            return actual_travel
    
    def _apply_delay(self, planned_time, train_id):
        """Apply initial delay to train"""
        base_time = planned_time
        
        # Higher priority trains have lower delay probability
        priority_factor = self.trains[train_id]['priority'] / 5  # 1-5 scale
        
        if random.random() < (0.1 * priority_factor):  # 2-10% chance based on priority
            delay_minutes = random.randint(5, 45)
            return base_time + timedelta(minutes=delay_minutes)
        
        return base_time
    
    def _apply_variation(self, duration, variation):
        """Apply random variation to duration"""
        variation_factor = random.uniform(1 - variation, 1 + variation)
        return duration * variation_factor
    
    def run(self, simulation_hours: int = 24):
        """Run the simulation"""
        logger.info(f"Starting simulation for {simulation_hours} hours")
        
        # Convert simulation hours to minutes
        simulation_minutes = simulation_hours * 60
        self.env.run(until=simulation_minutes)
        
        logger.info(f"Simulation completed. Total events: {len(self.events)}")
        
        return self._generate_report()
    
    def _generate_report(self):
        """Generate simulation report with KPIs"""
        delays = []
        throughput = defaultdict(int)
        
        for event in self.events:
            if event['event_type'] == 'arrival':
                delays.append(event['delay'])
            
            # Count trains passing through each station
            throughput[event['station_id']] += 1
        
        report = {
            'total_trains': len(self.trains),
            'total_events': len(self.events),
            'average_delay_minutes': sum(delays) / len(delays) if delays else 0,
            'max_delay_minutes': max(delays) if delays else 0,
            'throughput_per_station': dict(throughput),
            'events': self.events,
            'train_status': {tid: train for tid, train in self.trains.items()}
        }
        
        return report
