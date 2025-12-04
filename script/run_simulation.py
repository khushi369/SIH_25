#!/usr/bin/env python3
"""
Script to run simulation with different scenarios
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
from datetime import datetime
from src.simulation.discrete_event_sim import DiscreteEventSimulator
from src.utils.data_loader import DataLoader
from src.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Run train traffic simulation")
    parser.add_argument("--scenario", type=str, default="normal", 
                       choices=["normal", "congested", "disruption", "maintenance"])
    parser.add_argument("--hours", type=int, default=24, help="Simulation duration in hours")
    parser.add_argument("--output", type=str, default="simulation_results.json")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("simulation", level="DEBUG" if args.verbose else "INFO")
    
    logger.info(f"Starting simulation with scenario: {args.scenario}")
    
    # Load data
    loader = DataLoader()
    network_config = loader.load_network_config()
    
    # Adjust configuration based on scenario
    if args.scenario == "congested":
        # Increase traffic density
        network_config['traffic_density'] = 1.5
    elif args.scenario == "disruption":
        # Simulate disruption at a station
        network_config['disrupted_stations'] = ["station_3"]
    elif args.scenario == "maintenance":
        # Add maintenance blocks
        network_config['maintenance_blocks'] = [
            {"segment": "station_2-station_3", "start": 60, "duration": 120}  # minutes
        ]
    
    # Create simulator
    simulator = DiscreteEventSimulator(network_config)
    
    # Load trains
    trains = loader.load_trains(limit=50)  # Limit for performance
    
    # Add trains to simulation
    for train in trains:
        # Create schedule for each train
        schedule = {
            'route': [
                (train.source_station_id, 
                 datetime.now(), 
                 datetime.now() + timedelta(minutes=10)),
                (train.destination_station_id,
                 datetime.now() + timedelta(minutes=30),
                 datetime.now() + timedelta(minutes=32))
            ]
        }
        
        simulator.add_train(
            train_id=train.id,
            schedule=schedule,
            priority=train.priority.value,
            speed_variation=0.1
        )
    
    # Run simulation
    results = simulator.run(simulation_hours=args.hours)
    
    # Save results
    with open(args.output, 'w') as f:
        # Convert datetime objects to strings
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError("Type not serializable")
        
        json.dump(results, f, default=json_serial, indent=2)
    
    logger.info(f"Simulation completed. Results saved to {args.output}")
    
    # Print summary
    print("\n=== SIMULATION SUMMARY ===")
    print(f"Scenario: {args.scenario}")
    print(f"Duration: {args.hours} hours")
    print(f"Total Trains: {results['total_trains']}")
    print(f"Average Delay: {results['average_delay_minutes']:.1f} minutes")
    print(f"Maximum Delay: {results['max_delay_minutes']:.1f} minutes")
    print(f"Total Events: {results['total_events']}")

if __name__ == "__main__":
    main()
