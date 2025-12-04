import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List
import logging

from src.data_models.train import Train, TrainStatus
from src.optimization.milp_solver import MILPSolver
from src.simulation.discrete_event_sim import DiscreteEventSimulator

logger = logging.getLogger(__name__)

class TrafficControlDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="AI-Powered Train Traffic Control",
            page_icon="🚆",
            layout="wide"
        )
        
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'trains' not in st.session_state:
            st.session_state.trains = []
        if 'stations' not in st.session_state:
            st.session_state.stations = []
        if 'schedule' not in st.session_state:
            st.session_state.schedule = {}
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = {}
    
    def run(self):
        """Main dashboard application"""
        st.title("🚆 AI-Powered Train Traffic Control System")
        st.markdown("---")
        
        # Sidebar
        with st.sidebar:
            st.header("Control Panel")
            
            # Mode selection
            mode = st.radio(
                "Operation Mode",
                ["Real-time Control", "Planning", "Simulation", "Analytics"]
            )
            
            # Load sample data
            if st.button("📊 Load Sample Data"):
                self.load_sample_data()
            
            # Optimization settings
            st.subheader("Optimization Settings")
            horizon = st.slider("Planning Horizon (hours)", 2, 12, 6)
            time_step = st.selectbox("Time Step (minutes)", [1, 5, 10, 15], index=1)
            
            if st.button("🎯 Run Optimization"):
                with st.spinner("Optimizing schedule..."):
                    schedule = self.run_optimization(horizon, time_step)
                    st.session_state.schedule = schedule
                    st.success("Optimization complete!")
            
            # Simulation settings
            if st.button("🔄 Run Simulation"):
                with st.spinner("Running simulation..."):
                    results = self.run_simulation()
                    st.session_state.simulation_results = results
                    st.success("Simulation complete!")
        
        # Main content area
        if mode == "Real-time Control":
            self.show_real_time_control()
        elif mode == "Planning":
            self.show_planning_view()
        elif mode == "Simulation":
            self.show_simulation_view()
        else:
            self.show_analytics_view()
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        try:
            # Load sample stations
            with open('data/sample/stations.json') as f:
                stations_data = json.load(f)
                st.session_state.stations = stations_data
            
            # Load sample trains
            with open('data/sample/trains.json') as f:
                trains_data = json.load(f)
                st.session_state.trains = trains_data
            
            st.success("Sample data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
    
    def run_optimization(self, horizon: int, time_step: int) -> Dict:
        """Run optimization on current data"""
        try:
            solver = MILPSolver(horizon_hours=horizon, time_step=time_step)
            
            # Convert data to appropriate format
            trains = [Train(**t) for t in st.session_state.trains]
            stations = st.session_state.stations
            
            # Run optimization
            schedule = solver.solve(
                trains=trains,
                stations=stations,
                track_segments=[],  # Would load from data
                current_time=datetime.now()
            )
            
            return schedule
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            st.error(f"Optimization failed: {e}")
            return {}
    
    def run_simulation(self) -> Dict:
        """Run simulation on current schedule"""
        try:
            # Create network configuration
            network_config = {
                'stations': {s['id']: s for s in st.session_state.stations},
                'track_segments': self._create_track_segments()
            }
            
            simulator = DiscreteEventSimulator(network_config)
            
            # Add trains to simulation
            for train in st.session_state.trains:
                schedule = self._create_train_schedule(train)
                simulator.add_train(
                    train_id=train['id'],
                    schedule=schedule,
                    priority=train.get('priority', 3)
                )
            
            # Run simulation
            results = simulator.run(simulation_hours=24)
            
            return results
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            st.error(f"Simulation failed: {e}")
            return {}
    
    def show_real_time_control(self):
        """Display real-time control view"""
        st.header("Real-time Traffic Control")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.subheader("Current Traffic Status")
            
            # Display trains on map/timeline
            fig = self.create_time_distance_diagram()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Control Actions")
            
            # Train selection for manual control
            train_options = [f"{t['id']} - {t['name']}" 
                           for t in st.session_state.trains]
            
            selected_train = st.selectbox("Select Train", train_options)
            
            if selected_train:
                train_id = selected_train.split(" - ")[0]
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    if st.button("⏸️ Hold at Next Station", key="hold"):
                        self.hold_train(train_id)
                
                with col_b:
                    if st.button("🔄 Re-route", key="reroute"):
                        self.reroute_train(train_id)
                
                with col_c:
                    if st.button("⚡ Prioritize", key="prioritize"):
                        self.prioritize_train(train_id)
                
                # Speed control
                new_speed = st.slider("Adjust Speed (% of max)", 50, 100, 80)
                if st.button("Set Speed"):
                    self.adjust_train_speed(train_id, new_speed)
        
        with col3:
            st.subheader("KPIs")
            
            # Display key metrics
            metrics = self.calculate_kpis()
            
            for metric_name, value in metrics.items():
                st.metric(
                    label=metric_name,
                    value=f"{value:.1f}" if isinstance(value, float) else value,
                    delta=None
                )
    
    def create_time_distance_diagram(self):
        """Create time-distance diagram for trains"""
        fig = go.Figure()
        
        # Add station lines
        stations = st.session_state.stations
        for station in stations:
            fig.add_hline(
                y=station.get('distance_from_start', 0),
                line_dash="dot",
                opacity=0.3,
                annotation_text=station['name']
            )
        
        # Add train lines
        trains = st.session_state.trains
        for train in trains:
            if 'schedule' in train:
                schedule = train['schedule']
                
                # Plot train trajectory
                x_vals = []
                y_vals = []
                
                for station_id, timing in schedule.items():
                    # Convert to plot coordinates
                    station = next((s for s in stations if s['id'] == station_id), None)
                    if station:
                        x_vals.append(timing['departure'])
                        y_vals.append(station['distance_from_start'])
                
                if x_vals and y_vals:
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers',
                        name=train['name'],
                        line=dict(width=2)
                    ))
        
        fig.update_layout(
            title="Time-Distance Diagram",
            xaxis_title="Time",
            yaxis_title="Distance",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def calculate_kpis(self) -> Dict:
        """Calculate key performance indicators"""
        # This would calculate from real or simulated data
        return {
            "Throughput": 42,
            "Avg Delay (min)": 8.5,
            "Punctuality (%)": 86.2,
            "Utilization (%)": 72.4
        }
    
    def show_planning_view(self):
        """Display planning and scheduling view"""
        st.header("Schedule Planning")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Optimized Schedule")
            
            if st.session_state.schedule:
                # Display schedule as Gantt chart
                self.display_gantt_chart()
            else:
                st.info("Run optimization to see schedule")
        
        with col2:
            st.subheader("Schedule Details")
            
            # Interactive schedule editor
            self.display_schedule_editor()
    
    def show_simulation_view(self):
        """Display simulation results"""
        st.header("Simulation Results")
        
        if st.session_state.simulation_results:
            results = st.session_state.simulation_results
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trains", results.get('total_trains', 0))
            with col2:
                st.metric("Avg Delay", f"{results.get('average_delay_minutes', 0):.1f} min")
            with col3:
                st.metric("Max Delay", f"{results.get('max_delay_minutes', 0):.1f} min")
            with col4:
                total_events = results.get('total_events', 0)
                st.metric("Total Events", total_events)
            
            # Event log
            st.subheader("Event Log")
            events_df = pd.DataFrame(results.get('events', []))
            if not events_df.empty:
                st.dataframe(events_df.sort_values('time'))
            
            # Throughput chart
            st.subheader("Station Throughput")
            throughput = results.get('throughput_per_station', {})
            if throughput:
                fig = px.bar(
                    x=list(throughput.keys()),
                    y=list(throughput.values()),
                    labels={'x': 'Station', 'y': 'Trains'},
                    title="Trains per Station"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run simulation to see results")
    
    def show_analytics_view(self):
        """Display analytics and what-if scenarios"""
        st.header("Analytics & What-If Scenarios")
        
        tab1, tab2, tab3 = st.tabs(["Performance", "What-If", "Forecast"])
        
        with tab1:
            st.subheader("Performance Analytics")
            # Add performance charts here
        
        with tab2:
            st.subheader("What-If Scenarios")
            
            scenario = st.selectbox(
                "Select Scenario",
                ["Infrastructure Failure", "Weather Delay", "Special Event", "Maintenance Block"]
            )
            
            if scenario == "Infrastructure Failure":
                failed_station = st.selectbox(
                    "Select Station to Fail",
                    [s['name'] for s in st.session_state.stations]
                )
                
                failure_duration = st.slider("Failure Duration (hours)", 1, 12, 4)
                
                if st.button("Simulate Failure"):
                    self.simulate_failure(failed_station, failure_duration)
            
            # Add other scenario configurations
        
        with tab3:
            st.subheader("Demand Forecast")
            # Add forecasting visualizations
    
    def _create_track_segments(self):
        """Create track segments from stations"""
        segments = {}
        stations = st.session_state.stations
        
        for i in range(len(stations) - 1):
            from_station = stations[i]
            to_station = stations[i+1]
            
            segment_id = f"{from_station['id']}-{to_station['id']}"
            segments[segment_id] = {
                'id': segment_id,
                'from_station': from_station['id'],
                'to_station': to_station['id'],
                'length_km': abs(to_station.get('distance_from_start', 0) - 
                                from_station.get('distance_from_start', 0)),
                'min_travel_time': 10,  # minutes
                'max_travel_time': 20   # minutes
            }
        
        return segments
    
    def _create_train_schedule(self, train):
        """Create schedule for simulation"""
        # Simplified schedule creation
        stations = train.get('route_station_ids', [])
        schedule = []
        
        current_time = datetime.now()
        for i, station_id in enumerate(stations):
            arrival = current_time + timedelta(minutes=i * 30)
            departure = arrival + timedelta(minutes=2)
            schedule.append((station_id, arrival, departure))
        
        return {'route': schedule}

def main():
    """Main entry point for the dashboard"""
    app = TrafficControlDashboard()
    app.run()

if __name__ == "__main__":
    main()
