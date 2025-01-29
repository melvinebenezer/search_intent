import streamlit as st
import asyncio
from purpose import simple_crawl, get_activity_model, load_activities_map, save_activities_map
import json

st.set_page_config(layout="wide")

# Load activities map at startup
try:
    activities_map = load_activities_map()
except (FileNotFoundError, json.JSONDecodeError):
    activities_map = {}

async def get_activities(url):
    st.write("Debug: Starting URL processing...")
    # activities = await simple_crawl(url)
    activities = [{'activity': 'Bus Ticket Booking', 'url': 'www.redbus.in/bus-tickets'}, {'activity': 'Train Ticket Booking', 'url': 'www.redbus.in/railways'}, {'activity': 'Bus Operator Registration', 'url': 'onboardvendor.redbus.in'}, {'activity': 'Agent Registration', 'url': 'in3.seatseller.travel'}, {'activity': 'Bus Hire Services', 'url': 'www.redbus.in/bushire'}, {'activity': 'Cab Booking Services', 'url': 'www.redbus.in/car-rental/cab-booking'}, {'activity': 'Tempo Traveller Services', 'url': 'www.redbus.in/tempo-traveller'}]
    st.write(f"Debug: Activities received: {activities}")
    return activities

async def process_single_activity(activity):
    st.write(f"Debug: Getting activity model for {activity['activity']}...")
    # Check if activity is already in the map
    if activity['activity'] in activities_map:
        st.write("Loading activity model from cache...")
        return activities_map[activity['activity']]
    
    # Get the activity model from the activities map
    activity_name = activity['activity']
    activity_model = get_activity_model(activity_name, activities_map)
    
    if activity_model:
        # No need to update activities_map since we're getting data from it
        st.write(f"Debug: Activity model retrieved: {activity_model}")
        return activity_model
    return None

def display_activities(activities):
    st.write("### Identified Activities")
    for activity in activities:
        st.write(f"- {activity['activity']} ({activity['url']})")

def display_pom(pom_data):
    if not pom_data:
        st.write("No POM data available for this activity")
        return
    
    st.write("### Activity Steps")
    steps = pom_data.get('steps', [])
    
    if not steps:
        st.write("No steps defined for this activity")
        return
        
    for step in steps:
        with st.expander(f"Step {step['step_number']}: {step['description']}"):
            st.write(f"**URL:** {step['url']}")
            st.write(f"**Requires User Input:** {'Yes' if step['step_lock'] else 'No'}")
            
            if step.get('input_elements'):
                st.write("#### Input Elements:")
                for element in step['input_elements']:
                    # Create a container for each input element
                    st.markdown(f"**{element.get('label', 'Input Element')}**")
                    # Create a container with a border
                    with st.container():
                        st.markdown("---")  # Add a separator line
                        cols = st.columns(2)  # Create two columns for key-value pairs
                        items = [(k, v) for k, v in element.items() if k != 'label']
                        mid = len(items) // 2
                        
                        # First column
                        with cols[0]:
                            for key, value in items[:mid]:
                                st.markdown(f"**{key}:** {value}")
                        
                        # Second column
                        with cols[1]:
                            for key, value in items[mid:]:
                                st.markdown(f"**{key}:** {value}")
            else:
                st.write("*No input elements for this step*")

def main():
    # Create two columns
    left_col, right_col = st.columns([1, 2])
    
    # Session state initialization
    if 'activities' not in st.session_state:
        st.session_state.activities = None
    if 'activities_map' not in st.session_state:
        st.session_state.activities_map = {}
    if 'selected_activity' not in st.session_state:
        st.session_state.selected_activity = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'initial'

    with left_col:
        st.write("## Navigation")
        
        # Initial option
        if st.button("🏠 Activities Overview", key="home"):
            st.session_state.current_view = 'initial'
            st.session_state.selected_activity = None
        
        st.markdown("---")  # Divider
        
        # Activities section
        st.write("### Available Activities")
        if st.session_state.activities:
            for activity in st.session_state.activities:
                # Create a visual indicator for selected activity
                prefix = "▶️ " if st.session_state.selected_activity == activity['activity'] else "   "
                if st.button(f"{prefix}{activity['activity']}", key=activity['activity']):
                    st.session_state.selected_activity = activity['activity']
                    st.session_state.current_view = 'activity_detail'
                    
                    # Only get activity model when activity is clicked and not already processed
                    if activity['activity'] not in st.session_state.activities_map:
                        with st.spinner(f"Analyzing {activity['activity']}..."):
                            try:
                                activity_map = asyncio.run(process_single_activity(activity))
                                if activity_map:
                                    st.session_state.activities_map[activity['activity']] = activity_map
                                    st.success(f"Successfully analyzed {activity['activity']}")
                                else:
                                    st.error(f"No details found for {activity['activity']}")
                            except Exception as e:
                                st.error(f"Error analyzing {activity['activity']}: {str(e)}")
        else:
            st.write("*No activities loaded yet. Enter a URL and click 'Start Analysis'.*")

    with right_col:
        st.write("## Details")
        
        # URL input and process section
        url = st.text_input("Enter URL", "https://www.redbus.in/")
        if st.button("Start Analysis"):
            with st.spinner("Processing URL..."):
                try:
                    activities = asyncio.run(get_activities(url))
                    if activities:
                        st.session_state.activities = activities
                        # Don't reset activities_map completely, just update as needed
                        st.session_state.activities_map = activities_map.copy()
                        st.session_state.current_view = 'initial'
                        st.success("Successfully identified activities!")
                    else:
                        st.error("No activities found for this URL")
                except Exception as e:
                    st.error(f"Error processing URL: {str(e)}")
            
        # Show appropriate view based on current_view
        if st.session_state.current_view == 'initial' and st.session_state.activities:
            st.write("## Activities Overview")
            display_activities(st.session_state.activities)
            
        # Display POM for selected activity
        elif st.session_state.current_view == 'activity_detail' and st.session_state.selected_activity:
            st.write(f"## Details for: {st.session_state.selected_activity}")
            activity_data = st.session_state.activities_map.get(st.session_state.selected_activity, {})
            if activity_data:
                display_pom(activity_data)
            else:
                st.write("Loading activity details...")

if __name__ == "__main__":
    main() 