import streamlit as st
import asyncio
from purpose import simple_crawl, get_activity_model

st.set_page_config(layout="wide")

async def get_activities(url):
    st.write("Debug: Starting URL processing...")
    # activities = await simple_crawl(url)
    activities = [{'activity': 'Bus Ticket Booking', 'url': 'www.redbus.in/bus-tickets'}, {'activity': 'Train Ticket Booking', 'url': 'www.redbus.in/railways'}, {'activity': 'Bus Operator Registration', 'url': 'onboardvendor.redbus.in'}, {'activity': 'Agent Registration', 'url': 'in3.seatseller.travel'}, {'activity': 'Bus Hire Services', 'url': 'www.redbus.in/bushire'}, {'activity': 'Cab Booking Services', 'url': 'www.redbus.in/car-rental/cab-booking'}, {'activity': 'Tempo Traveller Services', 'url': 'www.redbus.in/tempo-traveller'}]
    st.write(f"Debug: Activities received: {activities}")
    return activities

async def process_single_activity(activity):
    st.write(f"Debug: Getting activity model for {activity['activity']}...")
    activity_map = await get_activity_model(activity)
    st.write(f"Debug: Activity map received: {activity_map}")
    return activity_map

def display_activities(activities):
    st.write("### Identified Activities")
    for activity in activities:
        st.write(f"- {activity['activity']} ({activity['url']})")

def display_pom(pom_data):
    if not pom_data:
        st.write("No POM data available for this activity")
        return
    
    st.write("### Page Object Model")
    for element in pom_data.get('input_elements', []):
        with st.expander(f"{element.get('label', 'Input Element')}"):
            for key, value in element.items():
                st.write(f"**{key}:** {value}")

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
                        st.session_state.activities_map = {}  # Reset activities map
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
                display_pom(activity_data.get(st.session_state.selected_activity))
            else:
                st.write("Loading activity details...")

if __name__ == "__main__":
    main() 