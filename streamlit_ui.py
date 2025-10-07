import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(page_title="Fraud Detection API", layout="wide")

st.title("🔍 Fraud Detection - Input Data for /predict API")

# Initialize session state for data storage
if 'data' not in st.session_state:
    st.session_state.data = []

# API endpoint configuration
api_url = st.text_input("API Endpoint", value="http://127.0.0.1:8000/predict")

st.markdown("---")

# Input section
st.subheader("📝 Add Transaction Data")
st.write("Enter 10 numbers separated by commas:")

col1, col2 = st.columns([4, 1])

with col1:
    input_text = st.text_input("Input", placeholder="e.g., 1.5, 2.3, 4.1, 0.5, 3.2, 1.8, 2.9, 0.7, 1.2, 3.5", label_visibility="collapsed")

with col2:
    add_button = st.button("➕ Add Row", use_container_width=True)

if add_button:
    if input_text:
        try:
            nums = [float(x.strip()) for x in input_text.split(',')]
            if len(nums) != 10:
                st.error("❌ Please enter exactly 10 numbers.")
            else:
                st.session_state.data.append(nums)
                st.success("✅ Row added successfully!")
                st.rerun()
        except ValueError:
            st.error("❌ Please enter valid numbers separated by commas.")
    else:
        st.warning("⚠️ Please enter some data first.")

# Display current data
st.markdown("---")
st.subheader("📊 Data Added")

if st.session_state.data:
    # Show as DataFrame for better visualization
    df = pd.DataFrame(st.session_state.data, columns=[f"Feature {i+1}" for i in range(10)])
    st.dataframe(df, use_container_width=True)
    
    # Show JSON format
    with st.expander("View as JSON"):
        st.json({"data": st.session_state.data})
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("🚀 Send to API", type="primary", use_container_width=True):
            with st.spinner("Sending data to API..."):
                try:
                    headers = {"Content-Type": "application/json"}
                    payload = {"data": st.session_state.data}
                    response = requests.post(api_url, json=payload, headers=headers, timeout=10)
                    response.raise_for_status()
                    result = response.json()
                    
                    st.success("✅ Request successful!")
                    st.subheader("📥 API Response:")
                    st.json(result)
                    
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Could not connect to API at {api_url}. Make sure the server is running.")
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. The server took too long to respond.")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Request failed: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.data = []
            st.rerun()
    
    with col3:
        st.write(f"**Rows:** {len(st.session_state.data)}")
else:
    st.info("👆 No data added yet. Enter 10 numbers above and click 'Add Row'.")

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    1. **Enter Data**: Input 10 comma-separated numbers
    2. **Add Rows**: Click 'Add Row' to add each transaction
    3. **Send**: Click 'Send to API' when ready
    4. **Clear**: Remove all data with 'Clear All Data'
    """)
    
    st.markdown("---")
    
    st.subheader("Example Input:")
    st.code("1.5, 2.3, 4.1, 0.5, 3.2, 1.8, 2.9, 0.7, 1.2, 3.5")
    
    st.markdown("---")
    
    st.subheader("API Requirements:")
    st.markdown("""
    - Endpoint must be running
    - Accepts POST requests
    - Expects JSON: `{"data": [[...]]}`
    """)
    
    st.markdown("---")
    st.caption("💡 Make sure your API server is running before sending data!")