import streamlit as st
import os
from agent import AsterRowAgent
from conversation import ConversationManager

st.set_page_config(
    page_title="Aster & Row Support Agent",
    page_icon="👜",
    layout="centered"
)

st.title("👜 Aster & Row Customer Support")
st.write("Welcome to the mock support center. Type your questions below regarding policies, products, or order lookups.")

# 1. Initialize Agent
@st.cache_resource
def load_agent():
    return AsterRowAgent()

agent = load_agent()

# 2. Setup Session State for Chat History and Trace Log
if "conv_mgr" not in st.session_state:
    st.session_state.conv_mgr = ConversationManager(session_id="streamlit-session")
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

# Sidebar for controls and trace logs
with st.sidebar:
    st.header("Observability & Controls")
    
    # Check for API Key status
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        st.success("Gemini API Connection: Configured")
    else:
        st.error("Gemini API Connection: API Key Missing")
        
    # Clear conversation button
    if st.button("Clear Conversation Session"):
        st.session_state.conv_mgr.clear()
        st.session_state.last_trace = None
        st.rerun()
        
    # Trace log display
    if st.session_state.last_trace:
        st.subheader("Last Turn Execution Trace")
        st.text_area(
            "Structured Trace Log",
            value=st.session_state.last_trace.format_trace(),
            height=400,
            disabled=True
        )

# 3. Render Conversation Messages
for msg in st.session_state.conv_mgr.history:
    # Do not render enriched system context to user, only the text content they typed or model responded
    display_content = msg.content
    if "[Retrieved Context]" in display_content:
        # Extract original query
        display_content = msg.raw_query if msg.raw_query else display_content.split("User Query:")[-1].strip()
        
    with st.chat_message(msg.role):
        st.write(display_content)

# 4. Handle New User Input
if user_query := st.chat_input("How can we help you today?"):
    # Display user query instantly
    with st.chat_message("user"):
        st.write(user_query)
        
    # Run Agent Turn
    with st.spinner("Processing request..."):
        try:
            response, trace = agent.chat("streamlit-session", user_query, st.session_state.conv_mgr)
            st.session_state.last_trace = trace
            
            # Re-render model response
            with st.chat_message("assistant"):
                st.write(response.answer)
                
                # Render sources
                if response.sources:
                    st.caption("**Sources cited:**")
                    for src in response.sources:
                        st.caption(f"- `{src.filename}` → *{src.heading}*")
                        
                # Render handoff recommendation
                if response.handoff:
                    st.warning(f"⚠️ **Human specialist handoff recommended:** {response.handoff_reason}")
            
            st.rerun()
        except Exception as e:
            st.error(f"Error handling request: {str(e)}")
