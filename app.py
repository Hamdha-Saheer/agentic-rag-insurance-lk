import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.multi_agent import multi_agent_query

st.set_page_config(page_title='Lanka Insurance AI Advisor', layout='wide')

# ── Improved Sidebar ───────────────────────────────────────
with st.sidebar:
    st.header('🛡️ User Profile Settings')
    with st.container(border=True):
        use_profile = st.toggle('Enable Personalization', value=True)
        if use_profile:
            age = st.slider('Select Age', 5, 75, 25)
            job = st.selectbox('Occupation', ['Government Employee', 'Student', 'Retired', 'Private Sector', 'Self-Employed'])
            vehicle = st.selectbox('Vehicle Type', ['No Vehicle', 'Car (Petrol)', 'Car (Diesel)', 'Motorcycle', 'Three-Wheeler', 'Van / Lorry'])
            user_profile = {'age': age, 'job': job, 'vehicle': vehicle}
        else:
            user_profile = {}
            st.info("General advice mode active.")

# ── Main UI ───────────────────────────────────────────────
st.title('🛡️ Sri Lanka Insurance AI Advisor')
question = st.text_area('How can I help you with your insurance queries?')

if st.button('🔍 Get Smart Advice', type='primary'):
    with st.spinner('Analyzing across insurance domains...'):
        result = multi_agent_query(question, user_profile)
    
    active_domains = result.get('domains', ['general'])
    tabs = st.tabs([d.upper() for d in active_domains])
    
    for i, tab in enumerate(tabs):
        with tab:
            raw = result['answer']
            if "+(94)" in raw:
                parts = raw.split("+(94)", 1)
                st.markdown(parts[0])
                with st.expander("📞 View Official Contact Numbers"):
                    st.write("+(94)" + parts[1])
            else:
                st.markdown(raw)
            st.markdown('**Sources:** ' + ', '.join(result.get('sources', [])))