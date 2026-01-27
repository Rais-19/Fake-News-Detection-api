import streamlit as st
import requests
import json
from datetime import datetime

# API Configuration
API_URL = "http://127.0.0.1:8000"

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border-color: #FF6B6B;
    }
    .fake-news {
        background-color: #FFE5E5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
    }
    .real-news {
        background-color: #E5F5E5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health(): #check if api is running:
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_news(text):
    """Send prediction request to the API"""
    try:
        payload = {"text": text}
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return None, error_detail
            
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure the server is running."
    except requests.exceptions.Timeout:
        return None, "Request timeout. The server is taking too long to respond."
    except Exception as e:
        return None, f"An error occurred: {str(e)}"


# Header
st.title("📰 Fake News Detector")
st.markdown(" Detect whether a news article is **REAL** or **FAKE** using AI")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application uses **Machine Learning** to classify news articles as real or fake.
    
    **How it works:**
    1. Enter a news headline or article
    2. Click "Analyze News"
    3. Get instant results with confidence score
    
    **Model Details:**
    - Algorithm: Logistic Regression
    - Features: TF-IDF Vectorization
    - Preprocessing: Stemming & Stopword Removal
    """)
    
    st.markdown("---")
    
    # API Status
    st.header("🔌 API Status")
    if check_api_health():
        st.success("✅ API is running")
    else:
        st.error("❌ API is not running")
        st.warning("Start the API with: `uvicorn app:app --reload`")
    
    st.markdown("---")
    
    # Examples
    st.header("💡 Example Headlines")
    examples = [
        "Scientists discover cure for cancer in breakthrough research",
        "Alien spaceship spotted landing in New York City",
        "President announces new economic policy at press conference",
        "Celebrity reveals shocking secret about government conspiracy"
    ]
    
    if 'example_idx' not in st.session_state:
        st.session_state.example_idx = 0
    
    for i, example in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"example_{i}"):
            st.session_state.example_idx = i
            st.session_state.input_text = example

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter News Article")
    
    # Text input
    default_text = st.session_state.get('input_text', '')
    news_text = st.text_area(
        "Paste your news headline or article here:",
        value=default_text,
        height=200,
        placeholder="Example: Breaking news: Scientists discover new planet in solar system...",
        help="Enter at least 10 characters"
    )
    
    # Character count
    char_count = len(news_text)
    if char_count < 10:
        st.warning(f"⚠️ Please enter at least 10 characters (current: {char_count})")
    else:
        st.info(f"📊 Character count: {char_count}")
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze News", disabled=(char_count < 10))
    
    # Clear button
    if st.button("🗑️ Clear"):
        st.session_state.input_text = ''
        st.rerun()

with col2:
    st.markdown("### 📊 Quick Stats")
    
    # Initialize session state for stats
    if 'total_analyzed' not in st.session_state:
        st.session_state.total_analyzed = 0
        st.session_state.fake_count = 0
        st.session_state.real_count = 0
    
    # Display stats
    st.metric("Total Analyzed", st.session_state.total_analyzed)
    
    col_fake, col_real = st.columns(2)
    with col_fake:
        st.metric("🚫 Fake", st.session_state.fake_count)
    with col_real:
        st.metric("✅ Real", st.session_state.real_count)

# Results section
if analyze_button and news_text:
    with st.spinner("🔄 Analyzing news article..."):
        result, error = predict_news(news_text)
        
        if error:
            st.error(f"❌ Error: {error}")
        else:
            # Update stats
            st.session_state.total_analyzed += 1
            if result['prediction'] == 'FAKE':
                st.session_state.fake_count += 1
            else:
                st.session_state.real_count += 1
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Analysis Results")
            
            prediction = result['prediction']
            confidence = result['confidence']
            confidence_pct = int(confidence * 100)
            
            # Result box with custom styling
            if prediction == 'FAKE':
                st.markdown(f"""
                <div class="fake-news">
                    <h2 style="color: #FF4B4B; margin: 0;">🚫 FAKE NEWS DETECTED</h2>
                    <p style="font-size: 18px; margin-top: 10px;">
                        This article is likely <strong>NOT RELIABLE</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="real-news">
                    <h2 style="color: #4CAF50; margin: 0;">✅ REAL NEWS DETECTED</h2>
                    <p style="font-size: 18px; margin-top: 10px;">
                        This article appears to be <strong>RELIABLE</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence meter
            st.markdown("### 📈 Confidence Score")
            st.progress(confidence)
            st.markdown(f"**{confidence_pct}%** confidence")
            
            # Additional details in expandable section
            with st.expander("📋 View Detailed Analysis"):
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Classification", prediction)
                
                with col_b:
                    st.metric("Confidence", f"{confidence_pct}%")
                
                with col_c:
                    st.metric("Text Length", result['text_length'])
                
                st.markdown("**Full Message:**")
                st.info(result['message'])
                
                st.markdown("**API Response:**")
                st.json(result)
            
        
            
            # Timestamp
            st.markdown("---")
            st.caption(f"🕐 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


