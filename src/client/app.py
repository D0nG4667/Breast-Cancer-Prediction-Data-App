import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from config import BEST_MODEL, SCALER, FEATURE_NAMES, FEATURE_CONFIG, CUSTOM_CSS

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Session state for lazy loading ---
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.scaler = None

# Load models
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model(BEST_MODEL)
        scaler = joblib.load(SCALER)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()
        
# Lazy load models only when needed
def lazy_load_models():
    if st.session_state.model is None or st.session_state.scaler is None:
        with st.spinner("Loading AI model..."):
            st.session_state.model, st.session_state.scaler = load_models()       
        message = 'Model loaded successfully!'
        icon = '✅'
        st.toast(message, icon=icon)
            

# Header
st.markdown("""
    <div class="header">
        <h1>🏥 Breast Cancer Prediction AI</h1>
        <p>Advanced machine learning model for early detection and classification</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", ["🔍 Single Prediction", "📈 Batch Prediction", "ℹ️ About Model", "📋 Feature Guide"])

if page == "🔍 Single Prediction":
    st.header("Single Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Feature Categories")
        
        # Feature categories
        feature_categories = {
            "Mean Features": FEATURE_NAMES[:10],
            "Error Features": FEATURE_NAMES[10:20],
            "Worst Features": FEATURE_NAMES[20:30]
        }
        
        selected_category = st.selectbox("Select Feature Category", list(feature_categories.keys()))
        
        user_input = {}
        category_features = feature_categories[selected_category]
        
        for feature in category_features:
            config = FEATURE_CONFIG[feature]
            value = st.slider(
                f"{feature}",
                min_value=float(config['min']),
                max_value=float(config['max']),
                value=float(config['default']),
                step=0.01,
                help=config['description']
            )
            user_input[feature] = value
    
    with col2:
        st.subheader("📊 Feature Visualization")
        
        if user_input:
            # Create a gauge chart for visualization
            fig = go.Figure()
            
            values_list = list(user_input.values())
            feature_list = list(user_input.keys())
            
            # Normalize values for visualization
            normalized_values = []
            for i, feature in enumerate(feature_list):
                config = FEATURE_CONFIG[feature]
                normalized = (values_list[i] - config['min']) / (config['max'] - config['min']) * 100
                normalized_values.append(normalized)
            
            fig = px.bar(
                x=normalized_values,
                y=feature_list,
                orientation='h',
                color=normalized_values,
                color_continuous_scale='RdYlGn_r',
                labels={'x': 'Normalized Value (%)', 'y': 'Feature'}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width='stretch')
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Make Prediction", use_container_width='stretch', type="primary"):
            st.session_state.predict = True
    
    if 'predict' in st.session_state and st.session_state.predict:    
            
        with st.spinner('Analyzing cell nuclei features...'):
            # Lazy load models
            lazy_load_models()   
            model = st.session_state.model
            scaler = st.session_state.scaler 
                   
            # Get all features
            full_input = {}
            for feature in FEATURE_NAMES:
                if feature in user_input:
                    full_input[feature] = user_input[feature]
                else:
                    full_input[feature] = FEATURE_CONFIG[feature]['default']
            
            # Create DataFrame
            input_df = pd.DataFrame([full_input])
            
            
            # Scale input
            scaled_input = scaler.transform(input_df)
            
            # Make prediction
            prediction_proba = model.predict(scaled_input, verbose=0)[0][0]
            prediction_class = int((prediction_proba >= 0.5).astype(int))
            
            # Display results
            st.divider()
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if prediction_class == 1:
                    message = 'MALIGNANT TUMOR DETECTED'
                    icon = '🚨'
                    st.toast(message, icon=icon)
                    message = f'''
                        {message}
                        
                        Confidence: {prediction_proba * 100:.2f}%
                    '''
                    st.error(message, icon=icon)
                else:
                    message = 'BENIGN TUMOR DETECTED'
                    icon = '✅'
                    st.toast(message, icon=icon)
                    message = f'''
                        {message}
                        
                        Confidence: {(1 - prediction_proba) * 100:.2f}%
                    '''
                    st.success(message, icon=icon)


            # Confidence gauge
            st.divider()
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig_gauge = go.Figure(data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Malignancy Score"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "lightcoral"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                )])
                fig_gauge.update_layout(height=350)
                st.plotly_chart(fig_gauge, width='stretch')
            
            with col2:
                # Probability breakdown
                st.metric("Benign Probability", f"{(1 - prediction_proba):.2%}")
                st.metric("Malignant Probability", f"{prediction_proba:.2%}")
                
                # Add timestamp
                st.caption(f"Prediction made at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.warning("⚠️ **Disclaimer**: This is a predictive model for demonstration purposes only. Results should not be used for medical diagnosis. Please consult with a healthcare professional for accurate medical diagnosis.")

elif page == "📈 Batch Prediction":
    st.header("Batch Prediction")
    st.write("Upload a CSV file with 30 features to make predictions on multiple samples.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"File uploaded: {df.shape[0]} samples, {df.shape[1]} features")
            
            if st.button("🚀 Process Batch Predictions"):
                with st.spinner('Processing batch predictions...'):
                    # Lazy load models
                    lazy_load_models()   
                    model = st.session_state.model
                    scaler = st.session_state.scaler 
            
                    # Scale and predict
                    scaled_data = scaler.transform(df)
                    predictions = model.predict(scaled_data, verbose=0)
                    
                    # Create results dataframe
                    results_df = df.copy()
                    results_df['Malignancy_Score'] = predictions
                    results_df['Classification'] = results_df['Malignancy_Score'].apply(
                        lambda x: 'Malignant' if x >= 0.5 else 'Benign'
                    )
                    results_df['Confidence'] = results_df['Malignancy_Score'].apply(
                        lambda x: f"{max(x, 1-x):.2%}"
                    )
                    
                    message = 'Batch predictions completed!'
                    icon = '✅'
                    st.toast(message, icon=icon)
                    st.success(message, icon=icon)
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    malignant_count = (results_df['Classification'] == 'Malignant').sum()
                    benign_count = (results_df['Classification'] == 'Benign').sum()
                    
                    with col1:
                        st.metric("Total Samples", len(results_df))
                    with col2:
                        st.metric("Malignant", malignant_count)
                    with col3:
                        st.metric("Benign", benign_count)
                    
                    st.divider()
                    
                    # Results table
                    st.subheader("Detailed Results")
                    display_cols = ['Malignancy_Score', 'Classification', 'Confidence']
                    st.dataframe(results_df[display_cols], use_container_width='stretch')
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "ℹ️ About Model":
    st.header("About the Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🧠 Model Architecture")
        st.write("""
        - **Type**: Multi-Layer Perceptron (MLP) Neural Network
        - **Framework**: TensorFlow/Keras
        - **Input Features**: 30 cell nuclei characteristics
        - **Output**: Binary Classification (Benign/Malignant)
        - **Training Data**: UCI Breast Cancer Dataset
        """)
    
    with col2:
        st.subheader("📊 Dataset Information")
        st.write("""
        - **Total Samples**: 569
        - **Benign Cases**: 357
        - **Malignant Cases**: 212
        - **Features**: 30 measurements per cell nucleus
        - **Data Source**: UCI Machine Learning Repository
        """)
    
    st.divider()
    
    st.subheader("🎯 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "96.5%")
    with col2:
        st.metric("Precision", "95.2%")
    with col3:
        st.metric("Recall", "97.8%")
    with col4:
        st.metric("F1-Score", "96.5%")
    
    st.divider()
    
    st.subheader("🔧 Preprocessing")
    st.write("""
    - **Scaling**: StandardScaler normalization applied to all features
    - **Feature Selection**: All 30 original features retained
    - **Train/Test Split**: 80/20 with stratification
    """)

elif page == "📋 Feature Guide":
    st.header("Feature Guide")
    st.write("Detailed descriptions of all 30 features used by the model.")
    
    feature_category = st.selectbox("Select Feature Category", 
                                    ["Mean Features", "Error Features", "Worst Features"])
    
    if feature_category == "Mean Features":
        features_to_show = FEATURE_NAMES[:10]
    elif feature_category == "Error Features":
        features_to_show = FEATURE_NAMES[10:20]
    else:
        features_to_show = FEATURE_NAMES[20:30]
    
    for feature in features_to_show:
        config = FEATURE_CONFIG[feature]
        with st.expander(f"**{feature}**"):
            st.write(f"📝 **Description**: {config['description']}")
            st.write(f"📊 **Range**: {config['min']} to {config['max']}")
            st.write(f"📌 **Default**: {config['default']}")
            
            # Show distribution
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Min Value", f"{config['min']}")
            with col2:
                st.metric("Max Value", f"{config['max']}")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px; margin-top: 20px;">
        <p>🏥 Breast Cancer Prediction AI | Built with Streamlit & TensorFlow</p>
        <p>⚠️ This application is for educational and demonstration purposes only.</p>
        <p>Always consult with healthcare professionals for medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
