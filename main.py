import streamlit as st
import numpy as np
import pandas as pd
from model.crop_recommendation_model import CropRecommender
from utils.visualization import create_gauge_chart, create_feature_importance_plot, create_model_comparison_plot
from utils.pdf_generator import create_prediction_pdf
from utils.advanced_features import IrrigationScheduler, EconomicAnalyzer, CropRotationPlanner
import os
from datetime import datetime
from streamlit_lottie import st_lottie
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('styles/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Add custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    /* Custom CSS for image gallery uniformity */
    .crop-gallery .stImage img {
        height: 150px; /* Set a fixed height */
        width: 150px; /* Set a fixed width for square images */
        object-fit: cover; /* Crop image to cover the area */
        border-radius: 8px; /* Optional: add some rounded corners */
    }
    
    /* Modern Page Border */
    .main > div {
        padding: 2rem;
        border: 1px solid #cccccc; /* Subtle grey border */
        border-radius: 10px; /* Rounded corners for the border */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Soft shadow for depth */
        background-color: #ffffff; /* White background inside the border */
        margin: 2rem auto; /* Center the bordered content and add space */
        max-width: 1200px; /* Limit the maximum width */
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Hero section with animation
lottie_agri = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_49rdyysj.json")
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🌾 Smart Crop Recommendation System")
    st.markdown("""
    <div style='font-size: 1.2rem; color: #2c3e50;'>
    Transform your farming decisions with AI-powered insights. Our system helps you make informed choices 
    about crop selection based on soil conditions and environmental factors.
    </div>
    """, unsafe_allow_html=True)
with col2:
    if lottie_agri:
        st_lottie(lottie_agri, height=200, key="agriculture")

# Initialize the model and advanced features
@st.cache_resource
def load_model():
    return CropRecommender(), IrrigationScheduler(), EconomicAnalyzer(), CropRotationPlanner()

try:
    model, irrigation_scheduler, economic_analyzer, rotation_planner = load_model()
    # Load dataset for range calculation
    df = pd.read_csv("attached_assets/Crop_recommendation (1).csv")
except Exception as e:
    st.error("Failed to initialize the system. Please try again later.")
    st.stop()

# Create tabs with modern styling
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌱 Crop Recommendation", 
    "💧 Irrigation Schedule", 
    "💰 Economic Analysis",
    "🔄 Crop Rotation",
    "📊 Database View"
])

with tab1:
    # Original crop recommendation interface with enhanced styling
    st.markdown("### 📊 Enter Parameters")
    
    # Use a container for inputs
    with st.container():
        st.markdown("#### 🌱 Soil Parameters")
        col1, col2, col3, col4 = st.columns(4) # Use more columns for input sliders

        with col1:
            nitrogen = st.slider("Nitrogen (N) mg/kg", 
                                float(df['N'].min()), float(df['N'].max()), 
                                float(df['N'].mean()))
        with col2:
            phosphorus = st.slider("Phosphorus (P) mg/kg", 
                                  float(df['P'].min()), float(df['P'].max()), 
                                  float(df['P'].mean()))
        with col3:
            potassium = st.slider("Potassium (K) mg/kg", 
                                 float(df['K'].min()), float(df['K'].max()), 
                                 float(df['K'].mean()))
        with col4:
            ph = st.slider("pH value", 
                           float(df['ph'].min()), float(df['ph'].max()), 
                           float(df['ph'].mean()))

        st.markdown("#### 🌡️ Environmental Conditions")
        col5, col6, col7 = st.columns(3) # Use more columns for input sliders
        
        with col5:
            temperature = st.slider("Temperature (°C)", 
                                  float(df['temperature'].min()), float(df['temperature'].max()), 
                                  float(df['temperature'].mean()))
        with col6:
            humidity = st.slider("Humidity (%)", 
                                float(df['humidity'].min()), float(df['humidity'].max()), 
                                float(df['humidity'].mean()))
        with col7:
            rainfall = st.slider("Rainfall (mm)", 
                                float(df['rainfall'].min()), float(df['rainfall'].max()), 
                                float(df['rainfall'].mean()))

    if st.button("🔍 Get Crop Recommendation", use_container_width=True):
        with st.spinner("Analyzing parameters..."):
            try:
                # Prepare input features
                features = np.array([nitrogen, phosphorus, potassium, 
                                   temperature, humidity, ph, rainfall])

                # Get prediction and probabilities
                prediction, probabilities = model.predict(features)

                # Display results
                st.markdown("""
                <div style='background-color: #4CAF50; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 1.5rem;'>
                    <h2>Recommended Crop: {}</h2>
                </div>
                """.format(prediction.title()), unsafe_allow_html=True)

                # Use a container for gauge charts
                with st.container():
                    st.subheader("Current Parameter Levels")
                    g1, g2, g3 = st.columns(3) # Keep three columns for gauges

                    with g1:
                        st.plotly_chart(create_gauge_chart(
                            nitrogen, "Nitrogen Level", 
                            float(df['N'].min()), float(df['N'].max())
                        ))
                    with g2:
                        st.plotly_chart(create_gauge_chart(
                            ph, "pH Level", 
                            float(df['ph'].min()), float(df['ph'].max())
                        ))
                    with g3:
                        st.plotly_chart(create_gauge_chart(
                            rainfall, "Rainfall", 
                            float(df['rainfall'].min()), float(df['rainfall'].max())
                        ))

                # Use expanders for detailed performance metrics and feature importance
                with st.expander("📊 View Model Performance Analysis"):
                    model_scores = model.get_model_scores()

                    c1, c2 = st.columns(2)

                    with c1:
                        st.markdown("#### Model Accuracy Comparison")
                        model_comparison_fig = create_model_comparison_plot(model_scores)
                        st.plotly_chart(model_comparison_fig, use_container_width=True)

                    with c2:
                        st.markdown("#### Detailed Model Metrics")
                        for model_name, scores in model_scores.items():
                            st.markdown(f"""
                            **{model_name.title()}**:
                            - Accuracy: {scores['accuracy']:.2%}
                            - Cross-validation Score: {scores['cv_mean']:.2%} (±{scores['cv_std']:.2%})
                            """)
                            
                with st.expander("📈 View Parameter Importance Analysis"):        
                    # Feature importance plot
                    importance_scores = model.get_feature_importance()
                    feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 
                                       'Temperature', 'Humidity', 'pH', 'Rainfall']
                    feature_importance_fig = create_feature_importance_plot(
                        importance_scores, feature_names
                    )
                    st.plotly_chart(feature_importance_fig, use_container_width=True)

                # Show top 3 predictions with probabilities in a container
                with st.container():
                    st.subheader("Top Crop Recommendations")
                    crop_labels = model.get_crop_labels()
                    top_3_idx = probabilities.argsort()[-3:][::-1]

                    crop_probs = []
                    for idx in top_3_idx:
                        crop = crop_labels[idx]
                        prob = probabilities[idx]
                        crop_probs.append((crop, prob * 100))
                        st.markdown(f"""
                        - **{crop.title()}**: {prob*100:.1f}% confidence
                        """)

                # Prepare data for PDF
                prediction_data = {
                    'prediction': prediction,
                    'parameters': {
                        'Nitrogen': nitrogen,
                        'Phosphorus': phosphorus,
                        'Potassium': potassium,
                        'Temperature': temperature,
                        'Humidity': humidity,
                        'pH': ph,
                        'Rainfall': rainfall
                    },
                    'model_scores': model_scores
                }

                # Generate PDF
                pdf_output = create_prediction_pdf(
                    prediction_data,
                    feature_importance_fig,
                    crop_probs
                )

                # Add download button
                st.download_button(
                    label="📄 Download Recommendation Report (PDF)",
                    data=pdf_output,
                    file_name="crop_recommendation_report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"An error occurred while making the prediction: {str(e)}")

with tab2:
    st.subheader("Irrigation Scheduling")

    # Use a container for irrigation inputs
    with st.container():
        st.markdown("#### 💧 Enter Irrigation Parameters")
        # Get available crops from the irrigation scheduler database
        available_crops_irrigation = list(irrigation_scheduler.water_requirements.keys())
        available_crops_irrigation.sort()

        # Input fields for irrigation
        col1, col2, col3 = st.columns(3)
        with col1:
            crop_name = st.selectbox("Select crop", available_crops_irrigation)
        with col2:
            area = st.number_input("Field area (hectares)", min_value=0.1, value=1.0)
        with col3:
            monthly_rainfall = st.number_input("Expected monthly rainfall (mm)", min_value=0.0, value=100.0)

    if st.button("💧 Calculate Irrigation Schedule", use_container_width=True):
        if crop_name:
            schedule = irrigation_scheduler.calculate_schedule(crop_name, area, monthly_rainfall)
            if schedule:
                st.success("Irrigation Schedule Generated")
                
                # Use an expander for the schedule details
                with st.expander("View Schedule Details"):
                    schedule_df = pd.DataFrame(schedule)
                    st.dataframe(schedule_df, use_container_width=True)

                    # Create downloadable CSV
                    csv = schedule_df.to_csv(index=False)
                    st.download_button(
                        label="Download Schedule (CSV)",
                        data=csv,
                        file_name="irrigation_schedule.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Crop not found in database")

with tab3:
    st.subheader("Economic Analysis")

    # Use a container for economic analysis inputs
    with st.container():
        st.markdown("#### 💰 Enter Economic Parameters")
        # Get available crops from the economic analyzer database
        available_crops_economics = list(economic_analyzer.crop_economics.keys())
        available_crops_economics.sort()

        # Input fields for economic analysis
        col1, col2 = st.columns(2)
        with col1:
             analysis_crop = st.selectbox("Select crop for analysis", available_crops_economics)
        with col2:
            analysis_area = st.number_input("Field area for analysis (hectares)", min_value=0.1, value=1.0)

    if st.button("💰 Analyze Economics", use_container_width=True):
        if analysis_crop:
            analysis = economic_analyzer.analyze_crop(analysis_crop, analysis_area)
            if analysis:
                st.success("Economic Analysis Results")
                # Use a container for metrics display
                with st.container():
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Total Cost (₹)", f"₹{analysis['total_cost']:,.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Expected Yield (kg)", f"{analysis['expected_yield']:,.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("Expected Revenue (₹)", f"₹{analysis['expected_revenue']:,.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.metric("ROI", f"{analysis['roi_percentage']}%", delta_color="inverse") # Highlight ROI
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Expected Profit (₹)", f"₹{analysis['expected_profit']:,.2f}", delta_color="inverse") # Highlight Profit
                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.error("Crop not found in database")

with tab4:
    st.subheader("Crop Rotation Planning")

    # Use a container for crop rotation inputs
    with st.container():
        st.markdown("#### 🔄 Enter Rotation Parameters")
        # Create a comprehensive list of all crops from all data sources
        all_seasonal_crops = []
        # Get crops from rotation planner
        for season_crops in rotation_planner.seasonal_crops.values():
            all_seasonal_crops.extend(season_crops)
        # Get crops from irrigation scheduler
        all_seasonal_crops.extend(irrigation_scheduler.water_requirements.keys())
        # Get crops from economic analyzer
        all_seasonal_crops.extend(economic_analyzer.crop_economics.keys())
        all_seasonal_crops = sorted(list(set(all_seasonal_crops)))  # Remove duplicates and sort

        # Input fields for rotation planning
        col1, col2 = st.columns(2)
        with col1:
            current_crop = st.selectbox("Select current crop", all_seasonal_crops)
        with col2:
            season = st.selectbox("Select season", ["Summer", "Winter", "Monsoon"])

    if st.button("🔄 Get Rotation Suggestions", use_container_width=True):
        if current_crop:
            rotation = rotation_planner.suggest_rotation(current_crop, season)
            if rotation:
                st.success("Rotation Suggestions Generated")
                # Use an expander for rotation suggestions
                with st.expander("View Suggested Crops and Benefits"):
                    st.write("Suggested crops for next season:")
                    for crop in rotation['suggested_crops']:
                        st.markdown(f"- {crop.title()}")
                    st.info(rotation['rotation_benefits'])
            else:
                st.error("Could not generate rotation suggestions")

# Enhanced footer
st.markdown("""
<div style='background-color: #2c3e50; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin-top: 2rem;'>
    <h3>Made with ❤️ for farmers</h3>
    <p>Data-driven agriculture for better yields</p>
</div>
""", unsafe_allow_html=True)

with tab5:
    st.subheader("Crop Database Information")
    
    db_view_option = st.radio(
        "Select Database to View",
        ["Irrigation Requirements", "Economic Analysis", "Crop Rotation"]
    )
    
    if db_view_option == "Irrigation Requirements":
        # Use an expander for Irrigation Requirements database
        with st.expander("💧 View Irrigation Requirements Database"):
            st.write("### Irrigation Requirements Database")
            
            # Convert irrigation scheduler data to DataFrame
            irrigation_data = []
            for crop, details in irrigation_scheduler.water_requirements.items():
                irrigation_data.append({
                    "Crop": crop.title(),
                    "Water Need (mm/season)": details['water_need'],
                    "Irrigation Frequency (days)": details['frequency']
                })
            
            irrigation_df = pd.DataFrame(irrigation_data)
            irrigation_df = irrigation_df.sort_values("Crop")
            
            st.dataframe(irrigation_df, use_container_width=True)
            
            # Add download button for irrigation data
            csv = irrigation_df.to_csv(index=False)
            st.download_button(
                label="Download Irrigation Database (CSV)",
                data=csv,
                file_name="irrigation_database.csv",
                mime="text/csv"
            )
        
    elif db_view_option == "Economic Analysis":
        # Use an expander for Economic Analysis database
        with st.expander("💰 View Economic Analysis Database"):
            st.write("### Economic Analysis Database")
            
            # Convert economic analyzer data to DataFrame
            economic_data = []
            for crop, details in economic_analyzer.crop_economics.items():
                economic_data.append({
                    "Crop": crop.title(),
                    "Cost per Hectare (₹)": details['cost_per_hectare'],
                    "Average Yield (kg/hectare)": details['avg_yield'],
                    "Price per kg (₹)": details['price_per_kg'],
                    "Estimated Revenue per Hectare (₹)": details['avg_yield'] * details['price_per_kg'],
                    "Estimated Profit per Hectare (₹)": (details['avg_yield'] * details['price_per_kg']) - details['cost_per_hectare'],
                    "ROI (%)": round(((details['avg_yield'] * details['price_per_kg']) - details['cost_per_hectare']) / details['cost_per_hectare'] * 100, 2)
                })
            
            economic_df = pd.DataFrame(economic_data)
            economic_df = economic_df.sort_values("Crop")
            
            st.dataframe(economic_df, use_container_width=True)
            
            # Add download button for economic data
            csv = economic_df.to_csv(index=False)
            st.download_button(
                label="Download Economic Database (CSV)",
                data=csv,
                file_name="economic_database.csv",
                mime="text/csv"
            )
        
    else:  # Crop Rotation
        # Use an expander for Crop Rotation database
        with st.expander("🔄 View Crop Rotation Database"):
            st.write("### Crop Rotation Database")
            
            # Seasonal crops
            st.subheader("Seasonal Crops")
            seasonal_data = []
            for season, crops in rotation_planner.seasonal_crops.items():
                seasonal_data.append({
                    "Season": season.title(),
                    "Suitable Crops": ", ".join([crop.title() for crop in crops])
                })
            
            seasonal_df = pd.DataFrame(seasonal_data)
            st.dataframe(seasonal_df, use_container_width=True)
            
            # Rotation benefits
            st.subheader("Crop Categories for Rotation")
            rotation_data = []
            for category, crops in rotation_planner.rotation_benefits.items():
                rotation_data.append({
                    "Category": category.title(),
                    "Crops": ", ".join([crop.title() for crop in crops]),
                    "Rotation Benefit": "Nitrogen fixing" if category == "legumes" else 
                                       "Soil structure improvement" if category == "cereals" else
                                       "Economic value"
                })
            
            rotation_df = pd.DataFrame(rotation_data)
            st.dataframe(rotation_df, use_container_width=True)
            
            # Add download buttons for rotation data
            csv_seasonal = seasonal_df.to_csv(index=False)
            csv_rotation = rotation_df.to_csv(index=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Seasonal Crops (CSV)",
                    data=csv_seasonal,
                    file_name="seasonal_crops.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="Download Crop Categories (CSV)",
                    data=csv_rotation,
                    file_name="crop_categories.csv",
                    mime="text/csv"
                )
