import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="📱 PhoneGenie - AI Phone Recommender", 
    page_icon="📱", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    :root {
        --primary: #4a6bdf;
        --secondary: #f8f9fa;
        --accent: #ff4b4b;
        --text: #2c3e50;
        --light-text: #6c757d;
        --bg: #ffffff;
        --card-bg: #ffffff;
    }
    
    .main {
        background-color: var(--bg);
    }
    
    .header {
        background: linear-gradient(135deg, #4a6bdf 0%, #6a11cb 100%);
        color: white;
        padding: 2rem;
        border-radius: 0 0 15px 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .phone-card {
        border-radius: 12px;
        padding: 1.5rem;
        background-color: var(--card-bg);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .phone-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .phone-card img {
        border-radius: 8px;
        # object-fit: cover;
        height: 350px;
        width:180px;
    }
    
    .phone-card h4 {
        color: var(--primary);
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .phone-card .price {
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--accent);
    }
    
    .phone-card .specs {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .phone-card .spec {
        background-color: var(--secondary);
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        color: var(--text);
    }
    
    .phone-card .rating {
        display: flex;
        align-items: center;
        margin-top: 0.5rem;
    }
    
    .phone-card .rating .stars {
        color: #ffc107;
        margin-right: 0.5rem;
    }
    
    .algorithm-card {
        border-radius: 12px;
        padding: 1.5rem;
        background-color: var(--card-bg);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary);
    }
    
    .metric-box {
        background-color: var(--secondary);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    
    .metric-box h3 {
        color: var(--primary);
        margin: 0.5rem 0;
        font-size: 1.5rem;
    }
    
    .metric-box p {
        color: var(--light-text);
        margin: 0;
        font-size: 0.9rem;
    }
    
    .sidebar .sidebar-content {
        background-color: var(--bg);
        padding: 1.5rem;
    }
    
    .sidebar-title {
        color: var(--primary);
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .stButton button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    
    .stButton button:hover {
        background-color: #3a5bd9;
        color: white;
    }
    
    .tab-content {
        padding: 1rem 0;
    }
    
    .comparison-table {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .comparison-chart {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .about-section {
        background-color: var(--secondary);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
    }
    
    .feature-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        color: var(--primary);
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header">
        <h1>📱 PhoneGenie</h1>
        <p>Your personal AI assistant for finding the perfect smartphone</p>
    </div>
    """, unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    # Read the dataset
    df = pd.read_csv("mobile_recommendation_system_dataset.csv", encoding="latin1")
    
    # Clean data
    df_clean = df.dropna()
    
    # Extract brand from name
    df_clean['Brand'] = df_clean['name'].str.extract(r'^([A-Za-z]+)')[0]
    
    # Feature extraction functions
    def extract_storage(corpus):
        match = re.search(r'Storage(\d+)', str(corpus))
        if match:
            return int(match.group(1))
        return None
    
    def extract_ram(corpus):
        match = re.search(r'RAM(\d+)', str(corpus))
        if match:
            return int(match.group(1))
        return None
    
    def extract_system(corpus):
        match = re.search(r'System(.*?)Processor', str(corpus))
        if match:
            return match.group(1).strip()
        return None
    
    def extract_processor(corpus):
        match = re.search(r'Processor (.*?) ', str(corpus))
        if match:
            return match.group(1).strip()
        return None
    
    def clean_system(system):
        if pd.isnull(system):
            return None
        system = str(system).lower()
        if 'android' in system:
            return 'Android'
        if 'ios' in system:
            return 'iOS'
        if 'tizen' in system:
            return 'Tizen'
        return 'Other'
    
    def clean_processor(processor):
        if pd.isnull(processor):
            return None
        processor = str(processor).lower()
        if 'mediatek' in processor:
            return 'MediaTek'
        if 'qualcomm' in processor or 'snapdragon' in processor:
            return 'Qualcomm'
        if 'apple' in processor:
            return 'Apple'
        if 'exynos' in processor:
            return 'Exynos'
        return 'Other'
    
    # Apply feature extraction
    df_clean['Storage'] = df_clean['corpus'].apply(extract_storage)
    df_clean['RAM'] = df_clean['corpus'].apply(extract_ram)
    df_clean['System'] = df_clean['corpus'].apply(extract_system)
    df_clean['Processor'] = df_clean['corpus'].apply(extract_processor)
    
    # Clean system and processor columns
    df_clean['System'] = df_clean['System'].apply(clean_system)
    df_clean['Processor'] = df_clean['Processor'].apply(clean_processor)
    
    # Clean price column
    df_clean['price'] = df_clean['price'].astype(str)
    df_clean['price'] = df_clean['price'].str.replace('[^0-9]', '', regex=True)
    df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce')
    df_clean = df_clean.dropna(subset=['price'])
    
    # Create price categories for classification
    df_clean['price_category'] = pd.cut(df_clean['price'], 
                                      bins=[0, 10000, 20000, 30000, 50000, float('inf')],
                                      labels=['Budget', 'Mid-Range', 'Premium', 'Flagship', 'Luxury'])
    
    # Drop rows with missing values in key columns
    df_clean = df_clean.dropna(subset=['Brand', 'Storage', 'RAM', 'System', 'Processor', 'price_category'])
    
    return df_clean

df = load_data()

# Dictionary of available classifiers
CLASSIFIERS = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Train recommendation classifier
@st.cache_resource
def train_classifiers():
    results = {}
    
    # Prepare data for classification
    X = df[['Brand', 'Storage', 'RAM', 'System', 'Processor', 'ratings']]
    y = df['price_category']
    
    # Preprocessing pipeline with imputation for numeric features
    numeric_features = ['Storage', 'RAM', 'ratings']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ])
    
    categorical_features = ['Brand', 'System', 'Processor']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train all classifiers
    for name, classifier in CLASSIFIERS.items():
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        # Train model
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'training_time': training_time
        }
    
    return results

classifiers = train_classifiers()

# Recommendation function using classifier
def get_personalized_recommendations(user_prefs, classifier_name, n_recommendations=10):
    # Get the selected classifier
    classifier = classifiers[classifier_name]['pipeline']
    
    # Handle 'Any' selections
    for key in ['Brand', 'System', 'Processor']:
        if user_prefs[key] == 'Any':
            user_prefs[key] = np.nan
    
    # Create a dataframe from user preferences
    input_data = pd.DataFrame([user_prefs])
    
    # Predict price category
    try:
        predicted_category = classifier.predict(input_data)[0]
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return pd.DataFrame()
    
    # Filter phones in the predicted category
    category_phones = df[df['price_category'] == predicted_category]
    
    # Apply additional filters from user preferences
    if not pd.isnull(user_prefs['Brand']):
        category_phones = category_phones[category_phones['Brand'] == user_prefs['Brand']]
    if not pd.isnull(user_prefs['System']):
        category_phones = category_phones[category_phones['System'] == user_prefs['System']]
    if not pd.isnull(user_prefs['Processor']):
        category_phones = category_phones[category_phones['Processor'] == user_prefs['Processor']]
    
    # Filter by minimum specs
    category_phones = category_phones[
        (category_phones['Storage'] >= user_prefs['Storage']) &
        (category_phones['RAM'] >= user_prefs['RAM']) &
        (category_phones['ratings'] >= user_prefs['ratings'])
    ]
    
    # Sort by rating and price
    recommendations = category_phones.sort_values(['ratings', 'price'], ascending=[False, True])
    
    return recommendations.head(n_recommendations)

# Sidebar - User Preferences
with st.sidebar:
    st.markdown('<div class="sidebar-title">🔍 Your Preferences</div>', unsafe_allow_html=True)
    
    # Get available brands and ensure default exists
    available_brands = sorted(df['Brand'].dropna().unique().tolist())
    preferred_brand = st.selectbox(
        "Brand",
        ['Any'] + available_brands,
        index=0
    )
    
    # Get available OS options
    available_os = sorted(df['System'].dropna().unique().tolist())
    preferred_os = st.selectbox(
        "Operating System",
        ['Any'] + available_os,
        index=0
    )
    
    # Get available processor options
    available_processors = sorted(df['Processor'].dropna().unique().tolist())
    preferred_processor = st.selectbox(
        "Processor",
        ['Any'] + available_processors,
        index=0
    )
    
    st.markdown("---")
    
    min_storage = st.select_slider(
        "Minimum Storage (GB)",
        options=sorted(df['Storage'].dropna().unique()),
        value=64
    )
    
    min_ram = st.select_slider(
        "Minimum RAM (GB)",
        options=sorted(df['RAM'].dropna().unique()),
        value=4
    )
    
    min_rating = st.slider(
        "Minimum Rating",
        min_value=3.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )
    
    st.markdown("---")
    
    # Algorithm selection
    selected_algorithms = st.multiselect(
        "Select AI Algorithms (Max 3)",
        list(CLASSIFIERS.keys()),
        default=["Random Forest"]
    )
    
    # Limit to 3 algorithms for comparison
    if len(selected_algorithms) > 3:
        st.warning("Please select no more than 3 algorithms")
        selected_algorithms = selected_algorithms[:3]
    
    if st.button("Find My Perfect Phone", type="primary"):
        st.session_state['get_recommendations'] = True
    else:
        st.session_state['get_recommendations'] = False

# Main Content
if st.session_state.get('get_recommendations', False):
    with st.spinner('🔮 PhoneGenie is analyzing thousands of phones to find your perfect match...'):
        # Store recommendations from all selected algorithms
        all_recommendations = {}
        user_prefs = {
            'Brand': preferred_brand,
            'System': preferred_os,
            'Processor': preferred_processor,
            'Storage': min_storage,
            'RAM': min_ram,
            'ratings': min_rating
        }
            
        for algo in selected_algorithms:
            recommendations = get_personalized_recommendations(user_prefs, algo, 5)
            if not recommendations.empty:
                all_recommendations[algo] = recommendations
        
        if all_recommendations:
            st.subheader("✨ Your Personalized Recommendations")
            
            # Show algorithm comparison metrics
            st.markdown("### 📊 Algorithm Performance")
            cols = st.columns(len(selected_algorithms))
            
            for i, algo in enumerate(selected_algorithms):
                with cols[i]:
                    st.markdown(f"""
                        <div class="algorithm-card">
                            <h4>{algo}</h4>
                            <div class="metric-box">
                                <p>Accuracy</p>
                                <h3>{classifiers[algo]['accuracy']:.1%}</h3>
                            </div>
                            <div class="metric-box" style="margin-top:10px;">
                                <p>Training Time</p>
                                <h3>{classifiers[algo]['training_time']:.2f}s</h3>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Display recommendations in tabs
            tabs = st.tabs([f"Recommendations by {algo}" for algo in all_recommendations.keys()])
            
            for tab, (algo, recommendations) in zip(tabs, all_recommendations.items()):
                with tab:
                    st.markdown(f"<h4 style='text-align:center; color: var(--primary);'>Top 5 Recommendations by {algo}</h4>", unsafe_allow_html=True)
                    
                    # Display recommendations in a grid
                    cols = st.columns(2)
                    for i, (_, row) in enumerate(recommendations.iterrows()):
                        with cols[i % 2]:
                            st.markdown(f"""
                                <div class="phone-card">
                                    <img src="{row['imgURL']}" onerror="this.src='https://via.placeholder.com/150?text=Phone+Image'">
                                    <h4>{row['name']}</h4>
                                    <div class="price">₹{row['price']:,}</div>
                                    <div class="specs">
                                        <span class="spec">{row['Brand']}</span>
                                        <span class="spec">{row['System']}</span>
                                        <span class="spec">{row['Storage']}GB</span>
                                        <span class="spec">{row['RAM']}GB RAM</span>
                                        <span class="spec">{row['Processor']}</span>
                                    </div>
                                    <div class="rating">
                                        <div class="stars">{"⭐" * int(round(row['ratings']))}</div>
                                        <div>{row['ratings']}/5.0</div>
                                    </div>
                                    <div class="spec" style="background-color: #e3f2fd; margin-top: 0.5rem;">{row['price_category']}</div>
                                </div>
                            """, unsafe_allow_html=True)
        else:
            st.warning("No phones match your preferences. Try adjusting your criteria.")

    # Algorithm comparison section
    if len(selected_algorithms) > 1:
        st.markdown("---")
        st.subheader("📈 Algorithm Comparison Report")
        
        # Create comparison dataframe
        comparison_data = []
        for algo in selected_algorithms:
            comparison_data.append({
                "Algorithm": algo,
                "Accuracy": classifiers[algo]['accuracy'],
                "Training Time (s)": classifiers[algo]['training_time'],
                "Parameters": str(CLASSIFIERS[algo].get_params())
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.markdown("### Comparison Table")
        st.dataframe(
            comparison_df.style
                .background_gradient(cmap='Blues', subset=['Accuracy'])
                .format({
                    "Accuracy": "{:.1%}",
                    "Training Time (s)": "{:.2f}"
                }),
            use_container_width=True
        )
        
        # Visualization
        st.markdown("### Comparison Charts")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="comparison-chart">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=comparison_df, x='Algorithm', y='Accuracy', palette='viridis', ax=ax)
            ax.set_title('Algorithm Accuracy Comparison', pad=20)
            ax.set_ylim(0, 1)
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.1%}", 
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', 
                          xytext=(0, 10), 
                          textcoords='offset points')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="comparison-chart">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=comparison_df, x='Algorithm', y='Training Time (s)', palette='magma', ax=ax)
            ax.set_title('Training Time Comparison (Seconds)', pad=20)
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}", 
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center', 
                          xytext=(0, 10), 
                          textcoords='offset points')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

# Default view when no recommendations requested
else:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            ## Welcome to PhoneGenie!
            
            Finding your perfect smartphone has never been easier. Our AI-powered recommendation engine analyzes thousands of phones
            to match your exact preferences and budget.
            
            **How it works:**
            1. Set your preferences in the sidebar
            2. Choose which AI algorithms to compare
            3. Get personalized recommendations instantly
            
            Our system uses multiple machine learning algorithms to ensure you get the most accurate recommendations possible.
            """)
        
        st.image("https://via.placeholder.com/800x400?text=Phone+Recommendation+Demo", use_column_width=True)
    
    with col2:
        st.markdown("""
            <div class="algorithm-card">
                <h4>Available AI Algorithms</h4>
                <p>Select up to 3 to compare:</p>
                <ul>
                    <li>Random Forest</li>
                    <li>Gradient Boosting</li>
                    <li>K-Nearest Neighbors</li>
                    <li>Support Vector Machine</li>
                    <li>Decision Tree</li>
                </ul>
            </div>
            
            <div class="metric-box" style="margin-top:20px;">
                <p>Total Phones in Database</p>
                <h3>{:,}</h3>
            </div>
            """.format(len(df)), unsafe_allow_html=True)

# About section
st.markdown("---")
with st.expander("ℹ️ About PhoneGenie", expanded=True):
    st.markdown("""
    **About PhoneGenie**
    
    PhoneGenie is an advanced AI-powered smartphone recommendation system that helps you find your perfect device based on your preferences, budget, and usage patterns.
    
    **📊 Our Technology**  
    We use multiple machine learning algorithms to analyze:  
    - Price patterns and value for money  
    - Brand reliability and performance  
    - Technical specifications and real-world performance  
    - User ratings and reviews  
    
    **🔍 Why Compare Algorithms?**  
    Different AI algorithms have different strengths. By comparing multiple approaches, we ensure you get the most accurate recommendations possible.
    
    **📱 Our Database**  
    We analyze data from over 2,074 smartphones across all price ranges and brands to bring you the most comprehensive recommendations.
    """)