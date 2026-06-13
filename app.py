
import gradio as gr
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================================================
# =========================================================
# 1. GENERATING ENTERPRISE-SCALE DATASET (WITH REALISTIC NOISE)
# =========================================================
print("🚀 Initializing Enterprise-Scale Data Pipeline...")
start_time = time.time()

DATASET_SIZE = 100000
np.random.seed(42)

data = {
    'SubscriptionPlan': np.random.choice([0, 1, 2], size=DATASET_SIZE, p=[0.4, 0.4, 0.2]), 
    'DeviceUsed': np.random.choice([0, 1, 2, 3], size=DATASET_SIZE), 
    'GenrePreference': np.random.choice([0, 1, 2, 3], size=DATASET_SIZE), 
    'SubscriptionLength': np.random.randint(1, 49, size=DATASET_SIZE), 
    'DailyWatchTime': np.random.uniform(0.2, 8.5, size=DATASET_SIZE), 
    'SatisfactionScore': np.random.randint(1, 11, size=DATASET_SIZE) 
}

df = pd.DataFrame(data)

# Baseline mathematical logic
churn_condition = (
    ((df['DailyWatchTime'] < 1.5) & (df['SatisfactionScore'] <= 4)) | 
    ((df['SubscriptionLength'] < 6) & (df['SubscriptionPlan'] == 2) & (df['SatisfactionScore'] <= 3)) |
    ((df['DailyWatchTime'] < 0.8) & (df['SubscriptionLength'] < 12))
)
df['Churn'] = np.where(churn_condition, 1, 0)

# 🔥 THE FIX: Inject 5% Random Noise to represent human unpredictability
# This flips the answer for 5% of users randomly so the model cannot be 100% perfect
noise_mask = np.random.rand(DATASET_SIZE) < 0.05
df.loc[noise_mask, 'Churn'] = 1 - df.loc[noise_mask, 'Churn']

print(f"📊 Dataset successfully structured with {df.shape[0]:,} rows. Noise injected.")

# =========================================================
# 2. HIGH-PERFORMANCE MODEL TRAINING Pipeline
# =========================================================
print("🧠 Splitting arrays and loading Random Forest Engine into system RAM...")

X = df.drop(columns=['Churn'])
y = df['Churn']

# 80/20 train/test split execution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# High-Performance Settings: n_jobs=-1 forces Python to use ALL computer CPU cores in parallel
ml_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
ml_model.fit(X_train, y_train)

accuracy = ml_model.score(X_test, y_test) * 100
duration = time.time() - start_time
print(f"✅ Engineering Optimization Complete in {duration:.2f} seconds!")
print(f"📈 Production Model Accuracy Verified at: {accuracy:.2f}%")

# =========================================================
# 3. GRADIO CORRELATION COUPLING INTERFACE
# =========================================================
# Structural inverse mappings for string-to-index encoding loops
plan_map = {"Basic": 0, "Standard": 1, "Premium": 2}
device_map = {"Smart TV": 0, "Mobile": 1, "Laptop": 2, "Tablet": 3}
genre_map = {"Action": 0, "Comedy": 1, "Drama": 2, "Sci-Fi": 3}

def predict_enterprise_churn(plan, device, genre, months, watch_time, score):
    # Convert incoming UI string text into numerical vectors
    encoded_row = [[
        plan_map[plan],
        device_map[device],
        genre_map[genre],
        months,
        watch_time,
        score
    ]]
    
    input_df = pd.DataFrame(encoded_row, columns=X.columns)
    
    # Calculate real-time predictions and analytical math arrays
    prediction = ml_model.predict(input_df)[0]
    prob_percentage = ml_model.predict_proba(input_df)[0][1] * 100
    
    # Context-aware business responses based on the calculations
    if prediction == 1 or prob_percentage >= 50.0:
        status = f"⚠️ HIGH CANCELLATION RISK PROFILE ({prob_percentage:.1f}% Probability)"
        strategy = f"💡 Strategy: Subscriber shows critical disengagement patterns. Dispatch an automated push notification offering targeted recommendations based on their favorite genre ({genre}) to drive retention."
    else:
        status = f"✅ HEALTHY ACTIVE SUBSCRIBER ({prob_percentage:.1f}% Probability)"
        strategy = "💡 Strategy: Account behaviors align perfectly with typical user baseline trends. Continue standard recommendation cycling."
        
    return status, strategy

# =========================================================
# 4. FRONTEND SYSTEM RE-DESIGN
# =========================================================
app_interface = gr.Interface(
    fn=predict_enterprise_churn,
    inputs=[
        gr.Dropdown(choices=["Basic", "Standard", "Premium"], value="Standard", label="Subscription Package Tier"),
        gr.Dropdown(choices=["Smart TV", "Mobile", "Laptop", "Tablet"], value="Smart TV", label="Primary Playback Streaming Device"),
        gr.Dropdown(choices=["Action", "Comedy", "Drama", "Sci-Fi"], value="Action", label="Core Content Genre Choice Preference"),
        gr.Slider(minimum=1, maximum=48, value=12, step=1, label="Account Age Duration (Months Active)"),
        gr.Slider(minimum=0.0, maximum=10.0, value=4.0, step=0.5, label="Average Daily Video Content Watch Time (Hours)"),
        gr.Slider(minimum=1, maximum=10, value=7, step=1, label="Self-Reported Customer Satisfaction Survey Rating (1-10)")
    ],
    outputs=[
        gr.Textbox(label="Machine Learning Pipeline Subscription Churn Classification"),
        gr.Textbox(label="Algorithmic Platform Retention Strategy")
    ],
    title="🔮 Scale-Optimized Subscriber Retention Engine",
    description=f"This production-ready Machine Learning system is trained on a dataset of **{DATASET_SIZE:,} user records**. System Validation Accuracy: **{accuracy:.2f}%**."
)

if __name__ == "__main__":
    app_interface.launch(share=True)
