import os
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import RobustScaler, LabelEncoder
from fastapi import Body

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL SETTINGS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Bank_data.csv")
KMEANS_PATH = os.path.join(BASE_DIR, "models", "kmeans.pkl")
LOAN_MODEL_PATH = os.path.join(BASE_DIR, "Loan_Predictions.pkl")

# --- GLOBAL MODELS & SCALERS (Initialized at Startup) ---
MODELS = {
    "kmeans": None,
    "scaler": RobustScaler(), # We store the scaler here to reuse it
    "loan_model": None
}

CACHE = {
    "summary_data": [],
    "user_lookup": {}
}

SCHEME_MASTER = {
    'High': [ 
        {'Scheme_Name': 'Platinum Zero Balance', 'Benefits': ['No min balance', 'Priority banking']},
        {'Scheme_Name': 'Lower Interest Loans', 'Benefits': ['Reduced EMIs', '0% Processing Fee']}
    ],
    'Medium': [
        {'Scheme_Name': 'Gold Salary Account', 'Benefits': ['Free transfers', 'Auto-sweep facility']},
        {'Scheme_Name': 'Cashback Credit Card', 'Benefits': ['5% on Shopping', 'Fuel surcharge waiver']}
    ],
    'Low': [
        {'Scheme_Name': 'Basic Savings (BSBD)', 'Benefits': ['Zero balance', 'Free ATM card']},
        {'Scheme_Name': 'Credit Builder', 'Benefits': ['Score improvement tips', 'Small secured loans']}
    ]
}

# --- STARTUP EVENT ---
@app.on_event("startup")
def load_and_process_data():
    print("🚀 Starting Server: Syncing Admin & User Predictions...")
    
    if not os.path.exists(CSV_PATH): return

    # 1. Load Data & KMeans Model
    df = pd.read_csv(CSV_PATH)
    MODELS["kmeans"] = joblib.load(KMEANS_PATH)
    MODELS["loan_model"] = joblib.load(LOAN_MODEL_PATH)
    
    features = [
        'Age', 'Gender', 'Account Type', 'Relationship_Tenure_Years',
        'Account Balance', 'Avg_Account_Balance', 'AnnualIncome',
        'Monthly_Transaction_Count', 'Avg_Transaction_Amount',
        'Digital_Transaction_Ratio', 'Days_Since_Last_Transaction',
        'Loan Amount', 'Loan Type', 'Loan Term', 'Interest Rate',
        'Active_Loan_Count', 'Credit Utilization', 'Avg_Credit_Utilization',
        'Card_Balance_to_Limit_Ratio', 'Payment Delay Days', 'CIBIL_Score',
        'Card Type', 'Credit Limit', 'Rewards Points', 'Reward_Points_Earned',
        'ActiveStatus'
    ]
    
    x_seg = df[features].copy().fillna(0)
    
    # 2. Process for Admin Summary
    le = LabelEncoder()
    x_seg['ActiveStatus'] = le.fit_transform(x_seg['ActiveStatus'].astype(str))
    x_seg = pd.get_dummies(x_seg, columns=['Gender', 'Account Type', 'Loan Type', 'Card Type'])
    
    # Align and Scale
    x_seg = x_seg.reindex(columns=MODELS["kmeans"].feature_names_in_, fill_value=0)
    x_scaled = MODELS["scaler"].fit_transform(x_seg) # FIT the scaler here
    
    x_scaled_df = pd.DataFrame(x_scaled, columns=MODELS["kmeans"].feature_names_in_)
    clusters = MODELS["kmeans"].predict(x_scaled_df)
    
    segment_map = {0: "Low", 1: "Medium", 2: "High"}
    df['riskLevel'] = [segment_map.get(c, "Low") for c in clusters]
    
    # 3. Cache Population
    df['fullName'] = df['First Name'].astype(str) + " " + df['Last Name'].astype(str)
    CACHE["user_lookup"] = {str(row['Customer ID']): row for row in df.to_dict(orient='records')}
    
    summary_df = df[['Customer ID', 'fullName', 'Email', 'Account Type', 'Account Balance', 'riskLevel', 'ActiveStatus', 'FreezeAccount_Flag']].copy()
    summary_df.columns = ['customerId', 'fullName', 'email', 'accountType', 'balance', 'riskLevel', 'activeStatus', 'isFrozen']
    CACHE["summary_data"] = summary_df.to_dict(orient='records')
    
    print(f"✅ Data Loaded. Admin & User models are now Synced.")

# --- API ENDPOINTS ---

@app.get("/api/dashboard-summary")
async def get_dashboard_summary():
    return CACHE["summary_data"]

@app.get("/api/user/{customer_id}")
async def get_user_details(customer_id: str):
    user = CACHE["user_lookup"].get(customer_id)
    if not user: return {"error": "User not found"}
        
    segment = user['riskLevel'] 
    return {
        "profile": {
            "customerId": str(user["Customer ID"]),
            "balance": float(user["Account Balance"]),
            "segment": f"{segment} Risk",
            "accountStatus": str(user["ActiveStatus"])
        },
        "recommendations": {"schemes": SCHEME_MASTER.get(segment, [])}
    }

# --- THE CORRECTED PREDICTION ENDPOINT ---
@app.post("/api/predict-risk")
async def predict_user_risk(user_data: dict = Body(...)):
    """
    This now uses the EXACT same KMeans logic as the Admin dashboard
    to ensure the risk level matches 100%.
    """
    if MODELS["kmeans"] is None:
        return {"success": False, "predictedRisk": "Model Offline"}

    try:
        df = pd.DataFrame([user_data]).fillna(0)
        
        # 1. Match the Startup Encoding
        # Label encode ActiveStatus manually to match fits
        df['ActiveStatus'] = 1 if str(df['ActiveStatus'].iloc[0]) == "Active" else 0
        
        # One-Hot Encoding
        df = pd.get_dummies(df, columns=['Gender', 'Account Type', 'Loan Type', 'Card Type'])
        
        # 2. Align Columns
        df_final = df.reindex(columns=MODELS["kmeans"].feature_names_in_, fill_value=0)
        
        # 3. SCALE using the RobustScaler from startup (TRANSFORM only, not fit)
        x_scaled = MODELS["scaler"].transform(df_final)
        x_scaled_df = pd.DataFrame(x_scaled, columns=MODELS["kmeans"].feature_names_in_)
        
        # 4. Predict Cluster
        cluster = MODELS["kmeans"].predict(x_scaled_df)[0]
        
        # 5. Map to same names as Admin Dashboard
        segment_map = {0: "Low", 1: "Medium", 2: "High"}
        final_risk = segment_map.get(cluster, "Low")
        
        return {"success": True, "predictedRisk": final_risk}
        
    except Exception as e:
        print(f"Prediction Error: {e}") 
        return {"success": False, "predictedRisk": "Data Error"}

@app.post("/api/predict-loan")
async def predict_loan_amount(user_data: dict = Body(...)):
    """
    Predicts loan amount using the XGBRegressor model.
    """
    if MODELS["loan_model"] is None:
        return {"success": False, "predictedLoan": "Model Offline"}

    try:
        df = pd.DataFrame([user_data]).fillna(0)
        
        # Features for loan model: ['Age', 'Employment Type_Business', 'Employment Type_Freelancer', 'Employment Type_Salaried', 'Employment Type_Self-Employed', 'Residential Status_Company Provided', 'Residential Status_Owned', 'Residential Status_Rented', 'Residence Type_Apartment', 'Residence Type_Independent House', 'Residence Type_Villa', 'CIBIL_Score', 'Relationship_Tenure_Years', 'Years_in_Current_City', 'Years_in_Current_Job', 'Insurance Premiums', 'AnnualIncome', 'Loan Type_Auto', 'Loan Type_Mortgage', 'Loan Type_Personal', 'Loan Type_other']
        
        # Assume user_data has 'Employment Type', 'Residential Status', 'Residence Type', 'Loan Type' as strings
        # One-hot encode them
        if 'Employment Type' in df.columns:
            df = pd.get_dummies(df, columns=['Employment Type'], prefix='Employment Type')
        if 'Residential Status' in df.columns:
            df = pd.get_dummies(df, columns=['Residential Status'], prefix='Residential Status')
        if 'Residence Type' in df.columns:
            df = pd.get_dummies(df, columns=['Residence Type'], prefix='Residence Type')
        if 'Loan Type' in df.columns:
            df = pd.get_dummies(df, columns=['Loan Type'], prefix='Loan Type')
        
        # Align to model's feature_names_in_
        df_final = df.reindex(columns=MODELS["loan_model"].feature_names_in_, fill_value=0)
        
        # Predict (no scaling needed for XGBoost typically)
        prediction = MODELS["loan_model"].predict(df_final)[0]
        
        # Generate recommendations based on predicted loan amount
        recommendations = []
        if prediction > 500000:  # Assuming high loan amount
            recommendations = [
                {"Scheme_Name": "High Value Loan Support", "Benefits": ["Extended repayment terms", "Lower interest rates for large loans"]},
                {"Scheme_Name": "Premium Banking Services", "Benefits": ["Dedicated loan manager", "Priority processing"]}
            ]
        elif prediction > 200000:
            recommendations = [
                {"Scheme_Name": "Medium Loan Assistance", "Benefits": ["Flexible EMI options", "Quick approval process"]},
                {"Scheme_Name": "Savings Account Boost", "Benefits": ["Higher interest on savings", "Loan-linked rewards"]}
            ]
        else:
            recommendations = [
                {"Scheme_Name": "Small Loan Scheme", "Benefits": ["No collateral required", "Fast track approval"]},
                {"Scheme_Name": "Basic Credit Building", "Benefits": ["Credit score improvement tips", "Low-interest personal loans"]}
            ]
        
        return {"success": True, "predictedLoanAmount": float(prediction), "recommendations": recommendations}
        
    except Exception as e:
        print(f"Loan Prediction Error: {e}") 
        return {"success": False, "predictedLoanAmount": "Data Error"}