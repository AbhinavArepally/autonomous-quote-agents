import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Autonomous Quote Agents System")
st.write("AI Multi-Agent Insurance Decision Engine")

data = pd.read_csv("Autonomous QUOTE AGENTS.csv")

st.subheader("Dataset Preview")
st.write(data.head())

def convert_miles(x):
    nums = re.findall(r'\d+', str(x))
    if len(nums) == 1:
        return int(nums[0]) * 1000
    elif len(nums) == 2:
        return (int(nums[0]) + int(nums[1])) / 2 * 1000
    else:
        return 12000

data["Annual_Miles_Range"] = data["Annual_Miles_Range"].apply(convert_miles)

cat_cols = [
    "Agent_Type",
    "Region",
    "Policy_Type",
    "Gender",
    "Marital_Status",
    "Education",
    "Sal_Range",
    "Coverage",
    "Veh_Usage",
    "Vehicl_Cost_Range",
    "Re_Quote",
    "Policy_Bind"
]

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data[[
    "Prev_Accidents",
    "Prev_Citations",
    "Driving_Exp",
    "Driver_Age",
    "Veh_Usage",
    "Annual_Miles_Range"
]]

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data["Risk_Tier"] = clusters
risk_map = {0:"Low",1:"Medium",2:"High"}
data["Risk_Tier"] = data["Risk_Tier"].map(risk_map)

y = data["Risk_Tier"]

X_train,X_test,y_train,y_test = train_test_split(
    X_scaled,y,test_size=0.2,random_state=42
)

agent1_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=3,
    random_state=42
)

agent1_model.fit(X_train,y_train)

y_pred = agent1_model.predict(X_test)

st.write("Agent 1 Accuracy:",accuracy_score(y_test,y_pred)*100)

X2 = data[[
    "Re_Quote",
    "Coverage",
    "Agent_Type",
    "Region",
    "Sal_Range",
    "HH_Drivers"
]]

y2 = data["Policy_Bind"]

X_train2,X_test2,y_train2,y_test2 = train_test_split(
    X2,y2,test_size=0.2,random_state=42
)

agent2_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    random_state=42
)

agent2_model.fit(X_train2,y_train2)

y_pred2 = agent2_model.predict(X_test2)

st.write("Agent 2 Accuracy:",accuracy_score(y_test2,y_pred2)*100)

st.subheader("Customer Input")

prev_acc = st.number_input("Previous Accidents",0,10,0)
prev_cit = st.number_input("Previous Citations",0,10,0)
exp = st.number_input("Driving Experience",0,40,5)
age = st.number_input("Driver Age",18,80,30)
veh_use = st.selectbox("Vehicle Usage",[0,1,2])
miles = st.number_input("Annual Miles",1000,50000,12000)

re_quote = st.selectbox("Re Quote [0: No, 1: Yes]",[0,1])
coverage = st.selectbox("Coverage[0: basic ,1: balanced ,2: enhance]",[0,1,2])
agent_type = st.selectbox("Agent Type [0: EA,1: IA]",[0,1])
region = st.selectbox("Region[1-A,7-H]",[0,1,2,3,4,5,6,7])
salary_range = st.selectbox("Salary Range[0:<=25K 1:25k-60k 2:>=60k]",[0,1,2])
hh_drivers = st.number_input("Household Drivers",1,9,2)
quoted_premium = st.number_input("Quoted Premium",0,5000,900)

if st.button("Run Decision Engine"):

    risk_features = pd.DataFrame([[
        prev_acc,
        prev_cit,
        exp,
        age,
        veh_use,
        miles
    ]],columns=[
        "Prev_Accidents",
        "Prev_Citations",
        "Driving_Exp",
        "Driver_Age",
        "Veh_Usage",
        "Annual_Miles_Range"
    ])

    risk_scaled = sc.transform(risk_features)

    risk_tier = agent1_model.predict(risk_scaled)[0]

    bind_features = pd.DataFrame([[
        re_quote,
        coverage,
        agent_type,
        region,
        salary_range,
        hh_drivers
    ]],columns=[
        "Re_Quote",
        "Coverage",
        "Agent_Type",
        "Region",
        "Sal_Range",
        "HH_Drivers"
    ])

    bind_score = agent2_model.predict_proba(bind_features)[0][1]

    if (risk_tier == "Low") and (bind_score > 0.7):
        decision = "AutoApprove"
    elif (0.4 <= bind_score <= 0.7):
        decision = "AgentFollowUp"
    else:
        decision = "Escalate"

    st.subheader("Decision Result")

    st.write("Risk Tier:",risk_tier)
    st.write("Bind Score:",round(bind_score,3))
    st.write("Decision:",decision)