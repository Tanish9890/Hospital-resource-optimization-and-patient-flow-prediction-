# main.py

import pandas as pd
import os
from prophet import Prophet
import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpVariable

# =============================
# STEP 1: Load Dataset
# =============================

file_csv = "healthcare_dataset.csv"
file_excel = "healthcare_dataset.xlsx"

if os.path.exists(file_csv):
    df = pd.read_csv(file_csv)
    print("✅ Loaded CSV file")
elif os.path.exists(file_excel):
    df = pd.read_excel(file_excel)
    print("✅ Loaded Excel file")
else:
    raise FileNotFoundError("❌ No dataset found. Place healthcare_dataset.csv or .xlsx in the same folder as main.py")

# Ensure required columns exist
required_cols = ['Date of Admission', 'Medical Condition']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# Convert date
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Year'] = df['Date of Admission'].dt.year

print("\n📊 Dataset Preview:")
print(df.head())

# =============================
# STEP 2: Forecast Patient Inflow Year-wise
# =============================

years = df['Year'].unique()
departments = df['Medical Condition'].unique()
forecast_results = {}

for year in years:
    print(f"\n--- Forecast for Year {year} ---")
    yearly_data = df[df['Year'] == year]
    
    # Group by date
    daily_data = yearly_data.groupby('Date of Admission').size().reset_index(name='patients')
    forecast_df = daily_data.rename(columns={'Date of Admission': 'ds', 'patients': 'y'})
    
    # Fit Prophet model
    model = Prophet()
    model.fit(forecast_df)
    
    # Predict next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Save forecast
    forecast_results[year] = forecast
    forecast.to_csv(f"forecast_{year}.csv", index=False)
    
    # Plot forecast
    fig = model.plot(forecast)
    plt.title(f"📈 Patient Inflow Forecast for Year {year}")
    plt.show()
    
    # Optional: plot components
    fig2 = model.plot_components(forecast)
    plt.show()

# =============================
# STEP 3: Resource Optimization Year-wise
# =============================

total_beds = 200  # Example: hospital capacity

for year in years:
    print(f"\n--- Bed Allocation for Year {year} ---")
    yearly_data = df[df['Year'] == year]
    
    # Get latest day of that year
    latest_day = yearly_data['Date of Admission'].max()
    predicted_patients = yearly_data.groupby('Medical Condition').size().to_dict()
    
    # LP Model
    opt_model = LpProblem(name=f"resource-optimization-{year}", sense=LpMaximize)
    beds = {dept: LpVariable(name=f"beds_{dept}", lowBound=0) for dept in departments}
    
    # Objective: maximize coverage
    opt_model += sum(beds[dept] for dept in departments)
    
    # Constraint: total beds limit
    opt_model += sum(beds[dept] for dept in departments) <= total_beds
    
    # Constraint: beds per department <= predicted demand
    for dept in departments:
        opt_model += beds[dept] <= predicted_patients.get(dept, 0)
    
    opt_model.solve()
    
    # Show allocation
    allocation = {}
    for dept in departments:
        allocation[dept] = beds[dept].value()
        print(f"{dept}: {beds[dept].value()} beds")
    
    # Save allocation to CSV
    pd.DataFrame(list(allocation.items()), columns=['Department', 'Beds']).to_csv(f"bed_allocation_{year}.csv", index=False)
# =============================
# STEP 4: Resource Optimization Year-wise
# =============================

total_beds = 200        # Example: hospital capacity
total_doctors = 50      # Example: number of doctors
total_equipment = 100   # Example: available equipment

for year in years:
    print(f"\n--- Resource Allocation for Year {year} ---")
    yearly_data = df[df['Year'] == year]
    
    # Get latest day of that year
    latest_day = yearly_data['Date of Admission'].max()
    predicted_patients = yearly_data[yearly_data['Date of Admission'] == latest_day] \
                            .groupby('Medical Condition').size().to_dict()
    
    # LP Model
    opt_model = LpProblem(name=f"resource-optimization-{year}", sense=LpMaximize)
    
    beds = {dept: LpVariable(name=f"beds_{dept}", lowBound=0) for dept in departments}
    doctors = {dept: LpVariable(name=f"doctors_{dept}", lowBound=0) for dept in departments}
    equipment = {dept: LpVariable(name=f"equip_{dept}", lowBound=0) for dept in departments}
    
    # Objective: maximize overall patient coverage
    opt_model += sum(beds[dept] + doctors[dept] + equipment[dept] for dept in departments)
    
    # Constraints: total hospital limits
    opt_model += sum(beds[dept] for dept in departments) <= total_beds
    opt_model += sum(doctors[dept] for dept in departments) <= total_doctors
    opt_model += sum(equipment[dept] for dept in departments) <= total_equipment
    
    # Constraint: department-wise demand
    for dept in departments:
        demand = predicted_patients.get(dept, 0)
        opt_model += beds[dept] <= demand
        opt_model += doctors[dept] <= demand
        opt_model += equipment[dept] <= demand
    
    opt_model.solve()
    
    # Show allocation
    allocation = []
    for dept in departments:
        allocation.append([dept, beds[dept].value(), doctors[dept].value(), equipment[dept].value()])
        print(f"{dept}: {beds[dept].value()} beds, {doctors[dept].value()} doctors, {equipment[dept].value()} equipment")
    
    # Save allocation to CSV
    pd.DataFrame(allocation, columns=['Department', 'Beds', 'Doctors', 'Equipment']) \
        .to_csv(f"resource_allocation_{year}.csv", index=False)

# =============================
# STEP 4 (REPLACE THIS BLOCK): Weighted, ratio-based Resource Allocation
#  - Beds, Doctors, Equipment will differ by:
#    (a) objective weights, and
#    (b) per-patient ratios (global + department overrides)
#  - Output: outputs/resource_allocation_weighted.xlsx (one sheet per Year)
# =============================
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, LpInteger, lpSum, LpStatus
import os

OUTPUT_DIR = "outputs"  # Define the output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------- CONFIG: tune these -----------
# Prefer some resources over others (solver maximizes this)
WEIGHT_BEDS      = 1.0
WEIGHT_DOCTORS   = 2.0
WEIGHT_EQUIPMENT = 0.6

# Base per-patient ratios (fallback if department not in overrides)
# e.g., each patient needs up to 1 bed, 0.25 doctor, 0.5 equipment units.
BASE_BED_PER_PATIENT    = 1.0
BASE_DOC_PER_PATIENT    = 0.25
BASE_EQUIP_PER_PATIENT  = 0.5

# Department-specific overrides (OPTIONAL). Omit missing depts; they’ll use BASE_*.
# Example below assumes common hospital units; adjust to your dataset’s actual names.
DEPT_DOC_NEED = {
    "ICU": 0.8,
    "Surgery": 0.6,
    "Emergency": 0.5,
    "General Medicine": 0.2,
}
DEPT_EQUIP_NEED = {
    "ICU": 0.9,
    "Surgery": 0.7,
    "Emergency": 0.5,
    "General Medicine": 0.2,
}
DEPT_BED_NEED = {
    # keep default 1.0 unless a dept is day-care/outpatient style
    "Day Care": 0.3,
}

# Optional: minimum beds guaranteed for any department that has demand
MIN_BEDS_PER_ACTIVE_DEPT = 0

# Total caps (can keep your earlier TOTAL_* or override here)
TOTAL_BEDS = 200        # Example: hospital capacity
TOTAL_DOCTORS = 50      # Example: number of doctors
TOTAL_EQUIPMENT = 100   # Example: available equipment
# ----------------------------------------

excel_path = os.path.join(OUTPUT_DIR, "resource_allocation_weighted.xlsx")

# Define column names for date and department/condition
date_col = "Date of Admission"
cond_col = "Medical Condition"

with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    for year in years:
        print(f"\n--- Resource Allocation (weighted) for Year {year} ---")
        yearly = df[df["Year"] == year]
        if yearly.empty:
            print("No data for this year, skipping.")
            continue

        # Demand basis: LATEST DAY (peak proxy). Change to 'Annual total' if you prefer.
        latest_day = yearly[date_col].max()
        latest_slice = yearly[yearly[date_col] == latest_day]
        # demand_by_dept = patients count on that day
        demand_by_dept = latest_slice.groupby(cond_col).size().astype(int).to_dict()

        # Fallback if all zero: spread average daily demand evenly
        if sum(demand_by_dept.values()) == 0:
            per_day = yearly.groupby(date_col).size()
            avg_daily = int(round(per_day.mean())) if len(per_day) else 0
            present_depts = yearly[cond_col].unique().tolist()
            even = int(round(avg_daily / max(len(present_depts), 1)))
            demand_by_dept = {d: even for d in present_depts}

        departments_year = sorted(demand_by_dept.keys())

        # Build LP
        model = LpProblem(name=f"resource-optimization-weighted-{year}", sense=LpMaximize)

        beds = {d: LpVariable(f"beds_{year}_{d}", lowBound=0, cat=LpInteger) for d in departments_year}
        docs = {d: LpVariable(f"docs_{year}_{d}", lowBound=0, cat=LpInteger) for d in departments_year}
        equip = {d: LpVariable(f"equip_{year}_{d}", lowBound=0, cat=LpInteger) for d in departments_year}

        # Objective: weighted sum so resources aren’t mirrored
        model += lpSum([
            WEIGHT_BEDS * beds[d] + WEIGHT_DOCTORS * docs[d] + WEIGHT_EQUIPMENT * equip[d]
            for d in departments_year
        ])

        # Global caps
        model += lpSum([beds[d] for d in departments_year]) <= TOTAL_BEDS
        model += lpSum([docs[d] for d in departments_year]) <= TOTAL_DOCTORS
        model += lpSum([equip[d] for d in departments_year]) <= TOTAL_EQUIPMENT

        # Dept caps via per-patient ratios (override → base)
        for d in departments_year:
            demand = int(max(demand_by_dept.get(d, 0), 0))

            bed_need   = DEPT_BED_NEED.get(d, BASE_BED_PER_PATIENT)
            doc_need   = DEPT_DOC_NEED.get(d, BASE_DOC_PER_PATIENT)
            equip_need = DEPT_EQUIP_NEED.get(d, BASE_EQUIP_PER_PATIENT)

            # Upper bounds: each resource <= demand * need ratio
            model += beds[d]  <= int(round(demand * bed_need))
            model += docs[d]  <= int(round(demand * doc_need))
            model += equip[d] <= int(round(demand * equip_need))

            # Optional minimums
            if MIN_BEDS_PER_ACTIVE_DEPT and demand > 0:
                model += beds[d] >= MIN_BEDS_PER_ACTIVE_DEPT

        status = model.solve()
        print(f"LP status: {LpStatus[status]}")

        # Collect results
        rows = []
        for d in departments_year:
            rows.append([
                d,
                int(beds[d].value() or 0),
                int(docs[d].value() or 0),
                int(equip[d].value() or 0),
                int(demand_by_dept.get(d, 0)),
                DEPT_BED_NEED.get(d, BASE_BED_PER_PATIENT),
                DEPT_DOC_NEED.get(d, BASE_DOC_PER_PATIENT),
                DEPT_EQUIP_NEED.get(d, BASE_EQUIP_PER_PATIENT),
            ])

        res_df = pd.DataFrame(
            rows,
            columns=["Department", "Beds", "Doctors", "Equipment", "Demand (patients)",
                     "Bed_need_per_patient", "Doc_need_per_patient", "Equip_need_per_patient"]
        ).sort_values(["Beds","Doctors","Equipment"], ascending=False).reset_index(drop=True)

        # Write a sheet per year
        sheet_name = f"{year}" if len(str(year)) <= 31 else f"Y{str(year)[-31:]}"
        res_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Also keep a CSV per year if you want to keep your old flow
        res_csv = os.path.join(OUTPUT_DIR, f"resource_allocation_{year}_weighted.csv")
        res_df.to_csv(res_csv, index=False)
        print(f"📁 Saved: {res_csv}")

print(f"\n✅ Excel workbook saved: {excel_path}")
