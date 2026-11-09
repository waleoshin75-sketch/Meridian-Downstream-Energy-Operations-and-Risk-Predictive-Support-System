"""
Meridian Downstream Energy: Operations and Risk Predictive Support System
==========================================================================
A simple Streamlit app with one page per business objective:

    1  Maintenance risk   -> will this asset break down next month?
    2  Margin forecast    -> what gross margin will this refinery run earn?
    3  Delivery risk      -> will this fuel trip arrive late?
    4  Off-spec risk      -> will this blend batch fail specification?
    5  Inventory check    -> is this tank-day a possible loss, theft or metering error?

HOW TO RUN
    1. Put this file (app.py) in a folder, and unzip meridian_model_package.zip next to it:

           meridian_app/
               app.py
               meridian_model_package/      <- the folder from your Step 18 download
                   obj1_model.joblib, obj1_scaler.joblib, obj1_features.joblib ... (and so on)
                   model_registry.joblib

    2. In the VS Code terminal:
           pip install streamlit pandas numpy joblib scikit-learn==1.9.1
           streamlit run app.py

How it works (short version): you type in the things a person would know (readings, distances, volumes).
The app turns them into the exact columns the saved model was trained on, fills anything you did not
enter with the typical value from the training data, scales the numbers with the saved scaler, and asks
the saved model for its answer. The scores are decision support: they rank risk, they do not guarantee outcomes.
"""
import os
import calendar
import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# ----------------------------------------------------------------------------------------------
# 1. SETTINGS you may want to change
# ----------------------------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meridian_model_package")

# Typical values used in the explanations. They come from the project data (Steps 15 and 19).
# Dollar figures marked "assumed" were assumptions in Step 15: replace them with your real costs.
COST = {
    "breakdown_repair_usd": 29842,   # average repair cost of a breakdown (from the data)
    "breakdown_downtime_h": 25,      # average downtime hours of a breakdown (from the data)
    "preventive_job_usd": 1941,      # average preventive job cost (from the data)
    "late_penalty_usd": 600,         # assumed penalty per late trip
    "rework_usd": 11699,             # average rework cost of an off-spec batch (from the data)
    "quality_check_usd": 250,        # assumed cost of an extra check
}
# Risk scores at which a case moves up a level. "Act now" = the action level saved with each model (the cases the project plan acts on).
# "Watch" = the next tier below it (roughly the top 20-30% of cases in the test data). Taken from the test results in Step 12.
WATCH_LEVEL = {1: 0.657, 3: 0.356, 4: 0.480}
AVERAGE_MARGIN = 11.5                # average gross margin in the data, USD per barrel
TYPICAL_DOSE_PPM, TYPICAL_BLEND_TEMP_C = 199.7, 38.0   # typical settings in the training period (Step 3)

st.set_page_config(page_title="Meridian Operations and Risk Support", page_icon="⛽", layout="wide")


# ----------------------------------------------------------------------------------------------
# 2. LOAD THE SAVED MODELS (once, then kept in memory)
# ----------------------------------------------------------------------------------------------
@st.cache_resource
def load_package():
    """Reads model, scaler and feature list for each objective, plus the registry."""
    package = {}
    for n in range(1, 6):
        package[n] = (joblib.load(os.path.join(MODEL_DIR, f"obj{n}_model.joblib")),
                      joblib.load(os.path.join(MODEL_DIR, f"obj{n}_scaler.joblib")),
                      joblib.load(os.path.join(MODEL_DIR, f"obj{n}_features.joblib")))
    registry = joblib.load(os.path.join(MODEL_DIR, "model_registry.joblib"))
    return package, registry


if not os.path.isdir(MODEL_DIR):
    st.error(f"The folder 'meridian_model_package' was not found next to app.py (looked in: {MODEL_DIR}). "
             "Unzip the file you downloaded in Step 18 into the same folder as app.py.")
    st.stop()
PACKAGE, REGISTRY = load_package()


# ----------------------------------------------------------------------------------------------
# 3. HELPERS shared by all pages
# ----------------------------------------------------------------------------------------------
def category(feats, label, prefix, key, exclude=(), default=None):
    """Drop-down for a category the model knows (made from its one-hot columns, e.g. country_Nigeria).
    Returns the chosen name and a dict of 0/1 values for those columns. 'Other' sets them all to 0."""
    cols = [c for c in feats["feature_names"] if c.startswith(prefix + "_") and not c.startswith(tuple(exclude))]
    names = [c[len(prefix) + 1:].replace("_", " ") for c in cols]
    index = names.index(default) if default in names else 0
    choice = st.selectbox(label, names + ["Other / not listed"], index=index, key=key)
    return choice, {c: float(n == choice) for c, n in zip(cols, names)}


def run_model(n, values):
    """Builds the one-row table the model expects. Anything not entered gets the typical training value."""
    model, scaler, feats = PACKAGE[n]
    row = {**feats["fill_values"], **values}
    X = pd.DataFrame([row])[feats["feature_names"]]
    X[feats["scaled_columns"]] = scaler.transform(X[feats["scaled_columns"]])    # same scaling as training
    return model, X[feats["model_input_columns"]]


def risk_score(n, values):
    """Risk score from 0 to 1 for the classification models (objectives 1, 3, 4)."""
    model, X = run_model(n, values)
    return float(model.predict_proba(X)[0, 1])


def cutoff(n):
    return float(REGISTRY["objectives"][str(n)]["decision_cutoff"])


def show_risk(n, score, factors, advice_high, advice_watch, advice_ok):
    """Shows the risk score, the action level, a verdict and the warning signs found in the inputs."""
    cut = cutoff(n)
    level = "high" if score >= cut else ("watch" if score >= WATCH_LEVEL[n] else "normal")
    a, b, c = st.columns(3)
    a.metric("Risk score", f"{score:.0%}")
    b.metric("Action level", f"{cut:.0%}")
    c.metric("Verdict", {"high": "Act now", "watch": "Watch", "normal": "Normal"}[level])
    st.progress(min(max(score, 0.0), 1.0))
    if level == "high":
        st.error(advice_high)
    elif level == "watch":
        st.warning(advice_watch)
    else:
        st.success(advice_ok)
    st.markdown("**Warning signs in these inputs:** " + ("; ".join(factors) if factors else "none of the usual warning signs"))
    st.caption("The risk score ranks cases from low to high. It is not an exact probability. "
               f"A score at or above {cut:.0%} puts the case in the group that is acted on in the project plan; "
               f"{WATCH_LEVEL[n]:.0%} to {cut:.0%} is the watch tier.")
    return {"high": "Act now", "watch": "Watch", "normal": "Normal"}[level]


def add_to_log(objective, case, result, verdict):
    if "log" not in st.session_state:
        st.session_state["log"] = []
    st.session_state["log"].append({"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "objective": objective,
                                    "case": case, "result": result, "verdict": verdict})


def feats_of(n):
    return PACKAGE[n][2]


TODAY = datetime.date.today()


# ----------------------------------------------------------------------------------------------
# 4. PAGES
# ----------------------------------------------------------------------------------------------
def page_home():
    st.title("⛽ Meridian Downstream Energy")
    st.subheader("Operations and Risk Predictive Support System")
    st.write("Five prediction tools built from the Meridian machine learning project. Pick one in the menu on the left, "
             "enter the details, and press the button to get a score and a suggested action.")
    rows = []
    names = {1: "Cut unplanned downtime and maintenance cost", 2: "Maximise refinery yield and gross margin",
             3: "Raise on-time delivery and cut distribution cost", 4: "Reduce off-spec product and rework cost",
             5: "Detect product loss, theft and metering errors"}
    for n in range(1, 6):
        o = REGISTRY["objectives"][str(n)]
        r = o["test_results"]
        if n == 2:
            res = f"error {r['MAE']:.2f} USD/bbl, R2 {r['R2']:.2f}"
        else:
            res = f"ROC AUC {r['ROC_AUC']:.2f}, PR AUC {r['PR_AUC']:.2f}"
        rows.append({"#": n, "Business objective": names[n], "Model": o["chosen_model"], "Test result": res})
    st.table(pd.DataFrame(rows).set_index("#"))
    with st.expander("Please read: how to use these scores"):
        st.markdown(
            "- The scores support decisions; engineers, planners and supervisors still make them.\n"
            "- Breakdown and off-spec models are **ranking tools**: they are right more often than chance, not every time.\n"
            "- For inventory, the simple **gap over 1%** rule was more accurate than the machine learning model, so the page shows both.\n"
            "- The project data is synthetic, and the money figures use assumed costs (see the `COST` box at the top of app.py).")
    saved = REGISTRY.get("scikit_learn_version")
    if saved and saved != sklearn.__version__:
        st.warning(f"The models were saved with scikit-learn {saved}, but this computer has {sklearn.__version__}. "
                   f"If you see errors, run: pip install scikit-learn=={saved}")


# ---------------------------------------------------------------- Objective 1
def page_maintenance():
    st.header("1. Maintenance risk")
    st.write("Will this asset have a breakdown next month? Enter its latest sensor readings and history.")
    f = feats_of(1)
    with st.form("form1"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**The asset**")
            atype, d_type = category(f, "Asset type", "asset_type", "a_type", default="Pump")
            crit, d_crit = category(f, "Criticality (A = most critical)", "criticality", "a_crit", default="B")
            site, d_site = category(f, "Site type", "site_type", "a_site", default="Terminal")
            country, d_country = category(f, "Country", "country", "a_country", default="Nigeria")
            maker, d_maker = category(f, "Manufacturer", "manufacturer", "a_maker", default="Siemens")
            age = st.number_input("Asset age (years)", 0.0, 60.0, 13.0, 0.5)
            life = st.number_input("Design life (years)", 1.0, 60.0, 25.0, 1.0)
        with c2:
            st.markdown("**Latest sensor readings**")
            vib = st.number_input("Vibration (mm/s)", 0.0, 30.0, 3.1, 0.1)
            vib3 = st.number_input("Vibration, 3-month average (mm/s)", 0.0, 30.0, 3.1, 0.1)
            temp = st.number_input("Temperature (°C)", 0.0, 250.0, 75.6, 0.5)
            temp3 = st.number_input("Temperature, 3-month average (°C)", 0.0, 250.0, 75.5, 0.5)
            press = st.number_input("Pressure (bar)", 0.0, 60.0, 12.5, 0.5)
            lube = st.number_input("Lubricant condition score (0-100)", 0.0, 100.0, 83.0, 1.0)
        with c3:
            st.markdown("**Recent history**")
            hours = st.number_input("Operating hours this month", 0.0, 744.0, 650.0, 10.0)
            month = st.number_input("Month of year (1-12)", 1, 12, TODAY.month, 1)
            bd6 = st.number_input("Breakdowns in the last 6 months", 0, 10, 0, 1)
            pm6 = st.number_input("Preventive jobs in the last 6 months", 0, 10, 0, 1)
            since = st.number_input("Months since last breakdown (36 = none)", 0.0, 36.0, 36.0, 1.0)
        go = st.form_submit_button("Check this asset")
    if not go:
        return
    values = {**d_type, **d_crit, **d_site, **d_country, **d_maker,
              "asset_age_years": age, "life_used_ratio": age / life,
              "avg_vibration_mm_s": vib, "vib_avg_3m": vib3, "avg_temperature_c": temp, "temp_avg_3m": temp3,
              "avg_pressure_bar": press, "press_avg_3m": press, "lubricant_condition_score": lube, "lube_avg_3m": lube,
              "operating_hours": hours, "utilisation_pct": hours / (calendar.monthrange(TODAY.year, int(month))[1] * 24) * 100,
              "month_of_year": month, "breakdowns_last_6m": bd6, "preventive_jobs_last_6m": pm6, "months_since_last_breakdown": since,
              "vib_alert": float(vib >= 4.5), "lube_poor": float(lube < 60)}
    factors = []
    if temp3 >= 79.8: factors.append("high 3-month temperature (the hottest quarter of assets break down about 7x as often)")
    if vib >= 4.5: factors.append("vibration alert (about 3.7x the normal breakdown rate)")
    if lube < 60: factors.append("poor lubricant condition (about 3.4x)")
    if age >= 19: factors.append("old asset (about 1.6x)")
    if bd6 > 0: factors.append(f"{int(bd6)} breakdown(s) in the last 6 months")
    score = risk_score(1, values)
    verdict = show_risk(1, score, factors,
                        f"Schedule a preventive inspection or job now. A preventive job costs about ${COST['preventive_job_usd']:,}; "
                        f"a breakdown costs about ${COST['breakdown_repair_usd']:,} plus roughly {COST['breakdown_downtime_h']} hours of downtime.",
                        "Elevated risk. Add this asset to the next inspection round and watch vibration and temperature.",
                        "No action needed beyond the normal maintenance plan.")
    add_to_log("1 Maintenance", f"{atype}, {country}", f"risk {score:.0%}", verdict)


# ---------------------------------------------------------------- Objective 2
def page_margin():
    st.header("2. Margin forecast")
    st.write("What gross margin will this refinery run earn? Enter the crude, plant conditions and market prices.")
    f = feats_of(2)
    with st.form("form2"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Crude and site**")
            grade, d_grade = category(f, "Crude grade", "crude_grade", "m_grade", default="Bonny Light")
            country, d_country = category(f, "Country", "country", "m_country", default="Nigeria")
            api = st.number_input("Crude API gravity", 10.0, 60.0, 34.8, 0.1)
            sulphur = st.number_input("Crude sulphur (%)", 0.0, 5.0, 0.24, 0.05)
            cost = st.number_input("Crude cost (USD/bbl)", 20.0, 150.0, 74.0, 0.5)
        with c2:
            st.markdown("**Plant conditions**")
            thr = st.number_input("Throughput (barrels)", 1000.0, 600000.0, 152000.0, 1000.0)
            util = st.number_input("Capacity utilisation (%)", 10.0, 120.0, 85.0, 1.0)
            furnace = st.number_input("Furnace outlet temperature (°C)", 200.0, 500.0, 365.0, 1.0)
            colp = st.number_input("Column pressure (bar)", 0.5, 5.0, 1.8, 0.1)
            energy = st.number_input("Energy use (GJ/bbl)", 0.1, 3.0, 0.65, 0.05)
            down = st.number_input("Unplanned downtime (hours)", 0.0, 48.0, 1.0, 0.5)
        with c3:
            st.markdown("**Market and date**")
            crack = st.number_input("Crack spread today (USD/bbl)", -5.0, 40.0, 13.7, 0.1)
            crack7 = st.number_input("Crack spread, 7-day average (USD/bbl)", -5.0, 40.0, 13.65, 0.1)
            prev = st.number_input("Average margin of the last 7 days (USD/bbl)", -10.0, 40.0, 12.0, 0.1)
            day = st.date_input("Run date", TODAY)
        go = st.form_submit_button("Forecast margin")
    if not go:
        return
    values = {**d_grade, **d_country, "crude_api_gravity": api, "crude_sulphur_pct": sulphur, "crude_cost_usd_per_bbl": cost,
              "throughput_bbl": thr, "capacity_utilisation": util / 100, "furnace_outlet_temp_c": furnace, "column_pressure_bar": colp,
              "energy_use_gj_per_bbl": energy, "unplanned_downtime_hours": down, "crack_spread_usd_per_bbl": crack,
              "crack_spread_7d_avg": crack7, "margin_prev_7d_avg": prev, "month": day.month, "quarter": (day.month - 1) // 3 + 1,
              "day_of_week": day.weekday(), "energy_x_sulphur": energy * sulphur,
              "sour_crude": float(sulphur > 1.5), "heavy_crude": float(api < 32), "had_downtime": float(down > 0)}
    model, X = run_model(2, values)
    pred = float(model.predict(X)[0])
    err = float(REGISTRY["objectives"]["2"]["test_results"]["MAE"])
    a, b, c = st.columns(3)
    a.metric("Forecast margin", f"{pred:.2f} USD/bbl", f"{pred - AVERAGE_MARGIN:+.2f} vs average ({AVERAGE_MARGIN})")
    b.metric("Likely range", f"{pred - err:.2f} to {pred + err:.2f}", "typical error ±" + f"{err:.2f}", delta_color="off")
    c.metric("Margin for this run", f"${pred * thr:,.0f}", f"{thr:,.0f} barrels", delta_color="off")
    if pred >= AVERAGE_MARGIN + 1:
        st.success("Above-average margin expected. Protect this run: avoid downtime and keep throughput steady.")
    elif pred <= AVERAGE_MARGIN - 1:
        st.warning("Below-average margin expected. Check the crude choice, crack spread and energy use before committing the run.")
    else:
        st.info("Margin is expected to be close to the average.")
    notes = []
    if sulphur > 1.5: notes.append("sour crude (earned about 3.5 USD/bbl less than sweet crude in the data)")
    if api < 32: notes.append("heavy crude")
    if crack < 12: notes.append("low crack spread (the strongest driver of margin)")
    if down > 0: notes.append("unplanned downtime (costs throughput even if margin per barrel holds)")
    st.markdown("**Things pulling the forecast down:** " + ("; ".join(notes) if notes else "none of the usual ones"))
    st.caption("This is a forecast, not a promise. On the project's test data it was off by about this much on average.")
    add_to_log("2 Margin", f"{grade}, {country}", f"{pred:.2f} USD/bbl", "Above avg" if pred >= AVERAGE_MARGIN + 1 else "Below avg" if pred <= AVERAGE_MARGIN - 1 else "Average")


# ---------------------------------------------------------------- Objective 3
def page_delivery():
    st.header("3. Delivery risk")
    st.write("Will this fuel trip arrive more than 30 minutes late? Enter the trip details at dispatch.")
    f = feats_of(3)
    with st.form("form3"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Route and load**")
            product, d_prod = category(f, "Product", "product", "d_prod", default="AGO")
            site, d_site = category(f, "Origin site type", "site_type", "d_site", default="Depot")
            country, d_country = category(f, "Origin country", "country", "d_country", default="Nigeria")
            region, d_region = category(f, "Destination region", "destination_region", "d_region", default="West Africa")
            volume = st.number_input("Volume (litres)", 1000.0, 60000.0, 30000.0, 1000.0)
        with c2:
            st.markdown("**Conditions**")
            distance = st.number_input("Distance (km)", 1.0, 1500.0, 108.0, 5.0)
            traffic = st.slider("Traffic index (0 = clear, 10 = gridlock)", 0.0, 10.0, 5.0, 0.5)
            weather, d_weather = category(f, "Weather", "weather", "d_weather", default="Clear")
            promised = st.number_input("Promised trip time (hours)", 0.5, 72.0, 4.5, 0.5)
        with c3:
            st.markdown("**Truck, driver, time**")
            truck = st.number_input("Truck age (years)", 0.0, 30.0, 7.0, 1.0)
            driver = st.number_input("Driver experience (years)", 0.0, 45.0, 8.0, 1.0)
            day = st.date_input("Dispatch date", TODAY)
            hour = st.slider("Dispatch hour (0-23)", 0, 23, 10)
        go = st.form_submit_button("Check this trip")
    if not go:
        return
    bad_weather = weather in ("Rain", "Storm", "Harmattan dust")
    values = {**d_prod, **d_site, **d_country, **d_region, **d_weather,
              "volume_litres": volume, "distance_km": distance, "traffic_index": traffic, "promised_hours": promised,
              "promised_speed_kmh": distance / promised, "truck_age_years": truck, "driver_experience_years": driver,
              "dispatch_hour": hour, "dispatch_day": day.weekday(), "dispatch_month": day.month,
              "traffic_x_distance": traffic * distance, "is_weekend": float(day.weekday() >= 5),
              "peak_hour": float(hour in (6, 7, 8, 9, 16, 17, 18, 19)), "long_haul": float(distance > 200),
              "old_truck": float(truck > 10), "novice_driver": float(driver < 3), "bad_weather": float(bad_weather)}
    factors = []
    if traffic >= 9: factors.append("very heavy traffic (such trips are late about 7x as often as light-traffic trips)")
    if distance >= 175: factors.append("long distance (about 3x)")
    if bad_weather: factors.append(f"bad weather: {weather.lower()} (about 2.4x)")
    if truck > 10: factors.append("older truck")
    if driver < 3: factors.append("novice driver")
    if distance / promised > 35: factors.append("tight promise (very high average speed needed)")
    score = risk_score(3, values)
    verdict = show_risk(3, score, factors,
                        f"High chance of a late arrival. Add a time buffer to the promised time, or tell the customer early. "
                        f"(A late trip is assumed to cost about ${COST['late_penalty_usd']} in penalties.)",
                        "Some risk. Keep an eye on traffic and weather, and prepare a revised time if needed.",
                        "Low risk of lateness. Dispatch as planned.")
    add_to_log("3 Delivery", f"{product}, {distance:.0f} km", f"risk {score:.0%}", verdict)


# ---------------------------------------------------------------- Objective 4
def page_quality():
    st.header("4. Off-spec risk")
    st.write("Will this blend batch fail the product specification? Enter the batch details before release.")
    f = feats_of(4)
    with st.form("form4"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Product and feed**")
            product, d_prod = category(f, "Product", "product", "q_prod", exclude=("product_group",), default="AGO")
            grade, d_grade = category(f, "Feed crude grade", "crude_grade", "q_grade", default="Bonny Light")
            site, d_site = category(f, "Site type", "site_type", "q_site", default="Terminal")
            country, d_country = category(f, "Country", "country", "q_country", default="Nigeria")
        with c2:
            st.markdown("**Blend settings**")
            sulphur = st.number_input("Feed crude sulphur (%)", 0.0, 5.0, 0.24, 0.05)
            dose = st.number_input("Additive dose (ppm)", 50.0, 400.0, 200.0, 1.0)
            temp = st.number_input("Blend temperature (°C)", 10.0, 80.0, 38.0, 0.5)
        with c3:
            st.markdown("**Quality indicators**")
            water = st.number_input("Water content (ppm)", 0.0, 1500.0, 40.0, 1.0)
            storage = st.number_input("Storage time (days)", 0.0, 60.0, 7.0, 1.0)
            day = st.date_input("Test date", TODAY)
        go = st.form_submit_button("Check this batch")
    if not go:
        return
    gasoline = product == "PMS"
    values = {**d_prod, **d_grade, **d_site, **d_country,
              "feed_crude_sulphur_pct": sulphur, "additive_dose_ppm": dose, "blend_temp_c": temp, "storage_days": storage,
              "water_content_ppm": water, "dose_deviation_ppm": abs(dose - TYPICAL_DOSE_PPM), "blend_temp_deviation": temp - TYPICAL_BLEND_TEMP_C,
              "test_month": day.month, "test_quarter": (day.month - 1) // 3 + 1,
              "high_water": float(water > 80), "long_storage": float(storage > 14), "sour_feed": float(sulphur > 1.0),
              "product_group_Gasoline": float(gasoline), "product_group_Middle_distillate": float(not gasoline)}
    factors = []
    if water > 80: factors.append("high water content (such batches fail about 4.7x as often)")
    elif water >= 60: factors.append("water content in the top quarter (about 6x the failure rate of the driest batches)")
    if sulphur > 1.0: factors.append("sour feed crude (about 2.7x)")
    if storage > 14: factors.append("long storage (about 1.5x)")
    if abs(dose - TYPICAL_DOSE_PPM) > 40: factors.append("additive dose far from the usual setting")
    score = risk_score(4, values)
    verdict = show_risk(4, score, factors,
                        f"Hold the batch for an extra check and blend correction. Rework of an off-spec batch costs about "
                        f"${COST['rework_usd']:,}; an extra check is assumed to cost about ${COST['quality_check_usd']}.",
                        "Some risk. Re-test water content and sulphur before release.",
                        "Low risk. Release as normal.")
    add_to_log("4 Quality", f"{product}, water {water:.0f} ppm", f"risk {score:.0%}", verdict)


# ---------------------------------------------------------------- Objective 5
def page_inventory():
    st.header("5. Inventory check")
    st.write("Does this tank-day look like a loss, theft or metering problem? Compare the book stock with the measured stock.")
    f = feats_of(5)
    with st.form("form5"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Tank**")
            product, d_prod = category(f, "Product", "product", "i_prod", default="AGO")
            site, d_site = category(f, "Site type", "site_type", "i_site", default="Depot")
            country, d_country = category(f, "Country", "country", "i_country", default="Nigeria")
            day = st.date_input("Record date", TODAY)
        with c2:
            st.markdown("**Volumes (kilolitres)**")
            opening = st.number_input("Opening stock", 1.0, 200000.0, 19000.0, 100.0)
            receipts = st.number_input("Receipts", 0.0, 100000.0, 0.0, 50.0)
            dispatches = st.number_input("Dispatches", 0.0, 100000.0, 500.0, 10.0)
            book = st.number_input("Book closing stock (what the records say)", 1.0, 200000.0, 18500.0, 100.0)
            measured = st.number_input("Measured closing stock (what the gauge says)", 1.0, 200000.0, 18500.0, 100.0)
        with c3:
            st.markdown("**Meter and history**")
            temp = st.number_input("Tank temperature (°C)", 0.0, 70.0, 30.0, 0.5)
            calib = st.number_input("Days since meter calibration", 0.0, 1000.0, 180.0, 10.0)
            mean14 = st.number_input("This tank's usual gap, last 14 days (%)", -5.0, 5.0, 0.0, 0.05)
            std14 = st.number_input("Spread of that gap, last 14 days (%)", 0.0, 5.0, 0.13, 0.01)
        go = st.form_submit_button("Check this tank-day")
    if not go:
        return
    gap_kl = measured - book
    gap_pct = gap_kl / book * 100
    arith = opening + receipts - dispatches - book
    values = {**d_prod, **d_site, **d_country,
              "opening_stock_kl": opening, "receipts_kl": receipts, "dispatches_kl": dispatches, "book_closing_kl": book,
              "measured_closing_kl": measured, "tank_temp_c": temp, "days_since_meter_calibration": calib,
              "gap_kl": gap_kl, "gap_pct": gap_pct, "abs_gap_pct": abs(gap_pct), "measured_to_book_ratio": measured / book,
              "book_arithmetic_gap": arith, "dispatch_rate_pct": dispatches / opening * 100, "received_today": float(receipts > 0),
              "gap_pct_mean_14d": mean14, "gap_pct_std_14d": std14, "gap_zscore": (gap_pct - mean14) / max(std14, 0.05),
              "record_month": day.month, "record_day": day.weekday()}
    model, X = run_model(5, values)
    model_flag = int(model.predict(X)[0]) == -1                  # Isolation Forest: -1 means "unusual"
    unusual = float(-model.decision_function(X)[0])               # higher = more unusual
    rule_flag = abs(gap_pct) > 1
    recording_error = abs(gap_pct) > 50

    a, b, c = st.columns(3)
    a.metric("Gap (measured minus book)", f"{gap_kl:,.0f} kl", f"{gap_pct:+.2f}%", delta_color="off")
    b.metric("Gap rule (over 1%)", "FLAG" if rule_flag else "OK")
    c.metric("Machine learning model", "FLAG" if model_flag else "OK", f"unusualness {unusual:+.2f}", delta_color="off")
    if recording_error:
        st.error("The measured stock is more than 50% away from book. This is almost certainly a **recording or reading error** "
                 "(for example a wrong digit or unit). Re-check the gauge reading and the entry before investigating a loss.")
    elif rule_flag:
        st.error("The gap is over 1% of book stock. Investigate this tank-day: check meter calibration, unrecorded receipts or "
                 "dispatches, leaks and possible theft. The gap rule is the main control; it was the most accurate in the project.")
    elif model_flag:
        st.warning("The gap is under 1%, but the model finds the pattern unusual. Use it as a second opinion: review the tank's "
                   "recent history if staff time allows.")
    else:
        st.success("No sign of a problem. Book and measured stock agree.")
    notes = []
    if abs(arith) > 0.01 * opening: notes.append(f"the book figures do not add up (opening + receipts - dispatches differs from book closing by {arith:,.0f} kl)")
    if abs(gap_pct - mean14) / max(std14, 0.05) > 3: notes.append("this gap is far outside the tank's usual range")
    if calib > 270: notes.append("meter not calibrated for a long time (note: in the project data this did not explain gaps by itself)")
    st.markdown("**Other things noticed:** " + ("; ".join(notes) if notes else "nothing else"))
    add_to_log("5 Inventory", f"{product}, {site}", f"gap {gap_pct:+.2f}%",
               "Recording error" if recording_error else "Investigate" if rule_flag else "Review" if model_flag else "OK")


# ---------------------------------------------------------------- Log
def page_log():
    st.header("Prediction log")
    st.write("Every check you run in this session is listed here. Download it to keep a record.")
    log = st.session_state.get("log", [])
    if not log:
        st.info("Nothing yet. Run a check on one of the pages.")
        return
    table = pd.DataFrame(log)
    st.dataframe(table)
    st.download_button("Download log as CSV", table.to_csv(index=False).encode("utf-8"), "meridian_prediction_log.csv", "text/csv")
    if st.button("Clear log"):
        st.session_state["log"] = []
        st.rerun()


# ----------------------------------------------------------------------------------------------
# 5. MENU
# ----------------------------------------------------------------------------------------------
PAGES = {"Home": page_home, "1  Maintenance risk": page_maintenance, "2  Margin forecast": page_margin,
         "3  Delivery risk": page_delivery, "4  Off-spec risk": page_quality, "5  Inventory check": page_inventory,
         "Prediction log": page_log}
st.sidebar.title("Meridian Downstream")
st.sidebar.caption("Operations and Risk Predictive Support System")
choice = st.sidebar.radio("Go to", list(PAGES))
PAGES[choice]()
st.sidebar.caption("Decision support only. Built from the Meridian ML project (synthetic data).")