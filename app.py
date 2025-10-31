# NutriSaarthi — app.py (Monthly 30-day planner; full meal slots incl. Dinner)
import streamlit as st
import pandas as pd
import re
from datetime import datetime
import os
from fpdf import FPDF
from io import BytesIO
import math
import random

# ---------- SANITIZE / PDF HELPERS ----------
def sanitize_text(s):
    if s is None:
        return ""
    try:
        s = str(s)
    except:
        s = ""
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    try:
        s.encode('latin-1')
        return s
    except UnicodeEncodeError:
        return s.encode('latin-1', 'replace').decode('latin-1')

# ---------- TARGETS & RULE ENGINE ----------
def compute_targets(weight, albumin, egfr=None, on_dialysis=False, hepatic_encephalopathy="No"):
    kcal = 30 * weight
    protein = 1.2 * weight
    try:
        if albumin is not None and float(albumin) < 3.5:
            protein = 1.5 * weight
    except:
        pass
    try:
        if egfr is not None and float(egfr) > 0:
            egfr_v = float(egfr)
            if egfr_v < 30 and not on_dialysis:
                protein = 0.8 * weight
            if on_dialysis:
                protein = 1.4 * weight
    except:
        pass
    return round(kcal), round(protein)

def apply_rules(labs, symptoms, primary_site=None, trismus_grade=0, egfr=None, on_dialysis=False, bilirubin=None, inr=None, hepatic_encephalopathy="No"):
    flags = []
    referrals = []
    neutropenic_mode = False
    texture = "normal"
    anc = labs.get("anc_x10e9_l", 5)
    try:
        anc = float(anc)
    except:
        anc = 5
    if anc < 1.5:
        neutropenic_mode = True
        flags.append("Neutropenic precautions required (ANC < 1.5)")
        if anc < 0.5:
            flags.append("Severe neutropenia — strict food safety required")
            referrals.append("Infectious diseases / treating clinician")
    try:
        muc = int(symptoms.get("mucositis_grade", 0))
    except:
        muc = 0
    if muc >= 2:
        if muc >= 3:
            texture = "liquid"
            flags.append("Mucositis grade ≥3 — liquid/puree diet recommended")
        else:
            texture = "soft"
            flags.append("Mucositis grade ≥2 — soft foods recommended")
    if trismus_grade >= 2 or (primary_site and isinstance(primary_site, str) and primary_site.lower().startswith("head")):
        if trismus_grade >= 3:
            texture = "liquid"
            flags.append(f"Trismus grade {trismus_grade} — puree/liquid diet; consider speech/swallow referral")
            referrals.append("Speech & Swallow therapy")
            referrals.append("Dietitian-led enteral feeding assessment")
        else:
            texture = "soft"
            flags.append(f"Trismus grade {trismus_grade} — soft/pureed diet recommended")
            referrals.append("Speech & Swallow therapy")
    try:
        if egfr is not None and float(egfr) > 0:
            egfr_v = float(egfr)
            if egfr_v < 30 and not on_dialysis:
                flags.append("Severe renal impairment (eGFR <30) — nephrology consult recommended; consider renal diet modifications")
                referrals.append("Nephrology")
            if on_dialysis:
                flags.append("On dialysis — higher protein needs; coordinate with nephrology/dietitian")
                referrals.append("Nephrology")
    except:
        pass
    try:
        if bilirubin is not None and float(bilirubin) > 3:
            flags.append("Marked hyperbilirubinaemia — hepatology consult recommended")
            referrals.append("Hepatology")
    except:
        pass
    try:
        if inr is not None and float(inr) > 1.5:
            flags.append("Coagulopathy (INR >1.5) — hepatology review recommended")
            referrals.append("Hepatology")
    except:
        pass
    if hepatic_encephalopathy and isinstance(hepatic_encephalopathy, str) and hepatic_encephalopathy.lower().startswith("y"):
        flags.append("History of hepatic encephalopathy — individualize protein prescription; hepatology consult")
        referrals.append("Hepatology")
    try:
        if float(labs.get("albumin_g_dl", 4)) < 3.5:
            flags.append("Low albumin — consider increasing protein intake if clinically appropriate")
    except:
        pass
    try:
        if float(labs.get("fbg_mg_dl", 100)) >= 126:
            flags.append("High FBG — consider low-GI carbohydrate options")
    except:
        pass
    return neutropenic_mode, texture, flags, referrals

# ---------- RECIPE LOAD & FILTER ----------
def load_recipes(path="recipes.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"recipes.csv not found at {os.path.abspath(path)}")
    df = pd.read_csv(path, dtype=str)
    for col in ["kcal_per_serving","protein_g","prep_minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0
    for col in ["neutropenic_safe","vegetarian","low_k","low_phos","low_na"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)
        else:
            df[col] = False
    for c in ["texture_tag","notes","ingredients","name","recipe_id"]:
        if c not in df.columns:
            df[c] = ""
    df['name_lower'] = df['name'].astype(str).str.lower()
    df['ingredients_lower'] = df['ingredients'].astype(str).str.lower()
    return df

# Keyword buckets to map to meal slots
MEAL_KEYWORDS = {
    'breakfast': ['idli','dosa','porridge','oat','poha','upma','kheer','smoothie','toast','pancake','dalia','rava','cheela','paratha','roti + milk'],
    'lunch': ['khichdi','dal','rice','curry','subzi','pulao','biryani','stew','sambar','rajma','chole','dal tadka','paneer','mutton','chicken','fish'],
    'dinner': ['roti','dal','curry','stew','pulao','rice','mutton','chicken','fish','paneer'],
    'snack': ['sandwich','patties','smoothie','milk','curd','cutlet','sattu','biscuits','toast','banana','milkshake','pakora','puri'],
    'liquid': ['soup','smoothie','kheer','drink','porridge','congee','broth','milk','curd','sattu','juice']
}

def detect_meal_slot_from_row(row, forced_texture):
    name = (row.get('name_lower') or "")
    ing = (row.get('ingredients_lower') or "")
    txt = (row.get('texture_tag') or "").lower()
    # If forced to liquid/puree, prefer liquid slot
    if forced_texture in ('liquid','puree'):
        return 'Mid-morning'
    # check keyword lists
    for kw in MEAL_KEYWORDS['breakfast']:
        if kw in name or kw in ing:
            return 'Breakfast'
    for kw in MEAL_KEYWORDS['lunch']:
        if kw in name or kw in ing:
            return 'Lunch'
    for kw in MEAL_KEYWORDS['dinner']:
        if kw in name or kw in ing:
            return 'Dinner'
    for kw in MEAL_KEYWORDS['snack']:
        if kw in name or kw in ing:
            return 'Evening snack'
    for kw in MEAL_KEYWORDS['liquid']:
        if kw in name or kw in ing or kw in txt:
            return 'Mid-morning'
    # fallback by protein
    prot = float(row.get('protein_g', 0) or 0)
    if prot >= 12:
        return 'Lunch'
    return 'Mid-morning'

# ---------- MONTHLY PLANNER (30 days) ----------
def pick_candidate_for_slot(df_candidates, used_counts, max_use, prefer_high_protein=False):
    """Pick best candidate from df_candidates that is under max_use limit, scoring by keyword score and protein."""
    if df_candidates.empty:
        return None
    # create random tie-break to increase variety
    df = df_candidates.copy()
    # score: protein dominant + lower prep time preference
    df['score'] = df['protein_g'].fillna(0) * (1.0 if prefer_high_protein else 0.5) - df.get('prep_minutes', 0).astype(float).fillna(0)*0.01 + (pd.Series([random.random() for _ in range(len(df))]) * 0.01)
    df = df.sort_values(by='score', ascending=False)
    for _, row in df.iterrows():
        key = row.get('name_lower','')
        if used_counts.get(key, 0) < max_use:
            return row.copy()
    # if none found under max_use, relax and return top
    return df.iloc[0].copy()

def generate_month_plan(allowed_df, kcal_target, protein_target, days=30):
    """
    Generate a month-long plan (default 30 days).
    For each day, fill meal slots: Breakfast, Mid-morning, Lunch, Evening snack, Dinner, Bedtime(optional).
    Attempts to limit repetition: each recipe used at most max_use times (computed from available pool).
    """
    if allowed_df.empty:
        return None, "No allowed recipes available."

    df = allowed_df.reset_index(drop=True).copy()
    total_recipes = max(1, len(df))
    # set maximum allowed uses per recipe (aim variety): assume target uses per recipe = ceil(days*5 / total_recipes)
    approx_total_items_needed = days * 5  # 5 main slots/day
    max_use = max(2, math.ceil(approx_total_items_needed / total_recipes))
    # cap max_use to reasonable number
    max_use = min(max_use, max(4, math.ceil(days / 7)))  # avoid too large
    used_counts = {}
    month_plan = {}
    slot_order = ['Breakfast','Mid-morning','Lunch','Evening snack','Dinner']  # bedtime omitted by default; can be added if boosters needed

    for day in range(1, days+1):
        day_items = []
        # shuffle candidate pool slightly each day to increase variety
        df_shuf = df.sample(frac=1.0, random_state=random.randint(1,10000)).reset_index(drop=True)
        for slot in slot_order:
            # prefer high-protein for lunch/dinner
            prefer_prot = slot in ('Lunch','Dinner')
            # candidates: if slot is liquid due to forced texture, include liquid/puree; else use all and detect_meal_slot
            df_shuf['candidate_slot'] = df_shuf.apply(lambda r: detect_meal_slot_from_row(r, forced_texture="normal"), axis=1)
            # prefer candidates where candidate_slot matches slot
            candidates = df_shuf[df_shuf['candidate_slot'] == slot]
            if candidates.empty:
                candidates = df_shuf.copy()
            picked = pick_candidate_for_slot(candidates, used_counts, max_use, prefer_high_protein=prefer_prot)
            if picked is None:
                continue
            # record usage
            key = picked.get('name_lower','')
            used_counts[key] = used_counts.get(key, 0) + 1
            # set default servings
            picked['servings'] = 1.0
            # compute totals
            picked['kcal_total'] = round(float(picked.get('kcal_per_serving', 0) or 0) * float(picked.get('servings', 1)), 1)
            picked['protein_total'] = round(float(picked.get('protein_g', 0) or 0) * float(picked.get('servings', 1)), 1)
            picked['meal_slot'] = slot
            day_items.append(picked)
        # if daily protein is insufficient, append a simple booster (milk/protein) as bedtime or snack
        day_df = pd.DataFrame(day_items)
        total_prot = float(day_df['protein_total'].sum()) if not day_df.empty else 0.0
        if total_prot < 0.9 * protein_target:
            booster = {
                'recipe_id': 'booster_milk_protein',
                'name': 'Milk + Protein (suggested)',
                'kcal_per_serving': 200,
                'protein_g': 20,
                'servings': 1.0,
                'kcal_total': 200.0,
                'protein_total': 20.0,
                'meal_slot': 'Bedtime',
                'name_lower': 'milk + protein',
                'ingredients_lower': 'milk'
            }
            day_items.append(pd.Series(booster))
        # assemble day df
        if day_items:
            day_df = pd.DataFrame(day_items)
            # ensure columns exist
            for c in ['kcal_total','protein_total','servings','meal_slot','name','notes']:
                if c not in day_df.columns:
                    day_df[c] = ""
            # order by custom slot order
            slot_priority = {s:i for i,s in enumerate(slot_order+['Bedtime'])}
            day_df['slot_order'] = day_df['meal_slot'].apply(lambda s: slot_priority.get(s, 99))
            day_df = day_df.sort_values(by='slot_order').reset_index(drop=True)
            # recalc totals to be safe
            day_df['kcal_total'] = pd.to_numeric(day_df['kcal_total'], errors='coerce').fillna(0)
            day_df['protein_total'] = pd.to_numeric(day_df['protein_total'], errors='coerce').fillna(0)
            total_kcal = float(day_df['kcal_total'].sum())
            total_prot = float(day_df['protein_total'].sum())
        else:
            day_df = pd.DataFrame()
            total_kcal = 0.0
            total_prot = 0.0
        month_plan[f"Day {day}"] = {'df': day_df, 'kcal': total_kcal, 'protein': total_prot}
    return month_plan, None

# ---------- PDF: monthly report ----------
class PDFReport:
    def __init__(self, title="NutriSaarthi Monthly Plan", logo_path="logo.png", clinic_header=None):
        self.pdf = FPDF()
        self.title = title
        self.logo_path = logo_path
        self.clinic_header = clinic_header
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def header(self, title_text):
        if os.path.exists(self.logo_path):
            try:
                self.pdf.image(self.logo_path, x=80, y=8, w=50)
            except Exception as e:
                print("Logo error:", e)
        self.pdf.ln(25)
        self.pdf.set_font("Arial","B",16)
        self.pdf.cell(0,8, sanitize_text(title_text), ln=True, align="C")
        if self.clinic_header:
            self.pdf.set_font("Arial","",10)
            self.pdf.cell(0,6, sanitize_text(self.clinic_header), ln=True, align="C")
        self.pdf.set_font("Arial","I",10)
        self.pdf.cell(0,6, sanitize_text("Aahar aur Asha - Nutrition with Hope"), ln=True, align="C")
        self.pdf.ln(6)

    def add_patient_info(self, patient_info, flags, targets, referrals=None):
        self.pdf.set_font("Arial","",11)
        self.pdf.cell(0,6,sanitize_text(f"Patient: {patient_info.get('name','-')}"), ln=True)
        self.pdf.cell(0,6,sanitize_text(f"Age: {patient_info.get('age','-')}    Sex: {patient_info.get('sex','-')}    Weight: {patient_info.get('weight','-')} kg"), ln=True)
        self.pdf.cell(0,6,sanitize_text(f"Albumin: {patient_info.get('albumin','-')} g/dL    ANC: {patient_info.get('anc','-')} x10^9/L    FBG: {patient_info.get('fbg','-')} mg/dL"), ln=True)
        self.pdf.cell(0,6,sanitize_text(f"Mucositis: {patient_info.get('mucositis','-')}    Trismus: {patient_info.get('trismus_grade','-')}    Primary site: {patient_info.get('primary_site','-')}"), ln=True)
        self.pdf.cell(0,6,sanitize_text(f"eGFR: {patient_info.get('egfr','-')}    Dialysis: {patient_info.get('on_dialysis','-')}    Bilirubin: {patient_info.get('bilirubin','-')} mg/dL    INR: {patient_info.get('inr','-')}"), ln=True)
        self.pdf.cell(0,6,sanitize_text(f"Diet: {patient_info.get('diet_pref','-')}    Allergies: {patient_info.get('allergies_display','-')}"), ln=True)
        self.pdf.cell(0,6,sanitize_text(f"Energy target: {targets.get('kcal')} kcal/day    Protein target: {targets.get('protein')} g/day"), ln=True)
        self.pdf.ln(3)
        if flags:
            self.pdf.set_font("Arial","B",11)
            self.pdf.cell(0,6,sanitize_text("Safety flags:"), ln=True)
            self.pdf.set_font("Arial","",11)
            for f in flags:
                self.pdf.multi_cell(0,6,sanitize_text(f"- {f}"))
        else:
            self.pdf.cell(0,6,sanitize_text("Safety flags: None"), ln=True)
        self.pdf.ln(2)
        if referrals:
            self.pdf.set_font("Arial","B",11)
            self.pdf.cell(0,6,sanitize_text("Referrals / Actions:"), ln=True)
            self.pdf.set_font("Arial","",11)
            for r in referrals:
                self.pdf.multi_cell(0,6,sanitize_text(f"- {r}"))
            self.pdf.ln(3)

    def add_neutropenic_guidance(self, flags):
        if not any("neutropenic" in (str(f).lower()) for f in (flags or [])):
            return
        self.pdf.set_font("Arial","B",11)
        self.pdf.cell(0,6,sanitize_text("Neutropenic Diet — Key Recommendations:"), ln=True)
        self.pdf.set_font("Arial","",10)
        lines = [
            "• Prefer cooked, hot foods; avoid raw/undercooked meats and seafood.",
            "• Avoid raw salads, unpeeled fruits, and raw sprouts; use well-washed, cooked fruits/vegetables.",
            "• Use pasteurized dairy products only.",
            "• Reheat leftovers until steaming hot; discard food kept >24 hours at room temp.",
            "• Practice strict food hygiene; avoid street/unregulated foods during neutropenia."
        ]
        for l in lines:
            self.pdf.multi_cell(0,5,sanitize_text(l))
        self.pdf.ln(3)

    def add_month(self, month_plan):
        # month_plan is dict Day X => {df, kcal, protein}
        for day, dat in month_plan.items():
            df_day = dat.get('df', pd.DataFrame())
            if df_day.empty:
                continue
            # new page per day to keep clarity; for compactness could fit 2-3 days per page but simple approach uses one/day
            self.pdf.add_page()
            self.pdf.set_font("Arial","B",12)
            self.pdf.cell(0,7,sanitize_text(day), ln=True)
            # group by slot
            for slot in ['Breakfast','Mid-morning','Lunch','Evening snack','Dinner','Bedtime']:
                slot_df = df_day[df_day['meal_slot']==slot]
                if slot_df.empty:
                    continue
                self.pdf.set_font("Arial","B",11)
                self.pdf.cell(0,6,sanitize_text(slot), ln=True)
                # header
                col_w = [90, 25, 35, 35]
                headers = ["Recipe", "Serv", "Kcal", "Prot(g)"]
                for i,h in enumerate(headers):
                    self.pdf.set_font("Arial","B",10)
                    self.pdf.cell(col_w[i],6,sanitize_text(h), border=1, align="C")
                self.pdf.ln()
                for _,r in slot_df.iterrows():
                    name = sanitize_text(str(r.get('name',''))[:60])
                    serv = sanitize_text(str(r.get('servings',1)))
                    kcal = sanitize_text(str(r.get('kcal_total',0)))
                    prot = sanitize_text(str(r.get('protein_total',0)))
                    self.pdf.set_font("Arial","",10)
                    self.pdf.cell(col_w[0],6,name,border=1)
                    self.pdf.cell(col_w[1],6,serv,border=1,align="C")
                    self.pdf.cell(col_w[2],6,kcal,border=1,align="C")
                    self.pdf.cell(col_w[3],6,prot,border=1,align="C")
                    self.pdf.ln()
                self.pdf.ln(3)
            # daily totals
            try:
                total_kcal = round(float(df_day['kcal_total'].sum()),1)
                total_prot = round(float(df_day['protein_total'].sum()),1)
                self.pdf.set_font("Arial","B",10)
                self.pdf.cell(0,6,sanitize_text(f"Daily total: {total_kcal} kcal | {total_prot} g protein"), ln=True)
                self.pdf.ln(4)
            except:
                self.pdf.ln(2)

    def output_bytes(self):
        s = self.pdf.output(dest='S')
        if isinstance(s, bytes):
            return s
        try:
            return s.encode('latin-1')
        except:
            return s.encode('latin-1','replace')

def create_month_pdf_bytes(month_plan, patient_info, flags, targets, referrals=None, clinic_header=None):
    report = PDFReport(title="NutriSaarthi Monthly Diet Plan", clinic_header=clinic_header)
    report.pdf.add_page()
    report.header("NutriSaarthi — Monthly Diet Plan (30 days)")
    report.add_patient_info(patient_info, flags, targets, referrals=referrals)
    report.add_neutropenic_guidance(flags)
    report.add_month(month_plan)
    report.pdf.set_font("Arial","I",9)
    report.pdf.multi_cell(0,6,sanitize_text("Note: clinician-reviewable plan. Confirm before implementing."))
    return report.output_bytes()

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="NutriSaarthi Monthly Planner", layout="wide")
st.title("🌿 NutriSaarthi — 30-Day Practical Diet Plan")
st.caption("Aahar aur Asha — Nutrition with Hope")
st.markdown("---")

col1, col2 = st.columns([1,2])

with col1:
    st.header("🧍‍ Patient Data Entry")
    patient_name = st.text_input("Patient name (optional)")
    age = st.number_input("Age (years)", 1, 120, 60)
    sex = st.selectbox("Sex", ["Female", "Male", "Other"])
    weight = st.number_input("Weight (kg)", 20.0, 200.0, 60.0)
    albumin = st.number_input("Albumin (g/dL)", 1.0, 6.0, 3.2, step=0.1)
    anc = st.number_input("ANC (x10⁹/L)", 0.0, 20.0, 1.2, step=0.1)
    fbg = st.number_input("Fasting Blood Glucose (mg/dL)", 40, 400, 100)
    mucositis = st.slider("Mucositis Grade (0–4)", 0, 4, 1)
    diet_pref = st.selectbox("Diet Preference", ["Vegetarian", "Non-Vegetarian"])
    allergies_raw = st.text_input("Allergies (comma-separated)")
    allergies = [a.strip() for a in allergies_raw.split(",") if a.strip()]
    primary_site = st.selectbox("Primary site of malignancy", [
        "None/Not specified", "Head & Neck", "Oesophagus/Upper GI", "Lower GI",
        "Hepatobiliary", "Renal/Urinary", "Thoracic (lung)", "Breast", "Hematologic", "Other"
    ])
    trismus_grade = st.slider("Trismus grade (0–4)", 0, 4, 0)
    eGFR = st.number_input("eGFR (mL/min/1.73m²) — leave 0 if unknown", 0.0, 200.0, 90.0, step=1.0)
    on_dialysis = st.checkbox("On dialysis (hemodialysis/peritoneal)?", value=False)
    bilirubin = st.number_input("Total bilirubin (mg/dL)", 0.0, 50.0, 0.8, step=0.1)
    inr = st.number_input("INR", 0.5, 10.0, 1.0, step=0.1)
    hepatic_encephalopathy = st.selectbox("Hepatic encephalopathy history", ["No", "Yes", "Unknown"])
    st.markdown("---")
    st.write("Optional: Enable renal-safe recipe filtering (requires low_k & low_phos tags).")
    renal_filter = st.checkbox("Apply renal-safe recipe filtering", value=False)
    st.markdown("---")
    st.write("Disclaimer: For educational/personal use only. Confirm with clinician.")
    st.write("App version: monthly v0.1")

with col2:
    st.header("🔎 Generate 30-Day Plan")
    if st.button("Generate 30-Day Plan (1 month)"):
        labs = {"anc_x10e9_l": anc, "albumin_g_dl": albumin, "fbg_mg_dl": fbg}
        symptoms = {"mucositis_grade": mucositis}
        neutro, texture, flags, referrals = apply_rules(labs, symptoms,
            primary_site=primary_site, trismus_grade=trismus_grade,
            egfr=eGFR, on_dialysis=on_dialysis,
            bilirubin=bilirubin, inr=inr, hepatic_encephalopathy=hepatic_encephalopathy)
        kcal_t, prot_t = compute_targets(weight, albumin, egfr=eGFR, on_dialysis=on_dialysis, hepatic_encephalopathy=hepatic_encephalopathy)
        patient_info = {
            "name": patient_name, "age": age, "sex": sex, "weight": weight,
            "albumin": albumin, "anc": anc, "fbg": fbg, "mucositis": mucositis,
            "diet_pref": diet_pref, "allergies_display": ", ".join(allergies) if allergies else "-",
            "primary_site": primary_site, "trismus_grade": trismus_grade,
            "egfr": eGFR, "on_dialysis": "Yes" if on_dialysis else "No",
            "bilirubin": bilirubin, "inr": inr, "hepatic_encephalopathy": hepatic_encephalopathy
        }
        targets = {"kcal": kcal_t, "protein": prot_t}
        st.subheader("📋 Safety & Targets")
        if flags:
            for f in flags:
                st.warning(f)
        else:
            st.info("No safety flags detected.")
        if referrals:
            for r in referrals:
                st.info(f"Referral / action suggested: {r}")
        st.info(f"Energy target: {kcal_t} kcal/day | Protein target: {prot_t} g/day")
        try:
            df = load_recipes("recipes.csv")
        except Exception as e:
            st.error(f"Could not load recipes.csv: {e}")
            st.stop()
        allowed = df.copy()
        # filters
        if neutro:
            allowed = allowed[allowed['neutropenic_safe'] == True]
        if texture == 'soft':
            allowed = allowed[allowed['texture_tag'].isin(['soft','liquid','puree'])]
        elif texture == 'liquid':
            allowed = allowed[allowed['texture_tag'].isin(['liquid','puree'])]
        if diet_pref.lower().startswith('veget'):
            allowed = allowed[allowed['vegetarian'] == True]
        if allergies:
            for a in allergies:
                pat = r"(?i)\b" + re.escape(a) + r"\b"
                allowed = allowed[~allowed['ingredients'].astype(str).str.contains(pat, na=False)]
        if renal_filter and eGFR and eGFR>0 and eGFR<30 and not on_dialysis:
            if 'low_k' in allowed.columns and 'low_phos' in allowed.columns:
                tmp = allowed[(allowed['low_k']==True) & (allowed['low_phos']==True)]
                if not tmp.empty:
                    allowed = tmp
        if allowed.empty:
            st.error("No safe recipes found with current filters. Adjust settings.")
        else:
            month_plan, err = generate_month_plan(allowed, kcal_t, prot_t, days=30)
            if err:
                st.error(err)
            else:
                st.success("✅ 30-Day Plan Generated")
                # show summary (collapsed per week for compactness)
                for week in range(0, 30, 7):
                    with st.expander(f"Days {week+1}–{min(week+7,30)}"):
                        for d in range(week+1, min(week+7,30)+1):
                            key = f"Day {d}"
                            dat = month_plan.get(key, {})
                            df_day = dat.get('df', pd.DataFrame())
                            kcal = dat.get('kcal', 0.0)
                            prot = dat.get('protein', 0.0)
                            st.write(f"**{key} — {kcal:.0f} kcal | {prot:.1f} g protein**")
                            if not df_day.empty:
                                for slot in ['Breakfast','Mid-morning','Lunch','Evening snack','Dinner','Bedtime']:
                                    slot_df = df_day[df_day['meal_slot']==slot]
                                    if not slot_df.empty:
                                        st.markdown(f"*{slot}*")
                                        display_cols = [c for c in ['name','servings','kcal_total','protein_total','notes'] if c in slot_df.columns]
                                        show = slot_df[display_cols].rename(columns={'name':'Recipe','servings':'Servings','kcal_total':'Kcal','protein_total':'Protein(g)','notes':'Notes'})
                                        st.table(show.reset_index(drop=True))
                # downloads
                pdf_bytes = create_month_pdf_bytes(month_plan, patient_info, flags, targets, referrals=referrals)
                pdf_name = f"NutriSaarthi_MonthPlan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button("📥 Download 30-Day PDF", data=pdf_bytes, file_name=pdf_name, mime="application/pdf")
                # CSV export generation: compile month into single CSV with day & slot
                rows = []
                for day,dat in month_plan.items():
                    df_day = dat.get('df', pd.DataFrame())
                    if df_day is None or df_day.empty:
                        continue
                    for _, r in df_day.iterrows():
                        rows.append({
                            'day': day,
                            'meal_slot': r.get('meal_slot',''),
                            'recipe': r.get('name',''),
                            'servings': r.get('servings',1),
                            'kcal': r.get('kcal_total',0),
                            'protein_g': r.get('protein_total',0),
                            'notes': r.get('notes','')
                        })
                csv_df = pd.DataFrame(rows)
                csv_bytes = csv_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download 30-Day CSV", data=csv_bytes, file_name=f"NutriSaarthi_MonthPlan_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
st.markdown("---")
st.write("Created by Dr Atul Gupta")
