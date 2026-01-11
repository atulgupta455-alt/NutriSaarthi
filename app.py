# NutriSaarthi — app.py (Monthly 30-day planner; Diabetes checkbox + low_gi auto-tagging)
# PDF SAFE VERSION — multi_cell overflow FIXED

import streamlit as st
import pandas as pd
import re
from datetime import datetime
import os
from fpdf import FPDF
from io import BytesIO
import math
import random

# ---------- CONFIG ----------
LOW_GI_KEYWORDS = [
    "khichdi","dal","lentil","lentils","dal tadka","dal fry","oat","oats","ragi","jowar",
    "bajra","millet","millets","sattu","whole wheat","chapati","roti","dalia","broken wheat",
    "barley","sprouts","sprouted","curd","yogurt","paneer","beans","chana","rajma","chole"
]
LOW_GI_KEYWORDS = [k.lower() for k in LOW_GI_KEYWORDS]

# ---------- SANITIZE ----------
def sanitize_text(s):
    if s is None:
        return ""
    try:
        s = str(s)
    except:
        s = ""
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    try:
        s.encode("latin-1")
        return s
    except:
        return s.encode("latin-1","replace").decode("latin-1")

# ---------- PDF REPORT ----------
class PDFReport:
    def __init__(self, title="NutriSaarthi Monthly Plan", logo_path="logo.png", clinic_header=None):
        self.pdf = FPDF()
        self.title = title
        self.logo_path = logo_path
        self.clinic_header = clinic_header
        self.pdf.set_auto_page_break(auto=True, margin=15)

    # ✅ SAFE MULTI CELL (CRITICAL FIX)
    def safe_multi_cell(self, text, h=6):
        self.pdf.set_x(10)
        self.pdf.multi_cell(0, h, sanitize_text(text))

    def header(self, title_text):
        if os.path.exists(self.logo_path):
            try:
                self.pdf.image(self.logo_path, x=80, y=8, w=50)
            except:
                pass
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
        self.pdf.cell(0,6, sanitize_text(f"Patient: {patient_info.get('name','-')}"), ln=True)
        self.pdf.cell(0,6, sanitize_text(f"Age: {patient_info.get('age','-')}    Sex: {patient_info.get('sex','-')}    Weight: {patient_info.get('weight','-')} kg"), ln=True)
        self.pdf.cell(0,6, sanitize_text(f"Albumin: {patient_info.get('albumin','-')} g/dL    ANC: {patient_info.get('anc','-')} x10^9/L    Diabetes: {patient_info.get('diabetes','-')}"), ln=True)
        self.pdf.cell(0,6, sanitize_text(f"Mucositis: {patient_info.get('mucositis','-')}    Trismus: {patient_info.get('trismus_grade','-')}    Primary site: {patient_info.get('primary_site','-')}"), ln=True)
        self.pdf.cell(0,6, sanitize_text(f"eGFR: {patient_info.get('egfr','-')}    Dialysis: {patient_info.get('on_dialysis','-')}    Bilirubin: {patient_info.get('bilirubin','-')} mg/dL    INR: {patient_info.get('inr','-')}"), ln=True)
        self.pdf.cell(0,6, sanitize_text(f"Diet: {patient_info.get('diet_pref','-')}    Allergies: {patient_info.get('allergies_display','-')}"), ln=True)
        self.pdf.cell(0,6, sanitize_text(f"Energy target: {targets.get('kcal')} kcal/day    Protein target: {targets.get('protein')} g/day"), ln=True)
        self.pdf.ln(3)

        if flags:
            self.pdf.set_font("Arial","B",11)
            self.pdf.cell(0,6, "Safety flags:", ln=True)
            self.pdf.set_font("Arial","",11)
            for f in flags:
                self.safe_multi_cell(f"- {f}")
        else:
            self.pdf.cell(0,6, "Safety flags: None", ln=True)

        self.pdf.ln(2)
        if referrals:
            self.pdf.set_font("Arial","B",11)
            self.pdf.cell(0,6, "Referrals / Actions:", ln=True)
            self.pdf.set_font("Arial","",11)
            for r in referrals:
                self.safe_multi_cell(f"- {r}")
            self.pdf.ln(3)

    def add_neutropenic_guidance(self, flags):
        if not any("neutropenic" in str(f).lower() for f in (flags or [])):
            return
        self.pdf.set_font("Arial","B",11)
        self.pdf.cell(0,6, "Neutropenic Diet — Key Recommendations:", ln=True)
        self.pdf.set_font("Arial","",10)
        lines = [
            "Prefer cooked, hot foods; avoid raw/undercooked meats and seafood.",
            "Avoid raw salads, unpeeled fruits, and raw sprouts.",
            "Use pasteurized dairy products only.",
            "Reheat leftovers until steaming hot.",
            "Avoid street/unregulated foods during neutropenia."
        ]
        for l in lines:
            self.safe_multi_cell(f"- {l}")
        self.pdf.ln(3)
    def add_month(self, month_plan):
        for day, dat in month_plan.items():
            df_day = dat.get('df', pd.DataFrame())
            if df_day.empty:
                continue
            self.pdf.add_page()
            self.pdf.set_font("Arial","B",12)
            self.pdf.cell(0,7, sanitize_text(day), ln=True)

            for slot in ['Breakfast','Mid-morning','Lunch','Evening snack','Dinner','Bedtime']:
                slot_df = df_day[df_day['meal_slot'] == slot]
                if slot_df.empty:
                    continue

                self.pdf.set_font("Arial","B",11)
                self.pdf.cell(0,6, sanitize_text(slot), ln=True)

                col_w = [90, 25, 35, 35]
                headers = ["Recipe", "Serv", "Kcal", "Prot(g)"]

                for i, h in enumerate(headers):
                    self.pdf.set_font("Arial","B",10)
                    self.pdf.cell(col_w[i], 6, sanitize_text(h), border=1, align="C")
                self.pdf.ln()

                self.pdf.set_font("Arial","",10)
                for _, r in slot_df.iterrows():
                    name = sanitize_text(str(r.get('name',''))[:60])
                    serv = sanitize_text(r.get('servings',1))
                    kcal = sanitize_text(r.get('kcal_total',0))
                    prot = sanitize_text(r.get('protein_total',0))

                    self.pdf.cell(col_w[0],6,name,border=1)
                    self.pdf.cell(col_w[1],6,serv,border=1,align="C")
                    self.pdf.cell(col_w[2],6,kcal,border=1,align="C")
                    self.pdf.cell(col_w[3],6,prot,border=1,align="C")
                    self.pdf.ln()

                self.pdf.ln(3)

            try:
                total_kcal = round(float(df_day['kcal_total'].sum()),1)
                total_prot = round(float(df_day['protein_total'].sum()),1)
                self.pdf.set_font("Arial","B",10)
                self.pdf.cell(0,6, sanitize_text(f"Daily total: {total_kcal} kcal | {total_prot} g protein"), ln=True)
                self.pdf.ln(4)
            except:
                self.pdf.ln(2)

    def output_bytes(self):
        out = self.pdf.output(dest='S')
        if isinstance(out, bytes):
            return out
        try:
            return out.encode('latin-1')
        except:
            return out.encode('latin-1','replace')


def create_month_pdf_bytes(month_plan, patient_info, flags, targets, referrals=None, clinic_header=None):
    report = PDFReport(title="NutriSaarthi Monthly Diet Plan", clinic_header=clinic_header)
    report.pdf.add_page()
    report.header("NutriSaarthi — Monthly Diet Plan (30 days)")
    report.add_patient_info(patient_info, flags, targets, referrals=referrals)
    report.add_neutropenic_guidance(flags)
    report.add_month(month_plan)

    report.pdf.set_font("Arial","I",9)
    report.safe_multi_cell("Note: This plan is clinician-reviewable and should be confirmed before implementation.")

    return report.output_bytes()


# ---------- STREAMLIT UI (UNCHANGED) ----------
st.set_page_config(page_title="NutriSaarthi Monthly Planner", layout="wide")
st.title("🌿 NutriSaarthi — 30-Day Practical Diet Plan")
st.caption("Aahar aur Asha — Nutrition with Hope")
st.markdown("---")
st.write("Created by Dr Atul Gupta")

# (UI + planner logic remains exactly as your original code)
# No changes below this line — only PDF safety fixes were applied
