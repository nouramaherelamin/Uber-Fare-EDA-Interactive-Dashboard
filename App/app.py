from pathlib import Path
import io
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "uber_fare_eda_clean.csv"
ASSET_DIR = BASE_DIR / "assets"
TAXI_IMAGE = ASSET_DIR / "taxi.png"

PAGE_TITLE = "Uber Fare — Interactive EDA"
PLOT_BG = "#11151B"
PAPER_BG = "#11151B"
FONT_COLOR = "#F8FAFC"
MUTED = "#9CA3AF"
GRID = "rgba(255,255,255,.08)"
ACCENT = "#FFC107"
ACCENT_SOFT = "#FFD54A"
SECONDARY = "#38BDF8"
SUCCESS = "#2DD4BF"
BORDER = "#2D3540"

PAGES = [
    "Overview", "Fare Analysis", "Demand Analysis", "Passenger Analysis",
    "Car & Traffic", "Distance Analysis", "Weather Analysis", "Airport Analysis",
    "Correlation Explorer", "Data Explorer", "Data Quality",
]
ANALYTICAL_COLUMNS = [
    "fare_amount", "pickup_datetime", "Car Condition", "Weather", "Traffic Condition",
    "passenger_count", "hour", "day", "month", "weekday", "year",
    "jfk_dist", "ewr_dist", "lga_dist", "sol_dist", "nyc_dist", "distance", "bearing",
]
NUMERIC_DEFAULTS = [
    "fare_amount", "distance", "passenger_count", "hour", "month", "year",
    "jfk_dist", "ewr_dist", "lga_dist", "sol_dist", "nyc_dist", "bearing",
]

st.set_page_config(page_title=PAGE_TITLE, page_icon="🚕", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
:root{--bg:#07090C;--panel:#11151B;--panel2:#171C23;--line:#2D3540;--text:#F8FAFC;--muted:#9CA3AF;--yellow:#FFC107;--yellow2:#FFD54A;}
.stApp{background:var(--bg)!important;color:var(--text)!important}
.block-container{max-width:1480px!important;padding:1.5rem 2rem 3rem!important}
.stApp .stMarkdown,.stApp .stMarkdown p,.stApp label,.stApp [data-testid="stWidgetLabel"],.stApp [data-testid="stWidgetLabel"] p{color:var(--text)!important}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] p{color:var(--muted)!important}
section[data-testid="stSidebar"]{background:#0B0E12!important;border-right:1px solid var(--line)!important}
section[data-testid="stSidebar"] .block-container{padding:1rem .85rem 2rem!important}
section[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:9px!important;padding:.48rem .6rem!important;color:#D1D5DB!important;transition:.2s ease}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#171C23!important;transform:translateX(2px)}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:var(--yellow)!important;color:#111!important;font-weight:800!important}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span{color:#111!important}
section[data-testid="stSidebar"] [data-testid="stExpander"]{background:var(--panel)!important;border:1px solid var(--line)!important;border-radius:11px!important;margin:.35rem 0!important}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary{color:#F8FAFC!important;font-weight:700!important}
[data-baseweb="select"]>div{background:var(--panel2)!important;border-color:#3A414D!important;color:#F8FAFC!important}
[data-baseweb="select"] input{color:#F8FAFC!important;caret-color:var(--yellow)!important}
[data-baseweb="tag"]{background:#252A33!important;color:#F8FAFC!important}
div[role="listbox"],[data-baseweb="popover"],div[role="option"]{background:#11151B!important;color:#F8FAFC!important}
div[role="option"]:hover{background:#211B05!important}
.stButton button{border-radius:9px!important;border:1px solid #3A414D!important;background:#11151B!important;color:#F8FAFC!important}
.stButton button:hover{border-color:var(--yellow)!important;color:var(--yellow2)!important}
button[kind="primary"]{background:var(--yellow)!important;color:#111!important;border-color:var(--yellow)!important;font-weight:800!important}
button[kind="primary"]:hover{background:#FFD54A!important;border-color:#FFD54A!important}

/* NATIVE HERO — reliable Streamlit rendering */
.st-key-hero_container{
  position:relative!important; overflow:hidden!important; min-height:330px!important;
  margin:0 0 2.4rem!important; padding:2.25rem 2.6rem!important;
  border:1px solid rgba(255,193,7,.34)!important; border-radius:26px!important;
  background:radial-gradient(ellipse at 82% 48%,rgba(255,193,7,.30) 0%,rgba(255,193,7,.11) 23%,transparent 48%),linear-gradient(120deg,#06080B 0%,#0D1116 54%,#1B1B19 100%)!important;
  box-shadow:0 24px 60px rgba(0,0,0,.48),inset 0 1px 0 rgba(255,255,255,.035)!important;
}
.st-key-hero_container:before{content:""!important;position:absolute!important;width:500px!important;height:500px!important;right:-70px!important;top:-220px!important;border-radius:50%!important;background:radial-gradient(circle,rgba(255,193,7,.19),transparent 68%)!important;animation:heroGlow 5s ease-in-out infinite!important;pointer-events:none!important}
.st-key-hero_container>div{position:relative!important;z-index:2!important}
.st-key-hero_container [data-testid="stHorizontalBlock"]{align-items:center!important;gap:2.5rem!important;margin:0!important}
.st-key-hero_container [data-testid="column"]{min-height:300px!important;display:flex!important;flex-direction:column!important;justify-content:center!important}
.hero-accent{width:76px;height:5px;border-radius:99px;background:#FFC107;box-shadow:0 0 24px rgba(255,193,7,.62);animation:accentPulse 2.6s ease-in-out infinite;margin-bottom:22px}
.hero-title{margin:0!important;color:#FFFFFF!important;font-size:clamp(2.3rem,4vw,3.55rem)!important;line-height:1.02!important;font-weight:900!important;letter-spacing:-.055em!important;animation:heroCopyIn .75s cubic-bezier(.2,.8,.2,1) both!important}
.hero-description{margin:1rem 0 0!important;max-width:690px!important;color:#D4D9E0!important;font-size:1rem!important;line-height:1.72!important;animation:heroCopyIn .9s .05s cubic-bezier(.2,.8,.2,1) both!important}
.hero-tag{display:inline-flex!important;width:max-content!important;margin-top:1.1rem!important;padding:.42rem .72rem!important;border:1px solid rgba(255,193,7,.30)!important;border-radius:999px!important;background:rgba(255,193,7,.08)!important;color:#FFD54A!important;font-size:.72rem!important;font-weight:800!important;animation:heroCopyIn 1s .1s cubic-bezier(.2,.8,.2,1) both!important}
.st-key-hero_container img{display:block!important;width:100%!important;max-width:620px!important;max-height:310px!important;object-fit:contain!important;filter:drop-shadow(0 28px 35px rgba(0,0,0,.72)) drop-shadow(0 0 38px rgba(255,193,7,.22))!important;animation:taxiEnter .95s cubic-bezier(.16,1,.3,1) both,taxiFloat 4.2s 1s ease-in-out infinite!important;transition:transform .45s ease,filter .45s ease!important}
.st-key-hero_container img:hover{transform:scale(1.06) translateY(-7px)!important;filter:drop-shadow(0 32px 42px rgba(0,0,0,.76)) drop-shadow(0 0 42px rgba(255,193,7,.27))!important}
.st-key-hero_container [data-testid="stImage"]{display:flex!important;align-items:center!important;justify-content:center!important;position:relative!important}
.st-key-hero_container [data-testid="stImage"]:after{content:"";position:absolute;left:12%;right:12%;bottom:8px;height:3px;border-radius:99px;background:linear-gradient(90deg,transparent,#FFC107 28%,#FFD54A 50%,#FFC107 72%,transparent);box-shadow:0 0 18px rgba(255,193,7,.38);animation:roadPulse 2.8s ease-in-out infinite}
/* More breathing room between card rows */
div[data-testid="stHorizontalBlock"]{column-gap:1.35rem!important;row-gap:1.35rem!important;margin-bottom:1.05rem!important}
.kpi,.quality-card{margin-bottom:.2rem!important}
@media(max-width:950px){.st-key-hero_container{padding:1.7rem 1.35rem!important}.st-key-hero_container [data-testid="stHorizontalBlock"]{gap:1rem!important}.st-key-hero_container [data-testid="column"]{min-height:190px!important}.st-key-hero_container img{max-height:320px!important}}

/* PREMIUM HERO */
.hero-shell{position:relative!important;overflow:hidden!important;min-height:330px!important;margin:0 0 2rem!important;padding:0!important;border:1px solid rgba(255,193,7,.34)!important;border-radius:26px!important;background:radial-gradient(ellipse at 78% 55%,rgba(255,193,7,.30) 0%,rgba(255,193,7,.11) 24%,transparent 48%),radial-gradient(ellipse at 100% 0%,rgba(255,193,7,.12),transparent 36%),linear-gradient(120deg,#06080B 0%,#0D1116 52%,#181A1E 100%)!important;box-shadow:0 24px 60px rgba(0,0,0,.48),inset 0 1px 0 rgba(255,255,255,.035)!important}
.hero-shell:before{content:""!important;position:absolute!important;width:520px!important;height:520px!important;right:20px!important;top:-220px!important;border-radius:50%!important;background:radial-gradient(circle,rgba(255,193,7,.20),transparent 68%)!important;animation:heroGlow 5s ease-in-out infinite!important;pointer-events:none!important}
.hero-shell:after{content:""!important;position:absolute!important;left:-30%!important;top:0!important;width:20%!important;height:100%!important;transform:skewX(-18deg)!important;background:linear-gradient(90deg,transparent,rgba(255,255,255,.055),transparent)!important;animation:heroSweep 6.5s ease-in-out infinite!important;pointer-events:none!important}
.hero-inner{position:relative!important;z-index:2!important;display:grid!important;grid-template-columns:minmax(0,1fr) minmax(460px,590px)!important;align-items:center!important;gap:1rem!important;min-height:330px!important;padding:2.4rem 2.7rem!important}
.hero-copy{animation:heroCopyIn .75s cubic-bezier(.2,.8,.2,1) both!important;position:relative!important;z-index:3!important}
.hero-copy:before{content:""!important;display:block!important;width:72px!important;height:5px!important;margin-bottom:22px!important;border-radius:99px!important;background:#FFC107!important;box-shadow:0 0 24px rgba(255,193,7,.62)!important;animation:accentPulse 2.6s ease-in-out infinite!important}
.hero h1{margin:0!important;color:#FFFFFF!important;font-size:clamp(2.35rem,4.2vw,3.65rem)!important;line-height:1.02!important;font-weight:900!important;letter-spacing:-.055em!important;text-shadow:0 2px 18px rgba(0,0,0,.25)!important}
.hero p{margin:1rem 0 0!important;max-width:700px!important;color:#D4D9E0!important;font-size:1rem!important;line-height:1.72!important}
.hero-art{position:relative!important;display:flex!important;justify-content:center!important;align-items:center!important;min-height:300px!important;animation:taxiFloat 4.2s ease-in-out infinite!important}
.hero-taxi{display:block!important;width:100%!important;max-width:560px!important;height:auto!important;filter:drop-shadow(0 28px 35px rgba(0,0,0,.72)) drop-shadow(0 0 38px rgba(255,193,7,.22))!important;transform-origin:center!important;animation:taxiEnter .95s cubic-bezier(.16,1,.3,1) both!important;transition:transform .45s cubic-bezier(.2,.8,.2,1),filter .45s ease!important}
.hero-art:hover .hero-taxi{transform:scale(1.065) translateY(-7px)!important;filter:drop-shadow(0 32px 42px rgba(0,0,0,.76)) drop-shadow(0 0 42px rgba(255,193,7,.27))!important}
.hero-art:after{content:""!important;position:absolute!important;width:78%!important;height:3px!important;bottom:28px!important;border-radius:99px!important;background:linear-gradient(90deg,transparent,#FFC107 28%,#FFD54A 50%,#FFC107 72%,transparent)!important;box-shadow:0 0 18px rgba(255,193,7,.38)!important;animation:roadPulse 2.8s ease-in-out infinite!important}
.hero-tag{display:inline-flex!important;margin-top:1.1rem!important;padding:.42rem .72rem!important;border:1px solid rgba(255,193,7,.30)!important;border-radius:999px!important;background:rgba(255,193,7,.08)!important;color:#FFD54A!important;font-size:.72rem!important;font-weight:800!important;letter-spacing:.03em!important}
.section-title{margin:1.25rem 0 .35rem!important;color:#fff!important;font-size:1.35rem!important;font-weight:800!important;letter-spacing:-.025em!important;animation:rise .45s ease both}.section-subtitle{margin:0 0 1rem!important;color:#9CA3AF!important;font-size:.88rem!important;line-height:1.55!important}
.kpi{min-height:112px;padding:1.05rem 1.15rem;background:linear-gradient(145deg,#12171E,#0F1318)!important;border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 24px rgba(0,0,0,.2);transition:.22s ease}.kpi:hover{transform:translateY(-3px);border-color:rgba(255,193,7,.45);box-shadow:0 13px 30px rgba(0,0,0,.3)}
.kpi-label{color:#AAB2BD!important;font-size:.68rem!important;font-weight:800;letter-spacing:.075em;text-transform:uppercase}.kpi-value{margin-top:.45rem;color:#fff!important;font-size:1.45rem;font-weight:850;line-height:1.15}
.insight{margin:.4rem 0 1rem;padding:1rem 1.1rem;background:linear-gradient(145deg,#15170F,#10130F)!important;border:1px solid #4B401B;border-left:3px solid var(--yellow)!important;border-radius:13px;color:#F3F4F6!important;line-height:1.5}.insight strong{color:var(--yellow)!important}
[data-testid="stMetric"]{min-height:110px!important;padding:1rem 1.1rem!important;background:linear-gradient(145deg,#12171E,#0F1318)!important;border:1px solid var(--line)!important;border-radius:14px!important;box-shadow:0 8px 24px rgba(0,0,0,.2)!important;transition:.22s ease}[data-testid="stMetric"]:hover{transform:translateY(-3px);border-color:rgba(255,193,7,.4)!important}[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p{color:#AAB2BD!important}[data-testid="stMetricValue"],[data-testid="stMetricValue"] div{color:#fff!important;font-weight:850!important}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{background:#11151B!important;border:1px solid var(--line)!important;border-radius:14px!important;box-shadow:0 8px 24px rgba(0,0,0,.2)!important;animation:rise .4s ease both}
[data-testid="stDataFrame"]{overflow:hidden!important}
.stTextInput input,.stNumberInput input{background:var(--panel2)!important;color:#fff!important;border-color:#3A414D!important}.stTextInput input::placeholder{color:#8D96A3!important}
hr{border-color:#252C35!important;margin:1.25rem 0!important}footer{visibility:hidden}
.empty-state{padding:2rem;text-align:center;border:1px dashed #4B5563;border-radius:14px;background:#10141A;color:#D1D5DB}.empty-state h3{margin:.2rem 0 .5rem;color:#fff}.empty-state p{margin:0;color:#9CA3AF}
.quality-card{min-height:118px;padding:1rem 1.05rem;border:1px solid var(--line);border-radius:13px;background:linear-gradient(145deg,#12171E,#0F1318)}.quality-card b{display:block;color:var(--yellow);font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.4rem}.quality-card span{color:#F3F4F6;font-size:.88rem;line-height:1.45}
@keyframes glow{0%,100%{transform:scale(1);opacity:.7}50%{transform:scale(1.1);opacity:1}}@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}@keyframes pulse{0%,100%{width:64px}50%{width:84px}}

/* PREMIUM CARDS */
.kpi,.quality-card{box-sizing:border-box!important;position:relative!important;overflow:hidden!important;min-height:132px!important;padding:1.25rem 1.3rem!important;border:1px solid #303844!important;border-radius:17px!important;background:linear-gradient(145deg,#151A21 0%,#10141A 72%,#0D1116 100%)!important;box-shadow:0 10px 28px rgba(0,0,0,.25),inset 0 1px 0 rgba(255,255,255,.025)!important;transition:transform .28s cubic-bezier(.2,.8,.2,1),border-color .28s ease,box-shadow .28s ease,background .28s ease!important;animation:cardIn .55s cubic-bezier(.2,.8,.2,1) both!important}
.kpi:before,.quality-card:before{content:""!important;position:absolute!important;left:0!important;top:0!important;width:100%!important;height:3px!important;background:linear-gradient(90deg,#FFC107,#FFD54A,transparent)!important;transform:scaleX(.35)!important;transform-origin:left!important;transition:transform .3s ease!important}
.kpi:after,.quality-card:after{content:""!important;position:absolute!important;width:110px!important;height:110px!important;right:-55px!important;top:-55px!important;border-radius:50%!important;background:radial-gradient(circle,rgba(255,193,7,.11),transparent 68%)!important;transition:transform .35s ease!important}
.kpi:hover,.quality-card:hover{transform:translateY(-7px)!important;border-color:rgba(255,193,7,.48)!important;background:linear-gradient(145deg,#191F27 0%,#10151B 100%)!important;box-shadow:0 18px 38px rgba(0,0,0,.38),0 0 0 1px rgba(255,193,7,.055),0 0 28px rgba(255,193,7,.055)!important}
.kpi:hover:before,.quality-card:hover:before{transform:scaleX(1)!important}
.kpi:hover:after,.quality-card:hover:after{transform:scale(1.5)!important}
.kpi-label,.quality-card b{position:relative!important;z-index:2!important;margin:0 0 .55rem!important;color:#9FA9B7!important;font-size:.66rem!important;line-height:1.2!important;font-weight:850!important;letter-spacing:.095em!important;text-transform:uppercase!important}
.kpi-value{position:relative!important;z-index:2!important;margin:0!important;color:#FFFFFF!important;font-size:1.55rem!important;line-height:1.12!important;font-weight:900!important;letter-spacing:-.035em!important}
.quality-card{display:flex!important;flex-direction:column!important;justify-content:center!important;min-height:142px!important}
.quality-card b{color:#FFC107!important;display:block!important}
.quality-card span{position:relative!important;z-index:2!important;display:block!important;color:#E9ECF1!important;font-size:.87rem!important;line-height:1.5!important}
.quality-icon{position:relative!important;z-index:2!important;width:36px!important;height:36px!important;display:flex!important;align-items:center!important;justify-content:center!important;margin-bottom:.8rem!important;border:1px solid rgba(255,193,7,.28)!important;border-radius:10px!important;background:rgba(255,193,7,.08)!important;color:#FFC107!important;font-size:1.05rem!important;font-weight:900!important;transition:transform .3s ease,background .3s ease!important}
.quality-card:hover .quality-icon{transform:rotate(-5deg) scale(1.08)!important;background:rgba(255,193,7,.14)!important}
.quality-title{position:relative!important;z-index:2!important;margin-bottom:.35rem!important;color:#FFC107!important;font-size:.66rem!important;line-height:1.2!important;font-weight:850!important;letter-spacing:.095em!important;text-transform:uppercase!important}
.quality-body{position:relative!important;z-index:2!important;color:#E9ECF1!important;font-size:.87rem!important;line-height:1.5!important}
div[data-testid="stHorizontalBlock"]{align-items:stretch!important;gap:1rem!important;margin-bottom:.15rem!important}
div[data-testid="column"]{min-width:0!important}
[data-testid="stMetric"]{min-height:118px!important;border:1px solid #303844!important;border-radius:17px!important;background:linear-gradient(145deg,#151A21,#10141A)!important;box-shadow:0 10px 28px rgba(0,0,0,.25)!important;transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease!important}
[data-testid="stMetric"]:hover{transform:translateY(-5px)!important;border-color:rgba(255,193,7,.45)!important;box-shadow:0 17px 34px rgba(0,0,0,.35)!important}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{border:1px solid #2D3540!important;border-radius:17px!important;box-shadow:0 10px 28px rgba(0,0,0,.25)!important;transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease!important}
[data-testid="stPlotlyChart"]:hover{border-color:rgba(255,193,7,.28)!important;box-shadow:0 15px 34px rgba(0,0,0,.33)!important}
.insight{position:relative!important;overflow:hidden!important;margin:.7rem 0 1.15rem!important;padding:1.05rem 1.2rem 1.05rem 1.25rem!important;border:1px solid #3B351F!important;border-left:4px solid #FFC107!important;border-radius:15px!important;background:linear-gradient(145deg,#17170F,#10130F)!important;color:#E7EAF0!important;box-shadow:0 8px 22px rgba(0,0,0,.18)!important}
.insight:after{content:""!important;position:absolute!important;right:-50px!important;top:-70px!important;width:160px!important;height:160px!important;border-radius:50%!important;background:radial-gradient(circle,rgba(255,193,7,.08),transparent 68%)!important}
.insight strong{color:#FFC107!important}
@keyframes heroGlow{0%,100%{transform:scale(1);opacity:.65}50%{transform:scale(1.12);opacity:1}}
@keyframes heroSweep{0%{left:-30%;opacity:0}12%{opacity:1}48%{left:120%;opacity:.4}100%{left:120%;opacity:0}}
@keyframes heroCopyIn{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:translateX(0)}}
@keyframes taxiEnter{from{opacity:0;transform:translateX(55px) scale(.84)}to{opacity:1;transform:translateX(0) scale(1)}}
@keyframes taxiFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes accentPulse{0%,100%{width:72px;box-shadow:0 0 16px rgba(255,193,7,.36)}50%{width:100px;box-shadow:0 0 28px rgba(255,193,7,.66)}}
@keyframes roadPulse{0%,100%{opacity:.55;transform:scaleX(.88)}50%{opacity:1;transform:scaleX(1)}}
@keyframes cardIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.kpi:nth-child(2),.quality-card:nth-child(2){animation-delay:.05s!important}
.kpi:nth-child(3),.quality-card:nth-child(3){animation-delay:.10s!important}
.kpi:nth-child(4),.quality-card:nth-child(4){animation-delay:.15s!important}
.kpi:nth-child(5),.quality-card:nth-child(5){animation-delay:.20s!important}
.kpi:nth-child(6),.quality-card:nth-child(6){animation-delay:.25s!important}
@media(max-width:950px){.block-container{padding:1rem .9rem 2.5rem!important}.hero-shell{min-height:0!important}.hero-inner{grid-template-columns:1fr!important;min-height:0!important;padding:1.8rem 1.4rem 1.5rem!important}.hero-art{min-height:210px!important;margin-top:.2rem!important}.hero-taxi{max-width:390px!important}.kpi,.quality-card{min-height:112px!important}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}

/* =========================================================
   FINAL POLISH — HERO + CARD SPACING
   ========================================================= */

/* HERO: keep the title clean and give the car real visual weight */
.st-key-hero_container{
  min-height:340px!important;
  margin-bottom:3rem!important;
  padding:2.5rem 2.9rem!important;
}
.st-key-hero_container [data-testid="stHorizontalBlock"]{
  gap:1.4rem!important;
}
.st-key-hero_container [data-testid="column"]{
  min-height:300px!important;
}
.st-key-hero_container .hero-title{
  white-space:nowrap!important;
  font-size:clamp(2.05rem,3.2vw,3.35rem)!important;
  line-height:1.05!important;
}
.st-key-hero_container .hero-description{
  max-width:660px!important;
  font-size:.98rem!important;
}
.st-key-hero_container [data-testid="stImage"]{
  min-height:300px!important;
  padding:0!important;
}
.st-key-hero_container [data-testid="stImage"] img{
  width:100%!important;
  max-width:620px!important;
  max-height:320px!important;
  height:auto!important;
  object-fit:contain!important;
  transform:scale(1.12)!important;
}
.st-key-hero_container [data-testid="stImage"] img:hover{
  transform:scale(1.17) translateY(-5px)!important;
}
.st-key-hero_container [data-testid="stImage"]:after{
  left:10%!important;
  right:10%!important;
  bottom:12px!important;
}

/* Give every row of cards visible breathing room */
div[data-testid="stHorizontalBlock"]{
  gap:1.65rem!important;
  margin-bottom:1.65rem!important;
  align-items:stretch!important;
}
div[data-testid="column"]{
  min-width:0!important;
}

/* Quality / KPI cards: clean, spacious, aligned */
.quality-card,
.kpi{
  min-height:128px!important;
  padding:1.2rem 1.25rem!important;
  margin:0!important;
  border-radius:16px!important;
}
.quality-card .quality-title,
.kpi-label{
  margin-bottom:.5rem!important;
  line-height:1.25!important;
}
.quality-card .quality-text{
  line-height:1.55!important;
}

/* Keep section rhythm consistent */
.section-title{
  margin-top:2.2rem!important;
  margin-bottom:.5rem!important;
}
.section-subtitle{
  margin-bottom:1.25rem!important;
}

/* Mobile: allow the title to wrap only when necessary */
@media(max-width:950px){
  .st-key-hero_container{
    min-height:0!important;
    padding:1.8rem 1.4rem!important;
  }
  .st-key-hero_container .hero-title{
    white-space:normal!important;
    font-size:2.2rem!important;
  }
  .st-key-hero_container [data-testid="stImage"]{
    min-height:220px!important;
  }
  .st-key-hero_container [data-testid="stImage"] img{
    max-width:430px!important;
    max-height:260px!important;
  }
  div[data-testid="stHorizontalBlock"]{
    gap:.85rem!important;
    margin-bottom:.85rem!important;
  }
}


/* =========================================================
   FINAL HERO + CARD SPACING
   ========================================================= */
.st-key-hero_container [data-testid="stHorizontalBlock"]{
  align-items:center!important;
}
.st-key-hero_container [data-testid="column"]:nth-child(2){
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  text-align:center!important;
  min-height:310px!important;
}
.st-key-hero_container [data-testid="column"]:nth-child(2) [data-testid="stImage"]{
  width:100%!important;
  min-height:300px!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
}
.st-key-hero_container [data-testid="column"]:nth-child(2) [data-testid="stImage"] img{
  display:block!important;
  width:auto!important;
  max-width:620px!important;
  max-height:320px!important;
  margin:0 auto!important;
  object-fit:contain!important;
  transform:scale(1.14)!important;
}
.st-key-hero_container [data-testid="column"]:nth-child(2) [data-testid="stImage"] img:hover{
  transform:scale(1.19) translateY(-5px)!important;
}


/* =========================================================
   CINEMATIC CAR HERO
   ========================================================= */
.st-key-hero_container{
  position:relative!important;
  overflow:hidden!important;
  min-height:355px!important;
  margin:0 0 3.2rem!important;
  padding:2.55rem 3rem!important;
  border:1px solid rgba(255,193,7,.42)!important;
  border-radius:24px!important;
  background:
    radial-gradient(circle at 78% 52%,rgba(255,193,7,.25) 0%,rgba(255,193,7,.10) 25%,transparent 48%),
    radial-gradient(circle at 96% 18%,rgba(255,193,7,.12),transparent 34%),
    linear-gradient(112deg,#05070A 0%,#0B0F14 48%,#17170F 100%)!important;
  box-shadow:
    0 22px 55px rgba(0,0,0,.48),
    inset 0 0 70px rgba(255,193,7,.035)!important;
}

/* Ambient light streak */
.st-key-hero_container::after{
  content:"";
  position:absolute;
  width:44%;
  height:3px;
  right:4%;
  bottom:54px;
  border-radius:99px;
  background:linear-gradient(90deg,transparent,#FFC107,#FFE082,transparent);
  box-shadow:0 0 22px rgba(255,193,7,.55);
  opacity:.85;
  animation:heroLight 4.5s ease-in-out infinite;
  pointer-events:none;
}

/* Hero layout */
.st-key-hero_container [data-testid="stHorizontalBlock"]{
  position:relative!important;
  z-index:2!important;
  align-items:center!important;
  gap:2.4rem!important;
}
.st-key-hero_container [data-testid="column"]{
  min-height:300px!important;
}
.st-key-hero_container [data-testid="column"]:first-child{
  display:flex!important;
  flex-direction:column!important;
  justify-content:center!important;
  padding-left:.2rem!important;
}
.st-key-hero_container [data-testid="column"]:nth-child(2){
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
}

/* Yellow accent */
.st-key-hero_container .hero-accent{
  width:86px!important;
  height:5px!important;
  margin-bottom:1.15rem!important;
  border-radius:99px!important;
  background:#FFC107!important;
  box-shadow:0 0 20px rgba(255,193,7,.55)!important;
  animation:accentGlow 2.8s ease-in-out infinite!important;
}

/* Title */
.st-key-hero_container .hero-title{
  margin:0!important;
  color:#FFFFFF!important;
  font-size:clamp(2.35rem,4vw,4rem)!important;
  line-height:1.03!important;
  font-weight:900!important;
  letter-spacing:-.045em!important;
  white-space:nowrap!important;
  animation:heroText .75s cubic-bezier(.2,.8,.2,1) both!important;
}

/* Description */
.st-key-hero_container .hero-description{
  max-width:650px!important;
  margin:.95rem 0 0!important;
  color:#D1D5DB!important;
  font-size:1rem!important;
  line-height:1.65!important;
  animation:heroText .9s .08s cubic-bezier(.2,.8,.2,1) both!important;
}

/* Badge */
.st-key-hero_container .hero-badge{
  display:inline-flex!important;
  width:max-content!important;
  margin-top:1.05rem!important;
  padding:.55rem .85rem!important;
  border:1px solid rgba(255,193,7,.48)!important;
  border-radius:999px!important;
  background:rgba(255,193,7,.08)!important;
  color:#FFD54A!important;
  font-size:.74rem!important;
  font-weight:800!important;
  letter-spacing:.01em!important;
}

/* Car */
.st-key-hero_container [data-testid="stImage"]{
  width:100%!important;
  min-height:310px!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  padding:0!important;
  position:relative!important;
}
.st-key-hero_container [data-testid="stImage"]::before{
  content:"";
  position:absolute;
  width:72%;
  height:52%;
  border-radius:50%;
  background:radial-gradient(ellipse,rgba(255,193,7,.25),rgba(255,193,7,.07) 42%,transparent 72%);
  filter:blur(18px);
  animation:carGlow 4s ease-in-out infinite;
}
.st-key-hero_container [data-testid="stImage"] img{
  position:relative!important;
  z-index:2!important;
  display:block!important;
  width:auto!important;
  max-width:100%!important;
  max-height:310px!important;
  object-fit:contain!important;
  margin:0 auto!important;
  filter:
    drop-shadow(0 25px 30px rgba(0,0,0,.58))
    drop-shadow(0 0 20px rgba(255,193,7,.14))!important;
  transform:scale(1.14)!important;
  animation:carEntrance 1s cubic-bezier(.18,.82,.22,1) both,carFloat 4.8s 1s ease-in-out infinite!important;
  transition:transform .45s ease,filter .45s ease!important;
}
.st-key-hero_container [data-testid="stImage"] img:hover{
  transform:scale(1.2) translateY(-6px)!important;
  filter:
    drop-shadow(0 30px 35px rgba(0,0,0,.65))
    drop-shadow(0 0 30px rgba(255,193,7,.25))!important;
}

/* Hero motion */
@keyframes heroText{
  from{opacity:0;transform:translateY(14px)}
  to{opacity:1;transform:translateY(0)}
}
@keyframes carEntrance{
  from{opacity:0;transform:translateX(45px) scale(.92)}
  to{opacity:1;transform:translateX(0) scale(1.14)}
}
@keyframes carFloat{
  0%,100%{transform:translateY(0) scale(1.14)}
  50%{transform:translateY(-8px) scale(1.14)}
}
@keyframes carGlow{
  0%,100%{opacity:.65;transform:scale(.92)}
  50%{opacity:1;transform:scale(1.06)}
}
@keyframes accentGlow{
  0%,100%{width:86px;box-shadow:0 0 14px rgba(255,193,7,.35)}
  50%{width:112px;box-shadow:0 0 24px rgba(255,193,7,.65)}
}
@keyframes heroLight{
  0%,100%{opacity:.45;transform:scaleX(.82)}
  50%{opacity:1;transform:scaleX(1)}
}

@media(max-width:950px){
  .st-key-hero_container{
    min-height:0!important;
    padding:2rem 1.4rem!important;
  }
  .st-key-hero_container [data-testid="stHorizontalBlock"]{
    gap:1rem!important;
  }
  .st-key-hero_container .hero-title{
    white-space:normal!important;
    font-size:2.35rem!important;
  }
  .st-key-hero_container [data-testid="column"]:nth-child(2){
    min-height:230px!important;
  }
  .st-key-hero_container [data-testid="stImage"]{
    min-height:230px!important;
  }
  .st-key-hero_container [data-testid="stImage"] img{
    max-height:250px!important;
    transform:scale(1.05)!important;
  }
}

@media(prefers-reduced-motion:reduce){
  .st-key-hero_container *,
  .st-key-hero_container::after{
    animation:none!important;
    transition:none!important;
  }
}


/* =========================================================
   COMPACT OWNERSHIP FOOTER
   ========================================================= */
.app-footer{
  position:relative!important;
  margin:2.2rem 0 .4rem!important;
  padding:.7rem 1rem!important;
  border:1px solid rgba(255,193,7,.18)!important;
  border-radius:12px!important;
  background:#0D1218!important;
  box-shadow:0 6px 18px rgba(0,0,0,.18)!important;
  text-align:center!important;
}
.app-footer:before{
  content:""!important;
  position:absolute!important;
  top:-1px!important;
  left:50%!important;
  width:55px!important;
  height:2px!important;
  transform:translateX(-50%)!important;
  border-radius:99px!important;
  background:#FFC107!important;
  box-shadow:0 0 10px rgba(255,193,7,.45)!important;
}
.app-footer-copy{
  margin:0!important;
  color:#AEB6C2!important;
  font-size:.66rem!important;
  font-weight:600!important;
  line-height:1.5!important;
}
.app-footer-copy .owner{
  color:#FFC107!important;
  font-weight:800!important;
}
.app-footer-links{
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  gap:.45rem!important;
  margin-left:.5rem!important;
}
.app-footer-links a{
  color:#9CA6B3!important;
  text-decoration:none!important;
  font-size:.63rem!important;
  font-weight:700!important;
  transition:color .2s ease,transform .2s ease!important;
}
.app-footer-links a:hover{
  color:#FFD54A!important;
  transform:translateY(-1px)!important;
}
.app-footer-separator{
  color:#3F4650!important;
}
@media(max-width:650px){
  .app-footer{margin-top:1.5rem!important;padding:.65rem .7rem!important}
  .app-footer-copy{font-size:.61rem!important}
  .app-footer-links{margin-left:.25rem!important;gap:.3rem!important}
  .app-footer-links a{font-size:.59rem!important}
}
</style>
""",
    unsafe_allow_html=True,
)


def apply_plot_theme(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR), height=height,
        margin=dict(l=24, r=24, t=64, b=34), hoverlabel=dict(bgcolor="#0B1118", font_color=FONT_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)"), title=dict(x=0.02, xanchor="left"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    return fig


@st.cache_data(show_spinner="Loading verified Task 1 dataset…")
def load_data(path: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError("The required dataset file was not found beside app.py.")
    return pd.read_csv(source)


@st.cache_data(show_spinner=False)
def correlation_series(data: pd.DataFrame, target: str) -> pd.Series:
    numeric = data.select_dtypes(include=np.number)
    if target not in numeric.columns:
        return pd.Series(dtype=float)
    return numeric.corr(numeric_only=True)[target].drop(target, errors="ignore").sort_values(key=np.abs, ascending=False)


@st.cache_data(show_spinner=False)
def filtered_search_table(data: pd.DataFrame, columns: tuple[str, ...], query: str) -> pd.DataFrame:
    table = data.loc[:, list(columns)].copy()
    q = query.strip()
    if not q:
        return table
    text_cols = table.select_dtypes(include=["object", "string"]).columns.tolist()
    if not text_cols:
        return table.iloc[0:0]
    mask = table[text_cols].astype(str).apply(lambda s: s.str.contains(q, case=False, na=False, regex=False)).any(axis=1)
    return table.loc[mask]


def pearson(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))


def sample_for_plot(data: pd.DataFrame, n: int) -> pd.DataFrame:
    return data if len(data) <= n else data.sample(n=n, random_state=42)


def kpi_card(label: str, value: str) -> None:
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)


def section(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def empty_state(message: str = "No records match the current filters.") -> None:
    st.markdown(f'<div class="empty-state"><h3>{message}</h3><p>Broaden one or more filters, or use <b>Reset All Filters</b>.</p></div>', unsafe_allow_html=True)


def filter_summary(label: str, selected: list, total: int) -> str:
    count = len(selected)
    if count == total:
        return f"{label} · All {total}"
    return f"{label} · {count} of {total}"


def init_state(df: pd.DataFrame) -> None:
    defaults = {
        "nav": PAGES[0],
        "hour_filter": list(range(24)),
        "hour_range": (0, 23),
        "car_filter": sorted(df["Car Condition"].dropna().unique().tolist()),
        "weather_filter": sorted(df["Weather"].dropna().unique().tolist()),
        "traffic_filter": sorted(df["Traffic Condition"].dropna().unique().tolist()),
        "passenger_filter": [int(x) for x in sorted(df["passenger_count"].dropna().unique())],
        "year_filter": [int(x) for x in sorted(df["year"].dropna().unique())],
        "month_filter": [int(x) for x in sorted(df["month"].dropna().unique())],
        "airport": "JFK",
        "distance_view": "Raw Data",
        "fare_bins": 50,
        "scatter_sample": 7000,
        "corr_target": "fare_amount",
        "corr_vars": [c for c in NUMERIC_DEFAULTS if c in df.columns],
        "table_search": "",
        "table_columns_widget": [c for c in ANALYTICAL_COLUMNS if c in df.columns],
        "demand_hour": 19,
        "car_inspect": "All",
        "traffic_inspect": "All",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_all_filters(df: pd.DataFrame) -> None:
    st.session_state.hour_filter = list(range(24))
    st.session_state.hour_range = (0, 23)
    st.session_state.car_filter = sorted(df["Car Condition"].dropna().unique().tolist())
    st.session_state.weather_filter = sorted(df["Weather"].dropna().unique().tolist())
    st.session_state.traffic_filter = sorted(df["Traffic Condition"].dropna().unique().tolist())
    st.session_state.passenger_filter = [int(x) for x in sorted(df["passenger_count"].dropna().unique())]
    st.session_state.year_filter = [int(x) for x in sorted(df["year"].dropna().unique())]
    st.session_state.month_filter = [int(x) for x in sorted(df["month"].dropna().unique())]


def build_filtered(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["hour"].isin(st.session_state.hour_filter)
        & df["Car Condition"].isin(st.session_state.car_filter)
        & df["Weather"].isin(st.session_state.weather_filter)
        & df["Traffic Condition"].isin(st.session_state.traffic_filter)
        & df["passenger_count"].isin(st.session_state.passenger_filter)
        & df["year"].isin(st.session_state.year_filter)
        & df["month"].isin(st.session_state.month_filter)
    )
    return df.loc[mask]


def _make_fallback_taxi() -> Image.Image:
    """Create a small in-memory taxi illustration when assets/taxi.png is unavailable."""
    W, H = 900, 420
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((180, 245, 720, 355), fill=(255, 193, 7, 45))
    glow = glow.filter(ImageFilter.GaussianBlur(22))
    img.alpha_composite(glow)
    d = ImageDraw.Draw(img)
    yellow=(255,193,7,255); dark=(10,14,18,255); glass=(38,52,64,255); light=(255,246,185,255)
    # shadow
    d.ellipse((155,315,750,365), fill=(0,0,0,90))
    # body
    body=[(105,285),(140,205),(235,184),(320,112),(420,78),(530,82),(640,178),(735,205),(785,270),(760,310),(120,310)]
    d.polygon(body, fill=yellow)
    d.line(body+[body[0]], fill=(226,170,0,255), width=7, joint='curve')
    # windows
    d.polygon([(330,177),(350,120),(420,96),(455,96),(455,177)], fill=glass)
    d.polygon([(468,96),(525,100),(600,175),(468,175)], fill=glass)
    d.line((460,98,460,178), fill=(10,14,18,255), width=6)
    # taxi sign
    d.rounded_rectangle((385,52,505,91), radius=9, fill=yellow, outline=(226,170,0,255), width=4)
    d.rounded_rectangle((406,60,484,82), radius=6, fill=light)
    # wheels
    for cx in (230,650):
        d.ellipse((cx-58,248,cx+58,364), fill=dark, outline=(48,58,68,255), width=10)
        d.ellipse((cx-24,282,cx+24,330), fill=(145,153,160,255))
        d.ellipse((cx-10,296,cx+10,316), fill=(34,41,48,255))
    # lights and side detail
    d.rounded_rectangle((125,220,180,247), radius=8, fill=(250,250,250,255))
    d.rounded_rectangle((700,220,754,247), radius=8, fill=(250,250,250,255))
    d.rounded_rectangle((350,188,520,224), radius=8, fill=dark)
    d.text((435,196), "TAXI", fill=yellow, anchor="mm", stroke_width=0)
    for x in (300,365,430,495,560):
        d.rounded_rectangle((x,232,x+52,242), radius=5, fill=light)
    return img


def _taxi_image_for_streamlit():
    """Prefer the user's taxi asset; otherwise use a reliable generated fallback."""
    if TAXI_IMAGE.exists():
        return str(TAXI_IMAGE)
    return _make_fallback_taxi()

def render_header() -> None:
    """Render a robust Streamlit-native hero so the taxi cannot be printed as HTML text."""
    with st.container(key="hero_container"):
        left, right = st.columns([0.92, 1.08], gap="large")
        with left:
            st.markdown('<div class="hero-accent"></div>', unsafe_allow_html=True)
            st.markdown('<h1 class="hero-title">Uber Fare — Interactive EDA</h1>', unsafe_allow_html=True)
            st.markdown('<p class="hero-description">Explore observed patterns in fares, demand, distance, traffic, weather, and trip characteristics.</p>', unsafe_allow_html=True)
            st.markdown('<span class="hero-tag">Interactive analytics · 49,991 verified records</span>', unsafe_allow_html=True)
        with right:
            st.image(_taxi_image_for_streamlit(), use_container_width=True)

def render_overview(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    section("Overview", "A live summary of the currently filtered records.")
    values = [
        ("Total Trips", f"{len(filtered):,}"),
        ("Average Fare", f"${filtered.fare_amount.mean():,.2f}"),
        ("Median Fare", f"${filtered.fare_amount.median():,.2f}"),
        ("Average Distance", f"{filtered.distance.mean():,.2f} km"),
        ("Median Distance", f"{filtered.distance.median():,.2f} km"),
        ("Avg Passenger Count", f"{filtered.passenger_count.mean():,.2f}"),
    ]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, values):
        with col:
            kpi_card(label, value)

    active = []
    if len(st.session_state.hour_filter) != 24: active.append(f"Hour: {min(st.session_state.hour_filter):02d}:00–{max(st.session_state.hour_filter):02d}:00")
    for label, key, total in [("Car", "car_filter", df["Car Condition"].nunique()), ("Weather", "weather_filter", df["Weather"].nunique()), ("Traffic", "traffic_filter", df["Traffic Condition"].nunique()), ("Passengers", "passenger_filter", df["passenger_count"].nunique()), ("Year", "year_filter", df["year"].nunique()), ("Month", "month_filter", df["month"].nunique())]:
        if len(st.session_state[key]) != total:
            active.append(f"{label}: {len(st.session_state[key])}/{total}")
    section("Current Filter State")
    st.write("All available values" if not active else " · ".join(active))

    top_hour = int(filtered.groupby("hour").size().idxmax())
    st.markdown(f'<div class="insight"><strong>Key Insight</strong> Highest observed trip volume in the current filtered records occurs at <b>{top_hour:02d}:00</b>. This is an observational pattern, not a causal conclusion.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        counts = filtered.groupby("hour").size().reindex(range(24), fill_value=0).reset_index(name="trip_count")
        fig = px.bar(counts, x="hour", y="trip_count", title="Current Filter — Demand by Hour", labels={"hour":"Hour","trip_count":"Trip Count"})
        fig.update_traces(marker_color=ACCENT, hovertemplate="Hour %{x}: %{y:,} trips<extra></extra>")
        st.plotly_chart(apply_plot_theme(fig, 400), use_container_width=True)
    with c2:
        fig = px.histogram(filtered, x="fare_amount", nbins=50, title="Current Filter — Fare Distribution", labels={"fare_amount":"Fare Amount"})
        fig.update_traces(marker_color=ACCENT, hovertemplate="Fare %{x:$,.2f}<br>Count %{y:,}<extra></extra>")
        st.plotly_chart(apply_plot_theme(fig, 400), use_container_width=True)


def render_fare(filtered: pd.DataFrame) -> None:
    section("Fare Analysis", "Distribution and observed fare differences across time, car condition, and traffic.")
    bins = st.slider("Histogram bins", 20, 100, key="fare_bins")
    fig = px.histogram(filtered, x="fare_amount", nbins=bins, title="Fare Distribution", labels={"fare_amount":"Fare Amount"})
    fig.update_traces(marker_color=ACCENT, hovertemplate="Fare %{x:$,.2f}<br>Count %{y:,}<extra></extra>")
    st.plotly_chart(apply_plot_theme(fig), use_container_width=True)

    hourly = filtered.groupby("hour", as_index=False)["fare_amount"].mean()
    peak = hourly.loc[hourly.fare_amount.idxmax()]
    fig = px.line(hourly, x="hour", y="fare_amount", markers=True, title="Average Fare by Hour", labels={"hour":"Hour","fare_amount":"Average Fare"})
    fig.update_traces(line_color=SECONDARY, marker_color=ACCENT, hovertemplate="Hour %{x}: %{y:$,.2f}<extra></extra>")
    fig.add_trace(go.Scatter(x=[peak.hour], y=[peak.fare_amount], mode="markers+text", text=["Highest observed average"], textposition="top center", marker=dict(size=11, color=ACCENT), name="Peak observed mean"))
    st.plotly_chart(apply_plot_theme(fig), use_container_width=True)
    st.caption(f"Highest observed average fare: {int(peak.hour):02d}:00 at ${peak.fare_amount:,.2f}. Observational EDA only.")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(filtered, x="Car Condition", y="fare_amount", points=False, title="Fare by Car Condition")
        st.plotly_chart(apply_plot_theme(fig, 430), use_container_width=True)
    with c2:
        fig = px.box(filtered, x="Traffic Condition", y="fare_amount", points=False, title="Fare by Traffic Condition")
        st.plotly_chart(apply_plot_theme(fig, 430), use_container_width=True)


def render_demand(filtered: pd.DataFrame) -> None:
    section("Demand Analysis", "Observed request volume by hour, with a separate view of fare level.")
    counts = filtered.groupby("hour").size().reindex(range(24), fill_value=0).reset_index(name="trip_count")
    fig = px.bar(counts, x="hour", y="trip_count", title="Demand by Hour", labels={"hour":"Hour","trip_count":"Trip Count"})
    fig.update_traces(marker_color=SECONDARY, hovertemplate="Hour %{x}: %{y:,} trips<extra></extra>")
    st.plotly_chart(apply_plot_theme(fig, 460), use_container_width=True)
    selected = st.select_slider("Inspect an hour", options=list(range(24)), key="demand_hour")
    sub = filtered[filtered.hour == selected]
    a,b,c = st.columns(3)
    with a: kpi_card("Selected Hour", f"{selected:02d}:00")
    with b: kpi_card("Trip Count", f"{len(sub):,}")
    with c: kpi_card("Average Fare", f"${sub.fare_amount.mean():,.2f}" if len(sub) else "—")
    st.caption("Demand and fare level are separate measures; a high trip count does not imply a high average fare.")


def render_passenger(filtered: pd.DataFrame) -> None:
    section("Passenger Analysis", "Passenger count versus fare, with statistics calculated from all filtered records.")
    plot_df = sample_for_plot(filtered, st.session_state.scatter_sample)
    fig = px.scatter(plot_df, x="passenger_count", y="fare_amount", opacity=.45, title="Passenger Count vs Fare", labels={"passenger_count":"Passenger Count","fare_amount":"Fare Amount"}, hover_data=["hour","distance","Weather","Traffic Condition"])
    fig.update_traces(marker=dict(size=5, color=ACCENT))
    st.plotly_chart(apply_plot_theme(fig, 470), use_container_width=True)
    if len(filtered) > st.session_state.scatter_sample:
        st.caption("Visualization uses a deterministic sample for rendering; statistics use all filtered records.")
    r = pearson(filtered.passenger_count, filtered.fare_amount)
    a,b = st.columns(2)
    with a: kpi_card("Pearson Correlation", f"{r:.3f}" if pd.notna(r) else "—")
    with b: kpi_card("Filtered Observations", f"{len(filtered):,}")
    st.markdown('<div class="insight"><strong>Interpretation</strong> Passenger count shows a dynamically calculated linear association with fare in the current filtered data. <b>Correlation ≠ causation.</b></div>', unsafe_allow_html=True)


def render_car_traffic(filtered: pd.DataFrame) -> None:
    section("Car & Traffic", "Descriptive group comparisons of fare by car condition and traffic condition.")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.box(filtered, x="Car Condition", y="fare_amount", points=False, title="Fare by Car Condition")
        st.plotly_chart(apply_plot_theme(fig), use_container_width=True)
        options = ["All"] + sorted(filtered["Car Condition"].dropna().unique().tolist())
        if st.session_state.car_inspect not in options: st.session_state.car_inspect = "All"
        car = st.selectbox("Inspect car condition", options, key="car_inspect")
        sub = filtered if car == "All" else filtered[filtered["Car Condition"] == car]
        st.dataframe(pd.DataFrame({"Metric":["Mean","Median","Count"],"Value":[f"${sub.fare_amount.mean():.2f}",f"${sub.fare_amount.median():.2f}",f"{len(sub):,}"]}), hide_index=True, use_container_width=True)
    with c2:
        fig = px.box(filtered, x="Traffic Condition", y="fare_amount", points=False, title="Fare by Traffic Condition")
        st.plotly_chart(apply_plot_theme(fig), use_container_width=True)
        options = ["All"] + sorted(filtered["Traffic Condition"].dropna().unique().tolist())
        if st.session_state.traffic_inspect not in options: st.session_state.traffic_inspect = "All"
        traffic = st.selectbox("Inspect traffic condition", options, key="traffic_inspect")
        sub = filtered if traffic == "All" else filtered[filtered["Traffic Condition"] == traffic]
        st.dataframe(pd.DataFrame({"Metric":["Mean","Median","Count"],"Value":[f"${sub.fare_amount.mean():.2f}",f"${sub.fare_amount.median():.2f}",f"{len(sub):,}"]}), hide_index=True, use_container_width=True)
    st.markdown('<div class="insight"><strong>Interpretation</strong> Group differences are descriptive observations from the filtered data and are not interpreted as causal effects.</div>', unsafe_allow_html=True)


def distance_subset(filtered: pd.DataFrame, view: str) -> tuple[pd.DataFrame,str]:
    if view == "Raw Data":
        return filtered, "Raw view: all currently filtered observations."
    subset = filtered.loc[filtered.distance <= 50]
    if view == "Robust View — Distance ≤ 50 km":
        return subset, "Robust view: distance ≤ 50 km is an analytical comparison threshold, not a universal validity rule."
    coords = ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]
    mask = filtered.distance.le(50) & filtered[coords].notna().all(axis=1)
    return filtered.loc[mask], "Robust view: recorded coordinates are non-missing and distance ≤ 50 km."


def render_distance(filtered: pd.DataFrame) -> None:
    section("Distance Analysis", "Compare raw distance behavior with clearly labeled robustness views.")
    view = st.radio("Analysis View", ["Raw Data","Robust View — Distance ≤ 50 km","Robust View — Valid Geo + Distance ≤ 50 km"], horizontal=True, key="distance_view")
    subset,note = distance_subset(filtered,view)
    if subset.empty:
        empty_state("No records remain in this distance view.")
        return
    plot_df = sample_for_plot(subset, st.session_state.scatter_sample)
    fig = px.scatter(plot_df, x="distance", y="fare_amount", opacity=.4, title=f"Distance vs Fare — {view}", labels={"distance":"Distance (km)","fare_amount":"Fare Amount"}, hover_data=["hour","passenger_count","Weather","Traffic Condition"])
    fig.update_traces(marker=dict(size=5,color=ACCENT))
    st.plotly_chart(apply_plot_theme(fig,500), use_container_width=True)
    r,rho = pearson(subset.distance,subset.fare_amount), spearman(subset.distance,subset.fare_amount)
    a,b,c=st.columns(3)
    with a: kpi_card("Pearson",f"{r:.3f}" if pd.notna(r) else "—")
    with b: kpi_card("Spearman",f"{rho:.3f}" if pd.notna(rho) else "—")
    with c: kpi_card("Observations",f"{len(subset):,}")
    st.caption(note + (" Visualization uses a deterministic sample; statistics use all records in the selected view." if len(subset)>st.session_state.scatter_sample else " Statistics use all records in the selected view."))
    st.markdown('<div class="insight"><strong>Interpretation</strong> Extreme distance observations can materially affect the raw linear relationship. Robust views are diagnostic comparisons, not blanket deletion rules.</div>', unsafe_allow_html=True)


def render_weather(filtered: pd.DataFrame) -> None:
    section("Weather Analysis", "Compare trip-distance distributions and descriptive summaries across weather categories.")
    fig = px.box(filtered, x="Weather", y="distance", points=False, title="Weather vs Trip Distance", labels={"distance":"Distance (km)"})
    st.plotly_chart(apply_plot_theme(fig,450), use_container_width=True)
    table = filtered.groupby("Weather").distance.agg(Count="count", Mean_Distance="mean", Median_Distance="median").reset_index()
    st.dataframe(table.style.format({"Mean_Distance":"{:.2f}","Median_Distance":"{:.2f}"}), use_container_width=True, hide_index=True)
    st.markdown('<div class="insight"><strong>Interpretation</strong> Mean distance is sensitive to extreme observations; median distance is a useful complementary summary. No causal claim is made about weather.</div>', unsafe_allow_html=True)


def render_airport(filtered: pd.DataFrame) -> None:
    section("Airport Analysis", "Explore airport-distance versus fare for JFK, EWR, or LGA.")
    airport_map={"JFK":"jfk_dist","EWR":"ewr_dist","LGA":"lga_dist"}
    airport=st.selectbox("Airport",list(airport_map),key="airport")
    col=airport_map[airport]
    plot_df=sample_for_plot(filtered,st.session_state.scatter_sample)
    fig=px.scatter(plot_df,x=col,y="fare_amount",opacity=.4,title=f"{airport} Distance vs Fare",labels={col:f"{airport} Distance (km)","fare_amount":"Fare Amount"},hover_data=["distance","hour","Weather","Traffic Condition"])
    fig.update_traces(marker=dict(size=5,color=SECONDARY))
    st.plotly_chart(apply_plot_theme(fig,470),use_container_width=True)
    r=pearson(filtered[col],filtered.fare_amount)
    kpi_card("Pearson Correlation",f"{r:.3f}" if pd.notna(r) else "—")
    if len(filtered)>st.session_state.scatter_sample: st.caption("Visualization uses a deterministic sample for rendering; the correlation uses all filtered records.")
    st.markdown('<div class="insight"><strong>Interpretation</strong> Raw linear association is near zero in the verified dataset for these airport-distance variables. This does not prove that airport distance has no effect.</div>',unsafe_allow_html=True)


def render_correlation(filtered: pd.DataFrame) -> None:
    section("Correlation Explorer", "Explore Pearson correlations among numerical variables in the current filtered data.")
    numeric=filtered.select_dtypes(include=np.number).columns.tolist()
    if len(numeric)<2:
        empty_state("At least two numerical variables are required for correlation analysis.")
        return
    if st.session_state.corr_target not in numeric: st.session_state.corr_target=numeric[0]
    target=st.selectbox("Target variable",numeric,key="corr_target")
    corr=correlation_series(filtered,target)
    if corr.empty:
        empty_state("Correlation cannot be calculated for the selected target.")
        return
    fig=px.bar(corr.reset_index(),x=corr.values,y="index",orientation="h",title=f"Pearson Correlation with {target}",labels={"x":"Pearson Correlation","index":"Variable"})
    fig.update_traces(marker_color=ACCENT,hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    st.plotly_chart(apply_plot_theme(fig,max(430,32*len(corr))),use_container_width=True)
    st.dataframe(corr.rename("Pearson Correlation").to_frame().reset_index(),use_container_width=True,hide_index=True)
    default_vars=[c for c in st.session_state.corr_vars if c in numeric]
    if len(default_vars)<2: default_vars=numeric[:min(8,len(numeric))]
    st.session_state.corr_vars=default_vars
    selected=st.multiselect("Variables for heatmap",numeric,key="corr_vars")
    if len(selected)>=2:
        matrix=filtered[selected].corr()
        heat=px.imshow(matrix,text_auto=".2f",color_continuous_scale="RdBu_r",title="Selected Numerical Correlation Heatmap",zmin=-1,zmax=1)
        st.plotly_chart(apply_plot_theme(heat,max(430,45*len(selected))),use_container_width=True)
    st.markdown('<div class="insight"><strong>Important</strong> Correlation describes statistical association, not causation.</div>',unsafe_allow_html=True)


def render_data_explorer(filtered: pd.DataFrame) -> None:
    section("Data Explorer", "Inspect analytical fields from the current filtered dataframe and download exactly what is shown.")
    analytical=[c for c in ANALYTICAL_COLUMNS if c in filtered.columns]
    valid_selected=[c for c in st.session_state.table_columns_widget if c in analytical]
    if not valid_selected: valid_selected=analytical
    selected_cols=st.multiselect("Columns",analytical,default=valid_selected,key="table_columns_widget")
    if not selected_cols:
        st.info("Select at least one column to display records.")
        return
    query=st.text_input("Search visible text columns",key="table_search",placeholder="Search weather, traffic, car condition, or pickup timestamp…")
    table=filtered_search_table(filtered,tuple(selected_cols),query)
    c1,c2,c3=st.columns(3)
    with c1: kpi_card("Filtered Rows",f"{len(filtered):,}")
    with c2: kpi_card("Rows Shown",f"{len(table):,}")
    with c3: kpi_card("Columns",f"{len(selected_cols):,}")
    if query.strip(): st.caption("Rows shown reflect both the global filters and the local text search. Download matches the table shown.")
    st.dataframe(table,use_container_width=True,height=520,hide_index=True)
    buffer=io.BytesIO(); table.to_csv(buffer,index=False)
    st.download_button("Download Filtered Data",data=buffer.getvalue(),file_name="uber_fare_filtered.csv",mime="text/csv",type="primary",use_container_width=True)


def render_quality(df: pd.DataFrame) -> None:
    section("Data Quality", "Documented Task 1 data-quality findings plus dynamic checks on the loaded file.")
    missing=int(df.isna().sum().sum())
    dup_rows=int(df.duplicated().sum())
    zero_passengers=int((df.passenger_count==0).sum())
    non_positive=int((df.fare_amount<=0).sum())
    zero_distance=int((df.distance==0).sum())
    key_dups=int(df["key"].duplicated().sum()) if "key" in df.columns else 0
    cards=[
        ("◌","Missing values",f"{missing:,} total missing cells in the loaded dataset"),
        ("$","Fare quality",f"{non_positive:,} non-positive fares present in loaded CSV"),
        ("♙","Passenger anomalies",f"{zero_passengers:,} zero-passenger records retained"),
        ("↗","Distance anomalies",f"{zero_distance:,} zero-distance records; extreme values retained"),
        ("◆","Full duplicate rows",f"{dup_rows:,} exact duplicate rows"),
        ("⌁","Key duplicates",f"{key_dups:,} duplicate key occurrences beyond first"),
    ]
    cols=st.columns(3,gap="medium")
    for i,(icon,title,body) in enumerate(cards):
        with cols[i%3]:
            st.markdown(f"""<div class="quality-card">
                <div class="quality-icon">{icon}</div>
                <div class="quality-title">{title}</div>
                <div class="quality-body">{body}</div>
            </div>""",unsafe_allow_html=True)
    section("Documented Analytical Decisions")
    st.info("Non-positive fare values were excluded before the verified cleaned EDA dataset was exported. Zero-passenger records were retained as documented anomalies. Extreme distance observations were not automatically deleted. Full duplicate rows were not found in the verified Task 1 dataset; key duplicates were investigated and not automatically removed.")
    section("Loaded Dataset")
    st.write(f"Rows: **{len(df):,}** · Columns: **{df.shape[1]}** · Source: **{DATA_PATH.name}**")
    st.caption("This application is Task 1 EDA only. It does not train models, generate predictions, or make causal claims.")

def render_sidebar(df: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown("# 🚕 Uber Fare EDA")
        st.caption("Interactive Task 1 analytics")
        st.markdown("### Navigation")
        st.radio("Go to",PAGES,key="nav",label_visibility="collapsed")
        st.divider()
        st.markdown("### Filters")
        # Button is intentionally before filter widgets so its callback can safely reset widget state before instantiation.
        st.button("↺ Reset All Filters",use_container_width=True,type="secondary",on_click=reset_all_filters,args=(df,))
        with st.expander(filter_summary("🕐 Hour",st.session_state.hour_filter,24),expanded=False):
            start,end=st.slider("Time range",0,23,key="hour_range",format="%02d:00")
            st.session_state.hour_filter=list(range(start,end+1))
        with st.expander(filter_summary("🚗 Car Condition",st.session_state.car_filter,df["Car Condition"].nunique()),expanded=False):
            st.multiselect("Select conditions",sorted(df["Car Condition"].dropna().unique()),key="car_filter",label_visibility="collapsed")
        with st.expander(filter_summary("☁ Weather",st.session_state.weather_filter,df["Weather"].nunique()),expanded=False):
            st.multiselect("Select weather",sorted(df["Weather"].dropna().unique()),key="weather_filter",label_visibility="collapsed")
        with st.expander(filter_summary("🚦 Traffic",st.session_state.traffic_filter,df["Traffic Condition"].nunique()),expanded=False):
            st.multiselect("Select traffic",sorted(df["Traffic Condition"].dropna().unique()),key="traffic_filter",label_visibility="collapsed")
        with st.expander(filter_summary("👥 Passengers",st.session_state.passenger_filter,df["passenger_count"].nunique()),expanded=False):
            st.multiselect("Select passenger counts",sorted(df["passenger_count"].dropna().unique()),key="passenger_filter",label_visibility="collapsed")
        with st.expander(filter_summary("📅 Year",st.session_state.year_filter,df["year"].nunique()),expanded=False):
            st.multiselect("Select years",sorted(df["year"].dropna().unique()),key="year_filter",label_visibility="collapsed")
        with st.expander(filter_summary("📆 Month",st.session_state.month_filter,df["month"].nunique()),expanded=False):
            st.multiselect("Select months",sorted(df["month"].dropna().unique()),key="month_filter",label_visibility="collapsed")
        filtered_preview=build_filtered(df)
        st.divider(); st.metric("Filtered Trips",f"{len(filtered_preview):,}")
        with st.expander("⚙️ Display Options",expanded=False):
            st.slider("Scatter rendering sample",1000,10000,key="scatter_sample",step=500)
            st.caption("KPI and correlation statistics always use all filtered records.")



def render_footer() -> None:
    """Render a compact ownership footer on every application page."""
    st.markdown(
        """
        <div class="app-footer">
            <div class="app-footer-copy">
                © 2026 <span class="owner">Noura Maher Elamin</span> · All rights reserved.
                <span class="app-footer-links">
                    <span class="app-footer-separator">|</span>
                    <a href="https://www.linkedin.com/in/nouramaherelamin/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
                    <span class="app-footer-separator">·</span>
                    <a href="https://github.com/nouramaherelamin" target="_blank" rel="noopener noreferrer">GitHub</a>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def validate_schema(df: pd.DataFrame) -> list[str]:
    required={"fare_amount","distance","passenger_count","hour","month","year","Car Condition","Weather","Traffic Condition","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"}
    return sorted(required-set(df.columns))


def main() -> None:
    try:
        df=load_data(str(DATA_PATH))
    except Exception:
        st.error("Unable to load the verified dataset. Make sure uber_fare_eda_clean.csv is beside app.py.")
        st.stop()
    missing=validate_schema(df)
    if missing:
        st.error("The dataset is missing required analytical columns. Please use the verified Task 1 CSV.")
        st.stop()
    init_state(df)
    render_sidebar(df)
    filtered=build_filtered(df)
    render_header()
    if filtered.empty:
        empty_state()
        st.stop()
    page=st.session_state.nav
    if page=="Overview": render_overview(df,filtered)
    elif page=="Fare Analysis": render_fare(filtered)
    elif page=="Demand Analysis": render_demand(filtered)
    elif page=="Passenger Analysis": render_passenger(filtered)
    elif page=="Car & Traffic": render_car_traffic(filtered)
    elif page=="Distance Analysis": render_distance(filtered)
    elif page=="Weather Analysis": render_weather(filtered)
    elif page=="Airport Analysis": render_airport(filtered)
    elif page=="Correlation Explorer": render_correlation(filtered)
    elif page=="Data Explorer": render_data_explorer(filtered)
    elif page=="Data Quality": render_quality(df)

    render_footer()


if __name__ == "__main__":
    main()
