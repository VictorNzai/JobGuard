import streamlit as st

st.set_page_config(page_title="JobGuard", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {background: linear-gradient(135deg, #0f766e 0%, #134e4a 100%); color: #fff; border-radius: 18px;
       padding: 1.6rem 1.8rem; margin-bottom: 1.4rem;}
.hero h1 {color: #fff; margin: 0; font-size: 2rem; padding: 0;}
.hero p {margin: .4rem 0 0; opacity: .88; font-size: .98rem;}
.step {font-weight: 700; color: #0f766e; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase;}
.verdict {border-radius: 16px; padding: 1.2rem 1.4rem; display: flex; align-items: center; gap: 1.2rem; margin: 1rem 0;}
.verdict.bad {background: #fef2f2; border: 1px solid #fecaca;}
.verdict.good {background: #ecfdf5; border: 1px solid #a7f3d0;}
.verdict .pct {font-size: 2.8rem; font-weight: 800; line-height: 1;}
.bad .pct, .bad .lbl {color: #b91c1c;} .good .pct, .good .lbl {color: #047857;}
.verdict .lbl {font-weight: 700; font-size: 1.1rem;}
.verdict .sub {color: #4b5d5b; font-size: .85rem;}
.bar {height: 8px; border-radius: 99px; background: #e5ece9; margin-top: .5rem; overflow: hidden;}
.bar > div {height: 100%; border-radius: 99px;}
.flag.known {background: #fef2f2; border-left-color: #dc2626;}
.flag .why {color: #4b5d5b; font-size: .85rem;}
.flag {background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 8px; padding: .5rem .8rem; margin: .35rem 0;
       font-size: .92rem; color: #14201f;}
.stButton > button, .stFormSubmitButton > button {border-radius: 10px; font-weight: 600;}
.tile {background: #fff; border: 1px solid #e3e9e7; border-radius: 16px; padding: 1rem 1.2rem; height: 100%;}
.tile .n {font-size: 2.1rem; font-weight: 800; color: #0f766e; line-height: 1.1;}
.tile .t {color: #4b5d5b; font-size: .88rem; margin-top: .2rem;}
.tile.red .n {color: #dc2626;}
.hero.big {padding: 2.4rem 2.2rem;} .hero.big h1 {font-size: 2.6rem;} .hero.big p {font-size: 1.1rem; max-width: 640px;}
.foot {color: #6b7c7a; font-size: .8rem; margin-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/check.py", title="Check a job ad", icon="🔍"),
    st.Page("views/data.py", title="The data", icon="📊"),
    st.Page("views/model.py", title="How we built it", icon="🧠"),
    st.Page("views/findings.py", title="Findings", icon="💡"),
]
st.navigation(pages).run()
st.markdown('<div class="foot">JobGuard · ICS3202 AI project · EMSCAD dataset, University of the Aegean. '
            'Trained on 2012–2014, mostly-US ads: treat scores as a warning, not proof.</div>', unsafe_allow_html=True)
