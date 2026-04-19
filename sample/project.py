import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Wheat Disease Detector", layout="wide", page_icon="🌾")

# Custom CSS for UI improvement and visibility
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #2e7d32; color: white; font-weight: bold;
    }
    .expert-card {
        padding: 15px; background-color: white; border-radius: 10px;
        border-left: 5px solid #2e7d32; box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        color: black !important; /* Ensures text is always black for visibility */
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. EXPERT KNOWLEDGE DATABASE ---
DISEASE_GUIDE = {
    "Yellow Rust": {
        "danger": "Critical",
        "treatment": "Apply fungicides containing Tebuconazole immediately.",
        "prevention": "Use rust-resistant cultivars and monitor nitrogen levels.",
        "faqs": {"Spread?": "Wind-borne spores.", "Climate?": "10-15°C with high humidity."}
    },
    "Brown Rust": {
        "danger": "High",
        "treatment": "Triazole or Strobilurin based fungicides are effective.",
        "prevention": "Eliminate 'green bridges' (volunteer wheat) before the season.",
        "faqs": {"Yield loss?": "Up to 30%.", "Synonym?": "Also known as Leaf Rust."}
    },
    "Stem Rust": {
        "danger": "Severe",
        "treatment": "Systemic fungicides required. Isolate infected zones.",
        "prevention": "Eradicate alternate hosts (barberry bushes) near fields.",
        "faqs": {"Visuals?": "Brick-red elongated pustules.", "Risk?": "Early infection can lead to total loss."}
    }
}


# --- 3. CORE LOGIC (AI & EMAIL) ---
@st.cache_resource
def load_trained_model():
    try:
        return YOLO("sample/best.pt")
    except Exception as e:
        st.error(f"Model Error: Place 'best.pt' in the 'sample' folder. Error: {e}")
        return None


def send_forwarded_email(user_name, user_email, user_msg, subject):
    """Sends diagnostic reports directly to Rishendra and the team."""
    ADMIN_RECEIVER = "rishendra009@gmail.com"
    SENDER_EMAIL = "rishendra009@gmail.com"
    APP_PASSWORD = "krnmaovfbaajlqwy"

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = ADMIN_RECEIVER
    msg['Subject'] = f"WHEATSENSE REPORT: {subject} from {user_name}"

    body = f"""
    New Inquiry from WheatSense App:

    Sender Name: {user_name}
    Sender Email: {user_email}
    Subject: {subject}

    Message Content:
    {user_msg}

    ---
    Project Team: Rishendra, Vignesh, Siddhartha, Sanjit
    """
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, ADMIN_RECEIVER, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.sidebar.error(f"Mail failed: {e}")
        return False


# --- 4. NAVIGATION & SIDEBAR ---
with st.sidebar:
    st.title("🌾 WheatSense AI")
    page = st.radio("Navigation", ["Scanner", "Expert Assistant", "Contact Team"])
    st.divider()
    st.subheader("Project Team")
    st.markdown("""
    * **Rishendra**
    * **Vignesh**
    * **Siddhartha**
    * **Sanjit**
    """)
    # System Version box removed as requested

model = load_trained_model()

# --- 5. SCANNER PAGE ---
if page == "Scanner":
    st.title("🌾 Precision Diagnostic Scanner")
    col1, col2 = st.columns([1, 1])

    with col1:
        choice = st.toggle("Use Camera")
        source = st.camera_input("Scan Leaf") if choice else st.file_uploader("Upload Leaf Image", type=['jpg', 'png'])

    if source and model:
        img = Image.open(source)
        with st.spinner('AI analyzing leaf patterns...'):
            results = model.predict(img, conf=0.25)
            res_plotted = results[0].plot()

        with col2:
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="AI Detection Result",
                     use_container_width=True)
            found = list(set([model.names[int(box.cls[0])] for box in results[0].boxes]))
            if found:
                st.error(f"Detected: {', '.join(found)}")
            else:
                st.success("Result: Healthy Crop Detected")

# --- 6. KNOWLEDGE BOT PAGE ---
elif page == "Expert Assistant":
    st.title("🤖 Expert Knowledge Bot")
    st.subheader("General FAQs")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("How accurate is the AI?"):
            # Fixed text color to black for visibility
            st.markdown(
                '<div class="expert-card"><b>Answer:</b> The YOLOv11 engine is optimized for high precision, but should always be paired with field verification.</div>',
                unsafe_allow_html=True)
    with c2:
        if st.button("Supported Varieties?"):
            # Fixed text color to black for visibility
            st.markdown(
                '<div class="expert-card"><b>Answer:</b> We currently support most common varieties of Spring, Winter, and Durum wheat.</div>',
                unsafe_allow_html=True)

    st.divider()
    selected = st.selectbox("Disease Encyclopedia", ["Choose..."] + list(DISEASE_GUIDE.keys()))
    if selected != "Choose...":
        data = DISEASE_GUIDE[selected]
        st.metric("Threat Level", data["danger"])
        st.warning(f"**Expert Recommended Treatment:** {data['treatment']}")
        with st.expander("Biological Details & FAQ"):
            for q, a in data["faqs"].items():
                st.write(f"**{q}**: {a}")

# --- 7. CONTACT PAGE ---
elif page == "Contact Team":
    st.title("📩 Contact the Expert Panel")
    st.write("Forward messages directly to Rishendra and the team for verification.")

    with st.form("mail_form", clear_on_submit=True):
        u_name = st.text_input("Full Name")
        u_email = st.text_input("Email Address")
        u_subj = st.selectbox("Topic", ["Diagnosis Verification", "Technical Support", "General Feedback"])
        u_msg = st.text_area("Detailed Observations")

        if st.form_submit_button("Send Email to Team"):
            if u_name and u_email and u_msg:
                with st.spinner("Connecting to mail server..."):
                    if send_forwarded_email(u_name, u_email, u_msg, u_subj):
                        st.success("Message sent successfully! The team will review your report.")
                        st.balloons()
            else:
                st.error("Please fill in all fields before sending.")
