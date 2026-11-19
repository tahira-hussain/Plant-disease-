import streamlit as st
from PIL import Image
from transformers import pipeline

MODEL_ID = "TahiraKhan01/vit-beans-plant-disease"

st.set_page_config(page_title="Bean Plant Disease Detector", page_icon="🌿")

CLASS_INFO = {
    "healthy": "Patta healthy hai. Koi bimari nazar nahi aayi.",
    "angular_leaf_spot": "Angular leaf spot bimari ke nishan hain. Pattay par kone daar dhabbe hote hain.",
    "bean_rust": "Bean rust bimari ke nishan hain. Pattay par zang jaisay bhoore dhabbe hote hain.",
}


@st.cache_resource(show_spinner="Model load ho raha hai...")
def load_model():
    return pipeline("image-classification", model=MODEL_ID)


st.title("🌿 Bean Plant Disease Detector")
st.write(
    "Bean ke patte ki tasveer upload karein. Model batayega ke patta healthy hai "
    "ya kisi bimari (angular leaf spot / bean rust) ka shikar hai."
)

classifier = load_model()

uploaded = st.file_uploader("Patte ki image chunein", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Upload ki hui image", use_container_width=True)

    with st.spinner("Prediction ho rahi hai..."):
        results = classifier(image, top_k=3)

    top = results[0]
    st.subheader(f"Natija: {top['label']}")
    st.metric("Confidence", f"{top['score'] * 100:.1f}%")
    st.info(CLASS_INFO.get(top["label"], ""))

    st.write("Tamam classes ke scores:")
    st.bar_chart({r["label"]: r["score"] for r in results})
else:
    st.caption("Abhi koi image upload nahi hui.")

st.markdown("---")
st.caption(f"Model: {MODEL_ID} | Fine-tuned ViT on Beans dataset")
