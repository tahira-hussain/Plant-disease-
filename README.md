# Plant-disease-
it detect the disease in beans plant
# 🌿 Bean Plant Disease Detector

Ye project **Transfer Learning** ki madad se bean ke patton ki bimari pehchanta hai. User patte ki image upload karta hai aur model batata hai ke patta healthy hai ya kisi bimari ka shikar hai.

## Problem Statement
Kisanon ke liye patton ki bimari ko jaldi pehchanna zaroori hai taake fasal bachai ja sake. Is project mein ek deep learning model banaya gaya hai jo bean ke patte ki tasveer se uski bimari classify karta hai.

## Classes
- `healthy`
- `angular_leaf_spot`
- `bean_rust`

## Dataset
- **Naam:** Beans Dataset (Makerere AI Lab)
- **Link:** https://huggingface.co/datasets/AI-Lab-Makerere/beans
- **Split:** train, validation aur test

## Model
- **Pretrained model:** [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) (Vision Transformer)
- **Fine-tuned model:** [TahiraKhan01/vit-beans-plant-disease](https://huggingface.co/TahiraKhan01/vit-beans-plant-disease)

## Project ke Phases
1. **Dataset preparation:** dataset download, explore, clean aur train/validation/test mein organize kiya.
2. **Fine-tuning:** Hugging Face ke pretrained ViT model ko Beans dataset par 4 epochs tak fine-tune kiya.
3. **Model hosting:** fine-tuned model ko Hugging Face Hub par upload kiya.
4. **GitHub:** application code is repository mein rakha gaya hai.
5. **Deployment:** Streamlit app banayi aur Streamlit Community Cloud par deploy ki.

## Files
- `app.py`: Streamlit application
- `requirements.txt`: zaroori libraries
- `README.md`: project ki tafseel

## Locally chalane ka tareeqa
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live App
👉 (yahan apna Streamlit app ka link likho)

## Technologies
Python, PyTorch, Hugging Face Transformers, Streamlit
