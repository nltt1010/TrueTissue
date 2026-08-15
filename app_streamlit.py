import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Core AI Analysis
from app.services.tissue_analyzer import TissueAnalyzer

st.set_page_config(page_title="TrueTissue AI - Deploy Ready", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource(show_spinner="Loading AI Models into Cache...")
def get_analyzer():
    return TissueAnalyzer(root_dir=Path(__file__).parent)

analyzer = get_analyzer()
checkpoints = analyzer.get_available_checkpoints()

def get_demo_samples():
    demo_dir = Path(__file__).parent / "demo"
    samples = []
    if demo_dir.exists():
        for img_path in demo_dir.glob("*.*"):
            if img_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
                name = img_path.stem.replace("_", " ").title()
                samples.append({"name": name, "path": str(img_path)})
    return samples

samples = get_demo_samples()

def get_default_index(options, preferred_name):
    for i, opt in enumerate(options):
        if preferred_name in opt:
            return i
    return 0

# ----------------- SIDEBAR -----------------
st.sidebar.title("🔬 TrueTissue Workspace")
mode = st.sidebar.radio("Workspace Mode:", ["Single Analysis", "Multi-Model Comparison"])

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Input Data")

input_option = st.sidebar.radio("Source:", ["Upload from Device", "Select Demo Image"])

image_path_to_process = None

temp_dir = Path(tempfile.gettempdir()) / "truetissue_st_uploads"
temp_dir.mkdir(exist_ok=True, parents=True)

if input_option == "Upload from Device":
    uploaded_file = st.sidebar.file_uploader("Upload Tissue Image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if uploaded_file is not None:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_path_to_process = str(file_path)
        st.sidebar.image(image_path_to_process, caption="Uploaded Image", use_column_width=True)
else:
    sample_options = {s["name"]: s["path"] for s in samples}
    if sample_options:
        selected_sample = st.sidebar.selectbox("Choose Image:", list(sample_options.keys()))
        image_path_to_process = sample_options[selected_sample]
        st.sidebar.image(image_path_to_process, caption=selected_sample, use_column_width=True)
    else:
        st.sidebar.warning("No demo images found in demo directory.")

st.sidebar.markdown("---")
st.sidebar.markdown("*Ready for HuggingFace Spaces / Streamlit Cloud deployment.*")


# ----------------- MAIN CONTENT -----------------

def get_abs_img_path(relative_path_from_analyzer):
    return str(Path(__file__).parent / "app" / relative_path_from_analyzer.lstrip('/'))

# Prepare reversed lists for dropdowns
my_ckpts = checkpoints["mymodel"][::-1] if checkpoints["mymodel"] else ["None"]
base_ckpts = checkpoints["basemodel"][::-1] if checkpoints["basemodel"] else ["None"]
cls_ckpts = checkpoints["classifier"][::-1] if checkpoints["classifier"] else ["None"]

# Find best defaults
my_default = get_default_index(my_ckpts, "gen_8")
base_default = get_default_index(base_ckpts, "gen_5")
cls_default = get_default_index(cls_ckpts, "cls_best")


if mode == "Single Analysis":
    st.title("🧫 Single Analysis Workspace")
    st.markdown("Run full Virtual H&E Staining and Abnormality Classification.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox("Staining Architecture:", ["mymodel", "basemodel"], format_func=lambda x: "My Model" if x == "mymodel" else "Base Model")
    with col2:
        if model_type == "mymodel":
            stain_ckpt = st.selectbox("Staining Checkpoint:", my_ckpts, index=my_default)
        else:
            stain_ckpt = st.selectbox("Staining Checkpoint:", base_ckpts, index=base_default)
    with col3:
        cls_ckpt = st.selectbox("Classification Checkpoint:", cls_ckpts, index=cls_default)
    
    if st.button("🚀 Run Analysis Pipeline", type="primary", use_container_width=True):
        if not image_path_to_process:
            st.error("Please upload an image or select a demo sample first.")
        else:
            with st.spinner(f"AI is analyzing using {model_type}..."):
                results = analyzer.run_sliced_analysis(
                    img_input=image_path_to_process,
                    model_type=model_type,
                    stain_ckpt=stain_ckpt if stain_ckpt != "None" else None,
                    cls_ckpt=cls_ckpt if cls_ckpt != "None" else None
                )
                
            st.success("✅ Analysis Complete!")
            
            if len(results) > 1:
                st.subheader("🧩 Tissue Grid")
                total_cols = results[0]["patch_info"]["total_cols"]
                
                for i in range(0, len(results), total_cols):
                    cols = st.columns(total_cols)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(results):
                            slice_data = results[idx]
                            pred = slice_data['prediction']
                            
                            with col:
                                img_path = get_abs_img_path(slice_data["images"]["grayscale"])
                                st.image(img_path, caption=f'Patch [{slice_data["patch_info"]["row"]}, {slice_data["patch_info"]["col"]}]')
                                
                                if pred["is_abnormal"]:
                                    st.error("Tumor Detected!")
                                
                                with st.expander("View Details"):
                                    st.markdown(f"**Diagnosis:** {pred['prediction']}")
                                    st.progress(pred['prob_tumor'] / 100, text=f"Tumor Probability: {pred['prob_tumor']}%")
                                    
                                    st.image(get_abs_img_path(slice_data["images"]["upscaled_stained"]), caption="Virtual Staining")
                                    st.image(get_abs_img_path(slice_data["images"]["upscaled_cam"]), caption="Abnormality Heatmap")
                                    
            else:
                slice_data = results[0]
                pred = slice_data["prediction"]
                
                st.subheader(f"📊 Diagnosis: {pred['prediction']}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Normal Ratio", f"{pred['prob_normal']}%")
                m2.metric("Tumor Ratio", f"{pred['prob_tumor']}%")
                m3.metric("Detected Tumor Regions", pred['num_regions_detected'])
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(get_abs_img_path(slice_data["images"]["grayscale"]), caption="Original Tissue", use_column_width=True)
                with c2:
                    st.image(get_abs_img_path(slice_data["images"]["upscaled_stained"]), caption="Virtual H&E Staining", use_column_width=True)
                with c3:
                    st.image(get_abs_img_path(slice_data["images"]["upscaled_cam"]), caption="AI Abnormality Bounding", use_column_width=True)

elif mode == "Multi-Model Comparison":
    st.title("⚖️ Multi-Model Comparison Suite")
    st.markdown("Run Base Model and My Model simultaneously to compare staining quality and noise.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        stain_ckpt_base = st.selectbox("Base Model Checkpoint:", base_ckpts, index=base_default)
    with col2:
        stain_ckpt_my = st.selectbox("My Model Checkpoint:", my_ckpts, index=my_default)
    with col3:
        cls_ckpt = st.selectbox("Classifier Checkpoint:", cls_ckpts, index=cls_default)
        
    if st.button("🚀 Start Comparison", type="primary", use_container_width=True):
        if not image_path_to_process:
            st.error("Please upload an image or select a demo sample first.")
        else:
            with st.spinner("Running parallel AI pipelines..."):
                results = analyzer.run_sliced_comparison(
                    img_input=image_path_to_process,
                    stain_ckpt_base=stain_ckpt_base if stain_ckpt_base != "None" else None,
                    stain_ckpt_my=stain_ckpt_my if stain_ckpt_my != "None" else None,
                    cls_ckpt=cls_ckpt if cls_ckpt != "None" else None
                )
                
            st.success("✅ Comparison Complete!")
            
            if len(results) > 1:
                st.subheader("🧩 Slicing Grid")
                total_cols = results[0]["patch_info"]["total_cols"]
                
                for i in range(0, len(results), total_cols):
                    cols = st.columns(total_cols)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(results):
                            slice_data = results[idx]
                            
                            with col:
                                img_path = get_abs_img_path(slice_data["basemodel"]["images"]["grayscale"])
                                st.image(img_path, caption=f'Patch [{slice_data["patch_info"]["row"]}, {slice_data["patch_info"]["col"]}]')
                                
                                with st.expander("Side-by-side Compare"):
                                    st.markdown("### Base Model")
                                    st.write(slice_data['basemodel']['prediction']['prediction'])
                                    st.image(get_abs_img_path(slice_data["basemodel"]["images"]["upscaled_cam"]))
                                    
                                    st.markdown("### My Model")
                                    st.write(slice_data['mymodel']['prediction']['prediction'])
                                    st.image(get_abs_img_path(slice_data["mymodel"]["images"]["upscaled_cam"]))
            
            else:
                comp_data = results[0]
                col_b, col_m = st.columns(2)
                
                with col_b:
                    st.header("🔵 Base Model")
                    pred_b = comp_data["basemodel"]["prediction"]
                    st.info(f"**Diagnosis:** {pred_b['prediction']}")
                    st.write(f"Normal: {pred_b['prob_normal']}% | Tumor: {pred_b['prob_tumor']}%")
                    st.image(get_abs_img_path(comp_data["basemodel"]["images"]["upscaled_stained"]), caption="Stained Image")
                    st.image(get_abs_img_path(comp_data["basemodel"]["images"]["upscaled_cam"]), caption="Heatmap")
                    
                with col_m:
                    st.header("🟢 My Model")
                    pred_m = comp_data["mymodel"]["prediction"]
                    st.success(f"**Diagnosis:** {pred_m['prediction']}")
                    st.write(f"Normal: {pred_m['prob_normal']}% | Tumor: {pred_m['prob_tumor']}%")
                    st.image(get_abs_img_path(comp_data["mymodel"]["images"]["upscaled_stained"]), caption="Stained Image")
                    st.image(get_abs_img_path(comp_data["mymodel"]["images"]["upscaled_cam"]), caption="Heatmap")
