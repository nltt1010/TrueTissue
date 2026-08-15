import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Tái sử dụng lõi phân tích AI từ project hiện tại
from app.services.tissue_analyzer import TissueAnalyzer

st.set_page_config(page_title="TrueTissue AI - Deploy Ready", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource(show_spinner="Đang nạp AI Models vào bộ nhớ đệm (Cache)...")
def get_analyzer():
    """
    Sử dụng cache_resource để đảm bảo mô hình Pytorch nặng chỉ được khởi tạo 1 lần 
    và giữ trong RAM, giúp app web chạy siêu nhanh khi người dùng thao tác.
    """
    return TissueAnalyzer(root_dir=Path(__file__).parent)

analyzer = get_analyzer()
checkpoints = analyzer.get_available_checkpoints()
samples = analyzer.get_demo_samples()

# ----------------- SIDEBAR -----------------
st.sidebar.title("🔬 TrueTissue Workspace")
mode = st.sidebar.radio("Chế độ hoạt động:", ["Phân Tích Đơn Lẻ", "So Sánh Mô Hình (Multi-Model)"])

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Dữ Liệu Đầu Vào")

input_option = st.sidebar.radio("Nguồn ảnh:", ["Tải ảnh lên từ máy", "Chọn ảnh Demo có sẵn"])

image_path_to_process = None

# Thư mục tạm chứa ảnh upload khi chạy trên Cloud
temp_dir = Path(tempfile.gettempdir()) / "truetissue_st_uploads"
temp_dir.mkdir(exist_ok=True, parents=True)

if input_option == "Tải ảnh lên từ máy":
    uploaded_file = st.sidebar.file_uploader("Upload Ảnh Mô Tế Bào (PNG, JPG, TIF)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if uploaded_file is not None:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_path_to_process = str(file_path)
        st.sidebar.image(image_path_to_process, caption="Ảnh bạn vừa tải lên", use_column_width=True)
else:
    sample_options = {s["name"]: s["path"] for s in samples}
    if sample_options:
        selected_sample = st.sidebar.selectbox("Chọn ảnh:", list(sample_options.keys()))
        image_path_to_process = sample_options[selected_sample]
        st.sidebar.image(image_path_to_process, caption=selected_sample, use_column_width=True)
    else:
        st.sidebar.warning("Không tìm thấy ảnh Demo trong thư mục dataset/test.")

st.sidebar.markdown("---")
st.sidebar.markdown("*Sẵn sàng triển khai lên HuggingFace Spaces / Streamlit Cloud.*")


# ----------------- MAIN CONTENT -----------------

def get_abs_img_path(relative_path_from_analyzer):
    """Hàm chuyển đường dẫn trả về từ analyzer (e.g. /static/outputs/...) thành đường dẫn tuyệt đối để st đọc"""
    return str(Path(__file__).parent / "app" / relative_path_from_analyzer.lstrip('/'))


if mode == "Phân Tích Đơn Lẻ":
    st.title("🧫 Không Gian Phân Tích Đơn Lẻ")
    st.markdown("Chạy toàn bộ quá trình Nhuộm ảo H&E và Phân loại Bất thường tự động.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox("Chọn Kiến trúc Nhuộm:", ["mymodel", "basemodel"], format_func=lambda x: "My Model (Khuyên Dùng)" if x == "mymodel" else "Base Model")
    with col2:
        if model_type == "mymodel":
            stain_ckpt = st.selectbox("Checkpoint Nhuộm:", checkpoints["mymodel"][::-1] if checkpoints["mymodel"] else ["None"])
        else:
            stain_ckpt = st.selectbox("Checkpoint Nhuộm:", checkpoints["basemodel"][::-1] if checkpoints["basemodel"] else ["None"])
    with col3:
        cls_ckpt = st.selectbox("Checkpoint Phân Loại:", checkpoints["classifier"][::-1] if checkpoints["classifier"] else ["None"])
    
    if st.button("🚀 Chạy Pipeline Phân Tích", type="primary", use_container_width=True):
        if not image_path_to_process:
            st.error("Vui lòng tải ảnh lên hoặc chọn ảnh mẫu trước.")
        else:
            with st.spinner(f"AI đang tiến hành phân tích bằng {model_type}..."):
                results = analyzer.run_sliced_analysis(
                    img_input=image_path_to_process,
                    model_type=model_type,
                    stain_ckpt=stain_ckpt if stain_ckpt != "None" else None,
                    cls_ckpt=cls_ckpt if cls_ckpt != "None" else None
                )
                
            st.success("✅ Phân tích hoàn tất!")
            
            # Xử lý hiển thị Grid Slicing nếu ảnh lớn bị cắt nhỏ
            if len(results) > 1:
                st.subheader("🧩 Lưới Tế Bào (Ảnh được cắt nhỏ để xử lý vi mô)")
                total_cols = results[0]["patch_info"]["total_cols"]
                
                for i in range(0, len(results), total_cols):
                    cols = st.columns(total_cols)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(results):
                            slice_data = results[idx]
                            pred = slice_data['prediction']
                            
                            with col:
                                # Ảnh thumb
                                img_path = get_abs_img_path(slice_data["images"]["grayscale"])
                                st.image(img_path, caption=f'Mảnh [{slice_data["patch_info"]["row"]}, {slice_data["patch_info"]["col"]}]')
                                
                                # Highlight nếu có bất thường
                                if pred["is_abnormal"]:
                                    st.error("Phát hiện Tumor!")
                                
                                # Modal expander để xem chi tiết
                                with st.expander("Xem Chi Tiết"):
                                    st.markdown(f"**Chẩn đoán:** {pred['prediction']}")
                                    st.progress(pred['prob_tumor'] / 100, text=f"Tỉ lệ Tumor: {pred['prob_tumor']}%")
                                    
                                    st.image(get_abs_img_path(slice_data["images"]["upscaled_stained"]), caption="Ảnh Nhuộm")
                                    st.image(get_abs_img_path(slice_data["images"]["upscaled_cam"]), caption="Heatmap Bất Thường")
                                    
            # Xử lý hiển thị nguyên bản nếu ảnh nhỏ
            else:
                slice_data = results[0]
                pred = slice_data["prediction"]
                
                st.subheader(f"📊 Kết Quả Chẩn Đoán: {pred['prediction']}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Tỉ lệ Normal", f"{pred['prob_normal']}%")
                m2.metric("Tỉ lệ Tumor", f"{pred['prob_tumor']}%")
                m3.metric("Số ổ Tumor khoanh vùng", pred['num_regions_detected'])
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(get_abs_img_path(slice_data["images"]["grayscale"]), caption="Ảnh Gốc Bệnh Học", use_column_width=True)
                with c2:
                    st.image(get_abs_img_path(slice_data["images"]["upscaled_stained"]), caption="Nhuộm Ảo H&E", use_column_width=True)
                with c3:
                    st.image(get_abs_img_path(slice_data["images"]["upscaled_cam"]), caption="AI Khoanh Vùng Bất Thường", use_column_width=True)

elif mode == "So Sánh Mô Hình (Multi-Model)":
    st.title("⚖️ Chế Độ So Sánh Đa Mô Hình")
    st.markdown("Tiến hành chạy song song Base Model và My Model để đối chiếu chất lượng màu nhuộm và độ nhiễu.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        stain_ckpt_base = st.selectbox("Base Model Checkpoint:", checkpoints["basemodel"][::-1] if checkpoints["basemodel"] else ["None"])
    with col2:
        stain_ckpt_my = st.selectbox("My Model Checkpoint:", checkpoints["mymodel"][::-1] if checkpoints["mymodel"] else ["None"])
    with col3:
        cls_ckpt = st.selectbox("Classifier Checkpoint:", checkpoints["classifier"][::-1] if checkpoints["classifier"] else ["None"])
        
    if st.button("🚀 Bắt Đầu So Sánh", type="primary", use_container_width=True):
        if not image_path_to_process:
            st.error("Vui lòng tải ảnh lên hoặc chọn ảnh mẫu trước.")
        else:
            with st.spinner("Đang ép xung chạy song song 2 Pipeline AI..."):
                results = analyzer.run_sliced_comparison(
                    img_input=image_path_to_process,
                    stain_ckpt_base=stain_ckpt_base if stain_ckpt_base != "None" else None,
                    stain_ckpt_my=stain_ckpt_my if stain_ckpt_my != "None" else None,
                    cls_ckpt=cls_ckpt if cls_ckpt != "None" else None
                )
                
            st.success("✅ Hoàn tất quá trình so sánh!")
            
            # Xử lý Slicing Grid cho chế độ Compare
            if len(results) > 1:
                st.subheader("🧩 Lưới Tế Bào Slicing (Compare)")
                total_cols = results[0]["patch_info"]["total_cols"]
                
                for i in range(0, len(results), total_cols):
                    cols = st.columns(total_cols)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(results):
                            slice_data = results[idx]
                            
                            with col:
                                img_path = get_abs_img_path(slice_data["basemodel"]["images"]["grayscale"])
                                st.image(img_path, caption=f'Mảnh [{slice_data["patch_info"]["row"]}, {slice_data["patch_info"]["col"]}]')
                                
                                with st.expander("So sánh ngang"):
                                    st.markdown("### Base Model")
                                    st.write(slice_data['basemodel']['prediction']['prediction'])
                                    st.image(get_abs_img_path(slice_data["basemodel"]["images"]["upscaled_cam"]))
                                    
                                    st.markdown("### My Model")
                                    st.write(slice_data['mymodel']['prediction']['prediction'])
                                    st.image(get_abs_img_path(slice_data["mymodel"]["images"]["upscaled_cam"]))
            
            # Xử lý ảnh nguyên khối cho chế độ Compare
            else:
                comp_data = results[0]
                col_b, col_m = st.columns(2)
                
                with col_b:
                    st.header("🔵 Base Model")
                    pred_b = comp_data["basemodel"]["prediction"]
                    st.info(f"**Chẩn đoán:** {pred_b['prediction']}")
                    st.write(f"Độ tin cậy Normal: {pred_b['prob_normal']}% | Tumor: {pred_b['prob_tumor']}%")
                    st.image(get_abs_img_path(comp_data["basemodel"]["images"]["upscaled_stained"]), caption="Ảnh Nhuộm (Base)")
                    st.image(get_abs_img_path(comp_data["basemodel"]["images"]["upscaled_cam"]), caption="Heatmap (Base)")
                    
                with col_m:
                    st.header("🟢 My Model (Cải Tiến)")
                    pred_m = comp_data["mymodel"]["prediction"]
                    st.success(f"**Chẩn đoán:** {pred_m['prediction']}")
                    st.write(f"Độ tin cậy Normal: {pred_m['prob_normal']}% | Tumor: {pred_m['prob_tumor']}%")
                    st.image(get_abs_img_path(comp_data["mymodel"]["images"]["upscaled_stained"]), caption="Ảnh Nhuộm (My Model)")
                    st.image(get_abs_img_path(comp_data["mymodel"]["images"]["upscaled_cam"]), caption="Heatmap (My Model)")
