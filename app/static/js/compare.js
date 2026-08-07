/**
 * TrueTissue Clinical AI Web Application - Comparison Suite Logic
 * Handles simultaneous execution of Base Model and My Model and large direct side-by-side display.
 */

document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("compare-form")) {
        initCompareWorkspace();
    }
});

function initCompareWorkspace() {
    const form = document.getElementById("compare-form");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!selectedFile) {
            alert("Please upload a tissue sample image first.");
            return;
        }

        const formData = new FormData(form);
        formData.set("file", selectedFile);
        formData.delete("sample_path");

        showLoader("Running Multi-Model Comparison Suite...", "Simultaneously executing Base Model & My Model virtual staining and abnormality localization with 4x ultra-sharp enhancement...");
        
        try {
            const response = await fetch("/api/compare", {
                method: "POST",
                body: formData
            });
            const resData = await response.json();
            
            if (resData.status === "success") {
                renderComparisonResults(resData.data);
            } else {
                alert(`Comparison Error: ${resData.message}`);
            }
        } catch (err) {
            console.error("API Error:", err);
            alert("An unexpected error occurred while communicating with the AI server.");
        } finally {
            hideLoader();
        }
    });
}

function renderComparisonResults(dataArray) {
    if (!Array.isArray(dataArray) || dataArray.length === 0) return;

    const placeholder = document.getElementById("placeholder-state");
    const resultsArea = document.getElementById("compare-results-content");
    if (placeholder) placeholder.style.display = "none";
    if (resultsArea) resultsArea.style.display = "flex";

    const galleryContainer = document.getElementById("compare-slice-gallery-container");
    const gallery = document.getElementById("compare-slice-gallery");
    
    if (dataArray.length > 1) {
        galleryContainer.style.display = "block";
        gallery.innerHTML = "";
        
        // Format gallery as a real 2D grid matching the sliced image
        const totalCols = dataArray[0].patch_info.total_cols;
        gallery.style.display = "grid";
        gallery.style.gridTemplateColumns = `repeat(${totalCols}, max-content)`;
        gallery.style.gap = "4px";
        gallery.style.justifyContent = "center";
        gallery.style.margin = "0 auto";
        
        dataArray.forEach((slice, idx) => {
            const thumb = document.createElement("img");
            // Use basemodel grayscale as thumbnail
            thumb.src = slice.basemodel.images.grayscale;
            thumb.style.width = "80px";
            thumb.style.height = "80px";
            thumb.style.objectFit = "cover";
            thumb.style.borderRadius = "4px";
            thumb.style.cursor = "pointer";
            thumb.style.border = idx === 0 ? "3px solid var(--accent-cyan)" : "3px solid transparent";
            
            // Highlight if either model predicts abnormal
            const isAbnormal = slice.basemodel.prediction.is_abnormal || slice.mymodel.prediction.is_abnormal;
            if (isAbnormal) {
                thumb.style.boxShadow = "0 0 10px rgba(255, 60, 60, 0.9)";
            }

            thumb.title = `Patch [Row ${slice.patch_info.row}, Col ${slice.patch_info.col}]`;

            thumb.addEventListener("click", () => {
                Array.from(gallery.children).forEach(c => c.style.border = "3px solid transparent");
                thumb.style.border = "3px solid var(--accent-cyan)";
                renderViewForComparisonSlice(slice);
            });

            gallery.appendChild(thumb);
        });
    } else {
        galleryContainer.style.display = "none";
    }

    renderViewForComparisonSlice(dataArray[0]);

    resultsArea.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderViewForComparisonSlice(data) {
    const base = data.basemodel;
    const my = data.mymodel;
    
    populateModelColumn("base", base, data.patch_info);
    populateModelColumn("my", my, data.patch_info);
}

function populateModelColumn(prefix, modData, patch_info) {
    const banner = document.getElementById(`diag-banner-${prefix}`);
    const diagIcon = document.getElementById(`diag-icon-${prefix}`);
    const diagText = document.getElementById(`diag-text-${prefix}`);
    const probTumor = document.getElementById(`val-tumor-${prefix}`);
    const probNormal = document.getElementById(`val-normal-${prefix}`);
    const regCount = document.getElementById(`val-regions-${prefix}`);

    let locationText = patch_info && patch_info.is_slice ? ` (Patch ${patch_info.row}, ${patch_info.col})` : '';

    if (banner) {
        banner.className = `diag-banner ${modData.prediction.is_abnormal ? 'status-tumor' : 'status-normal'}`;
    }
    if (diagIcon) {
        diagIcon.innerHTML = modData.prediction.is_abnormal ? '⚠️' : '✅';
    }
    if (diagText) {
        diagText.textContent = modData.prediction.prediction + locationText;
    }
    if (probTumor) probTumor.textContent = `${modData.prediction.prob_tumor}%`;
    if (probNormal) probNormal.textContent = `${modData.prediction.prob_normal}%`;
    if (regCount) regCount.textContent = modData.prediction.num_regions_detected;

    // Direct Display of Large 4x Ultra-Sharp Images
    const imgStainLarge = document.getElementById(`img-stain-${prefix}-large`);
    const imgCamLarge = document.getElementById(`img-cam-${prefix}-large`);
    const badgeCkpt = document.getElementById(`badge-ckpt-${prefix}`);

    if (imgStainLarge) imgStainLarge.src = modData.images.upscaled_stained;
    if (imgCamLarge) imgCamLarge.src = modData.images.upscaled_cam;
    if (badgeCkpt) badgeCkpt.textContent = modData.stain_checkpoint;
}
