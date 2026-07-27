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

function renderComparisonResults(data) {
    const placeholder = document.getElementById("placeholder-state");
    const resultsArea = document.getElementById("compare-results-content");
    if (placeholder) placeholder.style.display = "none";
    if (resultsArea) resultsArea.style.display = "flex";

    const base = data.basemodel;
    const my = data.mymodel;

    // 1. Populate Base Model Column
    populateModelColumn("base", base);

    // 2. Populate My Model Column
    populateModelColumn("my", my);

    resultsArea.scrollIntoView({ behavior: "smooth", block: "start" });
}

function populateModelColumn(prefix, modData) {
    const banner = document.getElementById(`diag-banner-${prefix}`);
    const diagIcon = document.getElementById(`diag-icon-${prefix}`);
    const diagText = document.getElementById(`diag-text-${prefix}`);
    const probTumor = document.getElementById(`val-tumor-${prefix}`);
    const probNormal = document.getElementById(`val-normal-${prefix}`);
    const regCount = document.getElementById(`val-regions-${prefix}`);

    if (banner) {
        banner.className = `diag-banner ${modData.prediction.is_abnormal ? 'status-tumor' : 'status-normal'}`;
    }
    if (diagIcon) {
        diagIcon.innerHTML = modData.prediction.is_abnormal ? '⚠️' : '✅';
    }
    if (diagText) {
        diagText.textContent = modData.prediction.prediction;
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
