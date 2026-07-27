/**
 * TrueTissue Clinical AI Web Application - Main Workspace Logic
 * Handles file drag-and-drop, single-model AI analysis, and large direct display of 4x ultra-sharp images.
 */

document.addEventListener("DOMContentLoaded", () => {
    initSharedControls();
    if (document.getElementById("analyze-form")) {
        initAnalyzeWorkspace();
    }
    initLightbox();
});

let selectedFile = null;

/**
 * Initializes shared controls (file upload dropzone and checkpoint loader).
 */
function initSharedControls() {
    // Fetch checkpoints for dropdowns
    fetch("/api/checkpoints")
        .then(res => res.json())
        .then(data => {
            if (data.status === "success") {
                populateCheckpoints("stain_ckpt", data.data.mymodel);
                populateCheckpoints("stain_ckpt_base", data.data.basemodel);
                populateCheckpoints("stain_ckpt_my", data.data.mymodel);
                populateCheckpoints("cls_ckpt", data.data.classifier);
            }
        })
        .catch(err => console.error("Failed to load checkpoints:", err));

    // Setup Drag and Drop Zone
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const filePreview = document.getElementById("file-preview");
    const removeBtn = document.getElementById("remove-file");

    if (dropzone && fileInput) {
        dropzone.addEventListener("click", () => fileInput.click());

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.classList.remove("dragover");
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        if (removeBtn) {
            removeBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                clearSelectedFile();
            });
        }
    }
}

function handleFileSelect(file) {
    if (!file.type.match("image.*") && !file.name.match(/\.(tif|tiff|bmp)$/i)) {
        alert("Please select a valid image file (PNG, JPG, TIF, BMP).");
        return;
    }
    selectedFile = file;
    const filePreview = document.getElementById("file-preview");
    const fileName = document.getElementById("file-name");
    const fileSize = document.getElementById("file-size");
    const fileThumb = document.getElementById("file-thumb");
    const submitBtn = document.getElementById("submit-btn");

    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

    // Preview thumbnail
    const reader = new FileReader();
    reader.onload = (e) => {
        if (fileThumb) fileThumb.src = e.target.result;
    };
    reader.readAsDataURL(file);

    if (filePreview) filePreview.style.display = "flex";
    if (submitBtn) submitBtn.disabled = false;
}

function clearSelectedFile() {
    selectedFile = null;
    const fileInput = document.getElementById("file-input");
    const filePreview = document.getElementById("file-preview");
    const submitBtn = document.getElementById("submit-btn");

    if (fileInput) fileInput.value = "";
    if (filePreview) filePreview.style.display = "none";
    if (submitBtn) submitBtn.disabled = true;
}

function populateCheckpoints(elementId, ckptList) {
    const select = document.getElementById(elementId);
    if (!select || !ckptList) return;
    
    const defaultText = select.getAttribute("data-default-text") || "Auto-Select Best Checkpoint";
    select.innerHTML = `<option value="">${defaultText}</option>`;
    
    ckptList.forEach(ckpt => {
        const opt = document.createElement("option");
        opt.value = ckpt;
        opt.textContent = ckpt;
        select.appendChild(opt);
    });
}

/**
 * Initializes Single Model Analysis Workspace
 */
function initAnalyzeWorkspace() {
    const form = document.getElementById("analyze-form");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!selectedFile) {
            alert("Please upload a tissue sample image first.");
            return;
        }

        const formData = new FormData(form);
        formData.set("file", selectedFile);
        formData.delete("sample_path");

        showLoader("Running Clinical AI Pipeline...", "Virtual Staining, Abnormality Localization, and 4x Razor-Sharp Enhancement in progress...");
        
        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });
            const resData = await response.json();
            
            if (resData.status === "success") {
                renderSingleResults(resData.data);
            } else {
                alert(`Analysis Error: ${resData.message}`);
            }
        } catch (err) {
            console.error("API Error:", err);
            alert("An unexpected error occurred while communicating with the AI server.");
        } finally {
            hideLoader();
        }
    });

    // Handle Model Radio Change to update Checkpoint Dropdown dynamically
    const radios = document.querySelectorAll('input[name="model_type"]');
    radios.forEach(radio => {
        radio.addEventListener("change", () => {
            fetch("/api/checkpoints")
                .then(res => res.json())
                .then(data => {
                    if (data.status === "success") {
                        const list = radio.value === "basemodel" ? data.data.basemodel : data.data.mymodel;
                        populateCheckpoints("stain_ckpt", list);
                    }
                });
        });
    });
}

/**
 * Renders Single Model Analysis Results directly into Dashboard UI in large full-size dimensions
 */
function renderSingleResults(data) {
    const placeholder = document.getElementById("placeholder-state");
    const resultsArea = document.getElementById("results-content");
    if (placeholder) placeholder.style.display = "none";
    if (resultsArea) resultsArea.style.display = "flex";

    // 1. Diagnostic Banner
    const banner = document.getElementById("diag-banner");
    const diagIcon = document.getElementById("diag-icon-el");
    const diagText = document.getElementById("diag-result-text");
    const probTumor = document.getElementById("val-tumor");
    const probNormal = document.getElementById("val-normal");
    const regCount = document.getElementById("val-regions");

    if (banner) {
        banner.className = `diag-banner ${data.prediction.is_abnormal ? 'status-tumor' : 'status-normal'}`;
    }
    if (diagIcon) {
        diagIcon.innerHTML = data.prediction.is_abnormal ? '⚠️' : '✅';
    }
    if (diagText) {
        diagText.textContent = data.prediction.prediction;
    }
    if (probTumor) probTumor.textContent = `${data.prediction.prob_tumor}%`;
    if (probNormal) probNormal.textContent = `${data.prediction.prob_normal}%`;
    if (regCount) regCount.textContent = data.prediction.num_regions_detected;

    // 2. Direct Display of 4x Ultra-Sharp Large Images
    const imgStainLarge = document.getElementById("img-stain-large");
    const imgCamLarge = document.getElementById("img-cam-large");
    const imgGray = document.getElementById("img-gray");
    const badgeModel = document.getElementById("badge-model-used");

    if (imgStainLarge) imgStainLarge.src = data.images.upscaled_stained;
    if (imgCamLarge) imgCamLarge.src = data.images.upscaled_cam;
    if (imgGray) imgGray.src = data.images.grayscale;
    if (badgeModel) badgeModel.textContent = `${data.model_used.toUpperCase()} (1024x1024)`;

    // Scroll smoothly to results
    resultsArea.scrollIntoView({ behavior: "smooth", block: "start" });
}

/**
 * Optional Lightbox modal logic for click-to-zoom if user wishes to inspect even closer
 */
function initLightbox() {
    const lightbox = document.getElementById("lightbox");
    const closeBtn = document.getElementById("lightbox-close");
    const img = document.getElementById("lightbox-img");

    if (!lightbox || !img) return;

    if (closeBtn) {
        closeBtn.addEventListener("click", () => {
            lightbox.classList.remove("active");
            img.classList.remove("zoomed");
        });
    }

    lightbox.addEventListener("click", (e) => {
        if (e.target === lightbox || e.target.classList.contains("lightbox-img-wrapper")) {
            lightbox.classList.remove("active");
            img.classList.remove("zoomed");
        }
    });

    img.addEventListener("click", () => {
        img.classList.toggle("zoomed");
    });
}

function openLightbox(imgSrc, titleText) {
    const lightbox = document.getElementById("lightbox");
    const img = document.getElementById("lightbox-img");
    const title = document.getElementById("lightbox-title");
    if (!lightbox || !img) return;

    img.src = imgSrc;
    if (title) title.textContent = titleText;
    lightbox.classList.add("active");
}

function showLoader(text, subText) {
    const overlay = document.getElementById("loader-overlay");
    const textEl = document.getElementById("loader-text");
    const subEl = document.getElementById("loader-sub");
    if (!overlay) return;

    if (textEl && text) textEl.textContent = text;
    if (subEl && subText) subEl.textContent = subText;
    overlay.classList.add("active");
}

function hideLoader() {
    const overlay = document.getElementById("loader-overlay");
    if (overlay) overlay.classList.remove("active");
}
