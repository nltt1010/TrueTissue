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
function renderSingleResults(dataArray) {
    if (!Array.isArray(dataArray) || dataArray.length === 0) return;

    const placeholder = document.getElementById("placeholder-state");
    const resultsArea = document.getElementById("results-content");
    if (placeholder) placeholder.style.display = "none";
    if (resultsArea) resultsArea.style.display = "flex";

    const galleryContainer = document.getElementById("slice-gallery-container");
    const gallery = document.getElementById("slice-gallery");
    
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
            thumb.src = slice.images.grayscale;
            thumb.style.width = "80px";
            thumb.style.height = "80px";
            thumb.style.objectFit = "cover";
            thumb.style.borderRadius = "4px";
            thumb.style.cursor = "pointer";
            thumb.style.border = idx === 0 ? "3px solid var(--accent-cyan)" : "3px solid transparent";
            
            // Highlight color if abnormal
            if (slice.prediction.is_abnormal) {
                thumb.style.boxShadow = "0 0 10px rgba(255, 60, 60, 0.9)";
            }

            thumb.title = `Patch [Row ${slice.patch_info.row}, Col ${slice.patch_info.col}] - ${slice.prediction.prediction}`;

            thumb.addEventListener("click", () => {
                // Update borders
                Array.from(gallery.children).forEach(c => c.style.border = "3px solid transparent");
                thumb.style.border = "3px solid var(--accent-cyan)";
                renderViewForSlice(slice);
            });

            gallery.appendChild(thumb);
        });
    } else {
        galleryContainer.style.display = "none";
    }

    // Render the first slice by default
    renderViewForSlice(dataArray[0]);

    // Scroll smoothly to results
    resultsArea.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderViewForSlice(data) {
    // 1. Diagnostic Banner
    const banner = document.getElementById("diag-banner");
    const diagIcon = document.getElementById("diag-icon-el");
    const diagText = document.getElementById("diag-result-text");
    const probTumor = document.getElementById("val-tumor");
    const probNormal = document.getElementById("val-normal");
    const regCount = document.getElementById("val-regions");

    let locationText = data.patch_info && data.patch_info.is_slice ? ` (Patch ${data.patch_info.row}, ${data.patch_info.col})` : '';

    if (banner) {
        banner.className = `diag-banner ${data.prediction.is_abnormal ? 'status-tumor' : 'status-normal'}`;
    }
    if (diagIcon) {
        diagIcon.innerHTML = data.prediction.is_abnormal ? '⚠️' : '✅';
    }
    if (diagText) {
        diagText.textContent = data.prediction.prediction + locationText;
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
    
    // Note: Inline zoom/pan removed, now using Lightbox Modal on click
}

/**
 * Optional Lightbox modal logic for click-to-zoom if user wishes to inspect even closer
 */
function initLightbox() {
    const lightbox = document.getElementById("lightbox");
    const closeBtn = document.getElementById("lightbox-close");
    const img = document.getElementById("lightbox-img");
    const wrapper = document.querySelector(".lightbox-img-wrapper");

    if (!lightbox || !img || !wrapper) return;

    // Strict styling for predictable math
    wrapper.style.overflow = "hidden";
    wrapper.style.position = "relative";
    
    img.style.transition = "none";
    img.style.transformOrigin = "0 0";
    img.style.width = "100%";
    img.style.height = "100%";
    img.style.objectFit = "contain";

    let scale = 1;
    let pointX = 0;
    let pointY = 0;
    let startX = 0;
    let startY = 0;
    let isDragging = false;

    function setTransform() {
        img.style.transform = `translate(${pointX}px, ${pointY}px) scale(${scale})`;
    }

    function resetZoom() {
        scale = 1;
        pointX = 0;
        pointY = 0;
        setTransform();
    }

    function closeLightbox() {
        lightbox.classList.remove("active");
        resetZoom();
    }

    if (closeBtn) {
        closeBtn.addEventListener("click", closeLightbox);
    }

    lightbox.addEventListener("click", (e) => {
        if (e.target === lightbox || e.target.classList.contains("lightbox-img-wrapper")) {
            closeLightbox();
        }
    });

    // Zoom on wheel (Wrapper acts as the stable viewport)
    wrapper.addEventListener("wheel", (e) => {
        e.preventDefault();
        
        const rect = wrapper.getBoundingClientRect();
        
        // Mouse coordinate relative to the stable wrapper (viewport)
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        // Zoom factor
        const delta = e.deltaY > 0 ? 0.85 : 1.15;
        const newScale = Math.max(1, Math.min(scale * delta, 25)); // Cap 1x -> 25x
        
        if (newScale === 1) {
            resetZoom();
        } else {
            // imageX/Y is the coordinate inside the unscaled image under the mouse
            const imageX = (mouseX - pointX) / scale;
            const imageY = (mouseY - pointY) / scale;
            
            // Adjust pointX/Y so that imageX/Y stays exactly under mouseX/Y
            pointX = mouseX - imageX * newScale;
            pointY = mouseY - imageY * newScale;
            
            scale = newScale;
            setTransform();
        }
    });

    // Pan on drag
    wrapper.addEventListener("mousedown", (e) => {
        e.preventDefault(); // Prevent native HTML image ghost drag
        if (scale > 1) {
            isDragging = true;
            startX = e.clientX - pointX;
            startY = e.clientY - pointY;
            wrapper.style.cursor = "grabbing";
        }
    });

    window.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        e.preventDefault();
        pointX = e.clientX - startX;
        pointY = e.clientY - startY;
        setTransform();
    });

    window.addEventListener("mouseup", () => {
        isDragging = false;
        wrapper.style.cursor = "default";
    });

    // Bind click events on all large images to open Lightbox
    document.querySelectorAll('.large-image-container img').forEach(displayImg => {
        displayImg.style.cursor = 'zoom-in';
        displayImg.addEventListener('click', (e) => {
            const container = e.target.closest('.large-image-card');
            let title = "Detailed Tissue Observation";
            if (container) {
                const titleEl = container.querySelector('.image-title');
                if (titleEl) title = titleEl.textContent.trim();
            }
            
            resetZoom();
            openLightbox(e.target.src, title);
        });
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
