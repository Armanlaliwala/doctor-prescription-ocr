import streamlit as st
import easyocr
import os
import numpy as np
from PIL import Image
import warnings
import time

# Set environment variable to bypass OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Streamlit Page Configuration
st.set_page_config(page_title="EasyOCR Line Detector", layout="centered")

# Title
st.title("🧠 EasyOCR Handwriting Reader (Line-by-Line)")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and convert image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize if too large
    max_width = 800
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size)

    # Display image
    st.image(image, caption='🖼️ Resized Image for Faster OCR', use_column_width=True)

    # Convert to numpy array
    image_np = np.array(image)

    # OCR Reader
    with st.spinner("🔍 Reading text..."):
        start_time = time.time()
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(image_np)
        elapsed = time.time() - start_time
        st.success(f"✅ OCR completed in {elapsed:.2f} seconds")

    # Sort results by Y-coordinate
    results.sort(key=lambda x: x[0][0][1])  # Sort by top-left corner's y

    # Group text line-by-line
    lines = []
    current_line = []
    line_threshold = 15  # Change if needed

    for bbox, text, prob in results:
        y = bbox[0][1]
        if not current_line:
            current_line.append((text, y))
        else:
            _, prev_y = current_line[-1]
            if abs(y - prev_y) <= line_threshold:
                current_line.append((text, y))
            else:
                line_text = " ".join([t for t, _ in sorted(current_line, key=lambda x: x[1])])
                lines.append(line_text)
                current_line = [(text, y)]

    if current_line:
        line_text = " ".join([t for t, _ in sorted(current_line, key=lambda x: x[1])])
        lines.append(line_text)

    # Display the results
    st.subheader("📄 Extracted Text:")
    for i, line in enumerate(lines, 1):
        st.markdown(f"**Line {i}:** {line}")
