import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
from number_plate_detector import NumberPlateDetector
from contour_detector import ContourDetector
from puc_checker import PUCChecker
import tempfile

# Page configuration
st.set_page_config(
    page_title="PUC Check System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    """Load the number plate detector."""
    try:
        # Try custom model first (best_bikee.pt), fallback to nano
        try:
            detector = NumberPlateDetector(model_size="l")  # Custom trained model
            st.info("✓ Using custom trained YOLO model (license_plate_detector.pt)")
            return detector
        except FileNotFoundError:
            detector = NumberPlateDetector(model_size="n")  # Fallback to YOLOv8 Nano
            st.info("✓ Using YOLOv8 Nano model")
            return detector
    except Exception as e:
        st.error(f"Error loading detector: {e}")
        return None

@st.cache_resource
def load_contour_detector():
    """Load the contour-based detector."""
    try:
        detector = ContourDetector()
        return detector
    except Exception as e:
        st.error(f"Error loading contour detector: {e}")
        return None

@st.cache_resource
def load_puc_checker():
    """Load the PUC checker."""
    try:
        db_path = Path(__file__).parent / "data" / "puc_database.json"
        checker = PUCChecker(str(db_path))
        return checker
    except Exception as e:
        st.error(f"Error loading PUC checker: {e}")
        return None


def draw_bounding_boxes_pil(image, detections, detection_method="YOLO"):
    """Draw bounding boxes on image using PIL."""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_large = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        font_large = ImageFont.load_default()
    
    # Choose color based on detection method
    color = (0, 255, 0) if detection_method == "YOLO" else (255, 0, 0)  # Green for YOLO, Blue for Contours
    
    for detection in detections:
        bbox = detection['bounding_box']
        text = detection['detected_text']
        confidence = detection['confidence']
        
        # Draw rectangle
        draw.rectangle(
            [(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
            outline=color,
            width=2
        )
        
        # Draw label with background
        label = f"{text} ({confidence:.1%})"
        bbox_size = draw.textbbox((0, 0), label, font=font)
        label_height = bbox_size[3] - bbox_size[1]
        label_width = bbox_size[2] - bbox_size[0]
        
        # Background rectangle for text
        draw.rectangle(
            [(bbox['x1'], bbox['y1'] - label_height - 8),
             (bbox['x1'] + label_width + 8, bbox['y1'])],
            fill=color
        )
        
        # Draw text
        draw.text(
            (bbox['x1'] + 4, bbox['y1'] - label_height - 4),
            label,
            font=font,
            fill=(0, 0, 0)
        )
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def display_puc_status(status, vehicle_number):
    """Display PUC status with appropriate styling."""
    
    if status['status'] == 'Valid':
        st.markdown(f"""
        <div class="success-box">
            <strong>✓ VALID</strong><br>
            Vehicle: {vehicle_number}<br>
            Expires: {status['expiry_date']}<br>
            Days Remaining: {status['days_remaining']} days
        </div>
        """, unsafe_allow_html=True)
    
    elif status['status'] == 'Grace Period':
        st.markdown(f"""
        <div class="warning-box">
            <strong>⚠ GRACE PERIOD</strong><br>
            Vehicle: {vehicle_number}<br>
            Days Expired: {abs(status['days_remaining'])} days<br>
            Owner: {status['vehicle_data']['owner_name']}
        </div>
        """, unsafe_allow_html=True)
    
    elif status['status'] == 'Expired':
        st.markdown(f"""
        <div class="danger-box">
            <strong>✗ EXPIRED</strong><br>
            Vehicle: {vehicle_number}<br>
            Days Expired: {abs(status['days_remaining'])} days<br>
            Owner: {status['vehicle_data']['owner_name']}
        </div>
        """, unsafe_allow_html=True)
    
    else:  # Not Found
        st.markdown(f"""
        <div class="warning-box">
            <strong>⚠ NOT FOUND</strong><br>
            Vehicle: {vehicle_number}<br>
            Status: Not in PUC database
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🚗 PUC Check System")
    st.markdown("**Automated Pollution Under Control Verification**")
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading models..."):
        yolo_detector = load_detector()
        contour_detector = load_contour_detector()
        puc_checker = load_puc_checker()
    
    if not puc_checker:
        st.error("PUC database failed to load.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "📊 Database", "ℹ️ Info"])
    
    with tab1:
        st.header("Upload Vehicle Image")
        st.write("Upload an image of a vehicle to detect number plates and check PUC status.")
        
        # Create columns for settings
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                label_visibility="collapsed"
            )
        
        with col2:
            detection_method = st.radio(
                "Detection Method",
                options=["YOLO", "Contours"],
                horizontal=True,
                help="YOLO: Deep learning. Contours: Edge-based detection"
            )
        
        with col3:
            grace_period = st.number_input(
                "Grace Period (days)",
                min_value=0,
                max_value=30,
                value=0,
                step=1
            )
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Failed to read image. Please try another file.")
                return
            
            # Show selected detection method
            st.info(f"🔍 Using {detection_method} detection method")
            
            # Process image with selected method
            with st.spinner(f"Detecting number plates using {detection_method}..."):
                try:
                    if detection_method == "YOLO":
                        if yolo_detector:
                            detections = yolo_detector.detect_and_extract_from_array(image)
                        else:
                            st.error("YOLO detector not available")
                            return
                    else:  # Contours
                        if contour_detector:
                            detections = contour_detector.detect_and_extract_from_array(image)
                        else:
                            st.error("Contour detector not available")
                            return
                except Exception as e:
                    st.error(f"Detection error: {e}")
                    return
            
            if not detections:
                st.warning("⚠️ No number plates detected in the image.")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
            else:
                # Filter to only show highest confidence detection
                highest_confidence_detection = max(detections, key=lambda x: x['confidence'])
                detections = [highest_confidence_detection]
                
                st.success(f"✓ Detected with highest confidence ({highest_confidence_detection['confidence']:.1%}) using {detection_method}")
                
                # Display image with bounding boxes
                image_with_boxes = draw_bounding_boxes_pil(image, detections, detection_method)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, channels="BGR", caption="Original Image", use_column_width=True)
                with col2:
                    st.image(image_with_boxes, channels="BGR", caption=f"Detected Plates ({detection_method})", use_column_width=True)
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Detection & Verification Results")
                
                for i, detection in enumerate(detections, 1):
                    plate_number = detection['detected_text']
                    confidence = detection['confidence']
                    bbox = detection['bounding_box']
                    
                    # Create columns for results
                    col_ocr, col_puc = st.columns(2)
                    
                    # OCR Results
                    with col_ocr:
                        st.subheader("🔍 OCR Results")
                        st.write(f"**Detected Text**: `{plate_number}`")
                        st.write(f"**Confidence**: {confidence:.1%}")
                        st.write(f"**Bounding Box**:")
                        st.write(f"  - X: {bbox['x1']} to {bbox['x2']}")
                        st.write(f"  - Y: {bbox['y1']} to {bbox['y2']}")
                        # PUC Status
                        with col_puc:
                            st.subheader("🚗 PUC Status")
                            puc_status = puc_checker.check_puc_status(
                                plate_number,
                                grace_period_days=grace_period
                            )
                            
                            if puc_status['found']:
                                vehicle = puc_status['vehicle_data']
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Status", puc_status['status'])
                                    st.metric("Owner", vehicle['owner_name'])
                                with col_b:
                                    st.metric("Days Remaining", puc_status['days_remaining'])
                                    st.metric("Expiry Date", puc_status['expiry_date'])
                            else:
                                st.warning("Vehicle not found in database")
                        
                        # Display colored status box
                        display_puc_status(puc_status, plate_number)
                
                # Summary section
                st.markdown("---")
                st.subheader("📊 Summary")
                
                valid_count = 0
                expired_count = 0
                not_found_count = 0
                
                for detection in detections:
                    plate_number = detection['detected_text']
                    puc_status = puc_checker.check_puc_status(
                        plate_number,
                        grace_period_days=grace_period
                    )
                    
                    if puc_status['status'] == 'Valid':
                        valid_count += 1
                    elif puc_status['status'] == 'Expired':
                        expired_count += 1
                    else:
                        not_found_count += 1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detection Confidence", f"{detections[0]['confidence']:.1%}")
                with col2:
                    st.metric("Detected Method", detection_method)
    
    with tab2:
        st.header("📊 PUC Database")
        st.write("View all vehicles in the PUC database")
        
        vehicles = puc_checker.get_all_vehicles()
        
        if vehicles:
            # Create a list of vehicles with their status
            vehicle_data = []
            for vnum, vdata in sorted(vehicles.items()):
                status_check = puc_checker.check_puc_status(vnum)
                vehicle_data.append({
                    "Vehicle Number": vnum,
                    "Owner": vdata['owner_name'],
                    "Status": status_check['status'],
                    "PUC Expiry": vdata['puc_expiry_date'],
                    "Days Remaining": status_check['days_remaining'],
                    "Contact": vdata['owner_contact']
                })
            
            # Display as dataframe with search
            search_col, filter_col = st.columns(2)
            
            with search_col:
                search_term = st.text_input("Search by vehicle number or owner name:")
            
            with filter_col:
                status_filter = st.selectbox(
                    "Filter by status:",
                    ["All", "Valid", "Expired", "Not Found"]
                )
            
            # Filter data
            filtered_data = vehicle_data
            
            if search_term:
                filtered_data = [
                    v for v in filtered_data 
                    if search_term.lower() in v['Vehicle Number'].lower() or 
                       search_term.lower() in v['Owner'].lower()
                ]
            
            if status_filter != "All":
                filtered_data = [
                    v for v in filtered_data 
                    if v['Status'] == status_filter
                ]
            
            # Display as table
            st.dataframe(
                filtered_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Vehicle Number": st.column_config.TextColumn(width="medium"),
                    "Owner": st.column_config.TextColumn(width="medium"),
                    "Status": st.column_config.TextColumn(width="small"),
                    "PUC Expiry": st.column_config.TextColumn(width="small"),
                    "Days Remaining": st.column_config.NumberColumn(width="small"),
                    "Contact": st.column_config.TextColumn(width="medium")
                }
            )
            
            # Summary stats
            st.markdown("---")
            st.subheader("Database Statistics")
            
            valid_vehicles = [v for v in vehicle_data if v['Status'] == 'Valid']
            expired_vehicles = [v for v in vehicle_data if v['Status'] == 'Expired']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Vehicles", len(vehicle_data))
            with col2:
                st.metric("Valid Certificates", len(valid_vehicles))
            with col3:
                st.metric("Expired Certificates", len(expired_vehicles))
        else:
            st.info("No vehicles in database")
    
    with tab3:
        st.header("ℹ️ About PUC Check System")
        
        st.subheader("Detection Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🔷 YOLO Detection")
            st.write("""
            - **Deep Learning** based approach
            - **Faster R-CNN** style object detection
            - **Trained model** for specific objects
            - **Pros:**
              - High accuracy
              - Handles complex scenarios
              - Custom trained models available
            - **Cons:**
              - Requires GPU for speed
              - Needs training data
              - Larger memory footprint
            """)
        
        with col2:
            st.write("### 🔵 Contour Detection")
            st.write("""
            - **Edge-based** approach
            - Detects boundaries and shapes
            - **No training required**
            - **Pros:**
              - Fast processing
              - No GPU needed
              - Works offline
              - Lightweight
            - **Cons:**
              - Sensitive to image quality
              - Lighting dependent
              - Poor with complex backgrounds
            """)
        st.write("""
        **Pollution Under Control (PUC)** is a certificate issued in India to indicate that a vehicle 
        meets emission standards set by the Pollution Control Board. All vehicles must carry a valid 
        PUC certificate to operate on roads.
        """)
        
        st.subheader("System Features")
        st.write("""
        - **YOLO Detection**: Advanced number plate detection using YOLOv8
        - **OCR Recognition**: Text extraction using EasyOCR
        - **PUC Verification**: Real-time database lookup
        - **Grace Period**: Configurable tolerance for expired certificates
        - **Web Interface**: User-friendly Streamlit dashboard
        """)
        
        st.subheader("How to Use")
        st.write("""
        1. **Upload Image**: Use the Image Upload tab to upload a vehicle photo
        2. **Auto Detection**: System automatically detects number plates
        3. **Verify Status**: Check PUC expiry status instantly
        4. **View Database**: Browse all vehicles in the database
        """)
        
        st.subheader("Status Meanings")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            <strong>Valid</strong><br>
            Certificate is active
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
            <strong>Grace Period</strong><br>
            Expired but within grace
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="danger-box">
            <strong>Expired</strong><br>
            Certificate expired
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="warning-box">
            <strong>Not Found</strong><br>
            Not in database
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**YOLO Model**: YOLOv8 Nano (Local)")
            st.write(f"**OCR Engine**: EasyOCR")
            st.write(f"**Total Vehicles**: {len(puc_checker.get_all_vehicles())}")
        
        with col2:
            st.write(f"**Version**: 1.0")
            st.write(f"**Platform**: Streamlit Web App")
            st.write(f"**Date**: February 2026")


if __name__ == "__main__":
    main()
