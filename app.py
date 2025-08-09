import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from src.segmentation_pipeline import FaceHandSegmentationPipeline


st.set_page_config(
    page_title="Face & Hand Segmentation",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🎭 Automated Face & Hand Segmentation")
    st.markdown("Upload an image to automatically detect and segment faces and hands using SAM2 API")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Token input
    api_token = st.sidebar.text_input(
        "Replicate API Token",
        type="password",
        help="Enter your Replicate API token to use SAM2"
    )
    
    # Detection settings
    st.sidebar.subheader("Detection Settings")
    face_confidence = st.sidebar.slider(
        "Face Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    hand_min_area = st.sidebar.slider(
        "Hand Detection Min Area",
        min_value=500,
        max_value=5000,
        value=1000,
        step=100
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing faces and/or hands"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert PIL image to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with col2:
        st.header("🎯 Segmentation Results")
        
        if uploaded_file is not None and api_token:
            if st.button("🚀 Process Image", type="primary"):
                try:
                    # Initialize pipeline
                    with st.spinner("Initializing pipeline..."):
                        pipeline = FaceHandSegmentationPipeline(api_token)
                    
                    # Save uploaded image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_path = tmp_file.name
                        cv2.imwrite(tmp_path, image_cv)
                    
                    # Process the image
                    with st.spinner("Processing image... This may take a few moments."):
                        result = pipeline.process_image(tmp_path)
                    
                    # Display result
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption="Segmented Image", use_column_width=True)
                    
                    # Provide download button
                    result_pil = Image.fromarray(result_rgb)
                    
                    # Convert to bytes for download
                    import io
                    buf = io.BytesIO()
                    result_pil.save(buf, format='JPEG')
                    
                    st.download_button(
                        label="📥 Download Result",
                        data=buf.getvalue(),
                        file_name="segmented_image.jpg",
                        mime="image/jpeg"
                    )
                    
                    # Clean up temporary file
                    os.unlink(tmp_path)
                    
                    st.success("✅ Processing completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    
                    # Try to show detection results as fallback
                    try:
                        with st.spinner("Showing detection results..."):
                            pipeline = FaceHandSegmentationPipeline(api_token)
                            faces = pipeline.detect_faces(image_cv)
                            hands = pipeline.detect_hands(image_cv)
                            
                            fallback_result = pipeline.draw_bounding_boxes(image_cv, faces, hands)
                            fallback_rgb = cv2.cvtColor(fallback_result, cv2.COLOR_BGR2RGB)
                            
                            st.image(fallback_rgb, caption="Detection Results (Fallback)", use_column_width=True)
                            st.info(f"Detected {len(faces)} faces and {len(hands)} hands")
                            
                    except Exception as fallback_error:
                        st.error(f"❌ Fallback also failed: {str(fallback_error)}")
        
        elif uploaded_file is not None and not api_token:
            st.warning("⚠️ Please enter your Replicate API token in the sidebar to process the image.")
        
        elif not uploaded_file:
            st.info("📋 Upload an image to see the segmentation results here.")
    
    # Instructions section
    st.markdown("---")
    st.header("📋 Instructions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Setup")
        st.markdown("""
        - Get your Replicate API token from [replicate.com](https://replicate.com)
        - Enter the token in the sidebar
        - Upload an image containing faces and/or hands
        """)
    
    with col2:
        st.subheader("2. Processing")
        st.markdown("""
        - Click "Process Image" to start
        - The system will detect faces and hands
        - SAM2 API will create segmentation masks
        - Results will be displayed automatically
        """)
    
    with col3:
        st.subheader("3. Download")
        st.markdown("""
        - Download the segmented image
        - Green masks indicate faces
        - Blue masks indicate hands
        - Original image remains unchanged
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Detection Methods:**
        - **Face Detection**: OpenCV DNN or Haar Cascade fallback
        - **Hand Detection**: HSV color space analysis and contour detection
        
        **Segmentation:**
        - Uses Meta's SAM2 (Segment Anything v2) via Replicate API
        - Bounding boxes are converted to SAM2 prompts
        - High-quality segmentation masks are generated
        
        **Visualization:**
        - Faces: Green overlay masks
        - Hands: Blue overlay masks
        - 50% transparency for visibility
        """)
    
    # Sample images section
    st.markdown("---")
    st.header("📸 Sample Images")
    st.markdown("Try these sample images to test the pipeline:")
    
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        st.markdown("**Portrait Image**")
        st.markdown("Good for testing face detection")
    
    with sample_col2:
        st.markdown("**Hand Gesture Image**")
        st.markdown("Good for testing hand detection")
    
    with sample_col3:
        st.markdown("**Full Body Image**")
        st.markdown("Good for testing both face and hand detection")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit, OpenCV, and SAM2 API"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()