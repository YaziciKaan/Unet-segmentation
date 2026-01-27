import streamlit as st
import requests
from PIL import Image
import numpy as np
import cv2
import io
import time
import tempfile
import os

# API Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Pothole Detection",
    layout="wide"
)

st.title("Pothole Detection System")
st.markdown("Upload an image or video to detect potholes")

# Sidebar - Model Info and Control
st.sidebar.title("Model Configuration")

# Check API status
try:
    response = requests.get(f"{API_URL}/model-info", timeout=5)
    if response.status_code == 200:
        model_info = response.json()
        st.sidebar.success("API Connected")
        
        # Display current model info
        st.sidebar.subheader("Current Model")
        st.sidebar.info(f"""
        **Type:** {model_info['model_type'].upper()}  
        **Path:** {model_info['model_path'].split('/')[-1]}  
        **Device:** {model_info['device']}  
        **Avg FPS:** {model_info.get('avg_fps', 0):.1f}
        """)
        
        # Model switching
        st.sidebar.subheader("Switch Model")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("ONNX", use_container_width=True):
                switch_response = requests.post(
                    f"{API_URL}/switch-model",
                    json={"model_name": "onnx"}
                )
                if switch_response.status_code == 200:
                    st.rerun()
        
        with col2:
            if st.button("PyTorch", use_container_width=True):
                switch_response = requests.post(
                    f"{API_URL}/switch-model",
                    json={"model_name": "pytorch"}
                )
                if switch_response.status_code == 200:
                    st.rerun()
    else:
        st.sidebar.error("API Error")
        model_info = None
except requests.exceptions.RequestException:
    st.sidebar.error("API Not Available")
    st.sidebar.info("Please start the API server:\n```bash\npython -m uvicorn app.main:app --reload\n```")
    model_info = None

# Main content
tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])

with tab1:
    st.subheader("Upload Image")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="image")
    
    if uploaded_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_image)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Detection Result")
            
            if st.button("Detect Potholes", key="detect_image"):
                with st.spinner("Processing..."):
                    try:
                        # Send image to API
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        img_byte_arr.seek(0)
                        
                        files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
                        
                        start_time = time.time()
                        response = requests.post(f"{API_URL}/predict", files=files)
                        inference_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            result_image = Image.open(io.BytesIO(response.content))
                            st.image(result_image, use_container_width=True)
                            
                            st.success(f"Processing completed in {inference_time:.2f}s")
                        else:
                            st.error(f"Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

with tab2:
    st.subheader("Upload Video")
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"], key="video")
    
    if uploaded_video is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Video")
            st.video(uploaded_video)
        
        with col2:
            st.subheader("Detection Result")
            
            if st.button("Process Video", key="detect_video"):
                with st.spinner("Processing all frames..."):
                    try:
                        # Save uploaded video to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                            tmp_input.write(uploaded_video.read())
                            input_path = tmp_input.name
                        
                        # Open video
                        cap = cv2.VideoCapture(input_path)
                        
                        # Get video properties
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Create temp output file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
                            output_path = tmp_output.name
                        
                        # Initialize video writer with mp4v codec
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        frame_count = 0
                        frame_times = []
                        
                        # Process each frame
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_start = time.time()
                            
                            # Encode frame as JPEG
                            _, buffer = cv2.imencode('.jpg', frame)
                            
                            # Send to API
                            files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
                            response = requests.post(f"{API_URL}/predict", files=files)
                            
                            if response.status_code == 200:
                                # Decode result
                                result_array = np.frombuffer(response.content, dtype=np.uint8)
                                result_frame = cv2.imdecode(result_array, cv2.IMREAD_COLOR)
                                
                                # Ensure correct dimensions
                                if result_frame.shape[:2] != (height, width):
                                    result_frame = cv2.resize(result_frame, (width, height))
                                
                                # Calculate FPS
                                frame_time = time.time() - frame_start
                                frame_fps = 1.0 / frame_time if frame_time > 0 else 0
                                frame_times.append(frame_fps)
                                
                                # Add FPS overlay
                                cv2.putText(result_frame, f"FPS: {frame_fps:.1f}", 
                                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.8, (0, 255, 0), 2)
                                cv2.putText(result_frame, f"Frame: {frame_count + 1}/{total_frames}", 
                                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.7, (255, 255, 255), 2)
                                
                                out.write(result_frame)
                            else:
                                # If API fails, write original frame
                                cv2.putText(frame, "API ERROR", (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                out.write(frame)
                            
                            frame_count += 1
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: {frame_count}/{total_frames} frames")
                        
                        # Release resources
                        cap.release()
                        out.release()
                        
                        # Calculate average FPS
                        avg_fps = np.mean(frame_times) if frame_times else 0
                        
                        # Convert to H.264 for better compatibility with Streamlit
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_h264:
                            h264_output_path = tmp_h264.name
                        
                        # Use ffmpeg to convert to H.264
                        import subprocess
                        try:
                            subprocess.run([
                                'ffmpeg', '-y', '-i', output_path,
                                '-c:v', 'libx264', '-preset', 'fast',
                                '-crf', '23', '-pix_fmt', 'yuv420p',
                                h264_output_path
                            ], check=True, capture_output=True)
                            
                            # Use H.264 version for display
                            final_output_path = h264_output_path
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            # If ffmpeg fails, use original mp4v version
                            st.warning("FFmpeg not available, using mp4v codec (may have playback issues)")
                            final_output_path = output_path
                        
                        # Display result video
                        with open(final_output_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                        
                        st.success(f"Video processed successfully!")
                        st.info(f"**Average FPS:** {avg_fps:.1f} | **Total Frames:** {total_frames}")
                        
                        # Download button
                        with open(final_output_path, 'rb') as f:
                            st.download_button(
                                label="Download Result",
                                data=f.read(),
                                file_name="pothole_detection_result.mp4",
                                mime="video/mp4"
                            )
                        
                        # Cleanup
                        os.unlink(input_path)
                        os.unlink(output_path)
                        if 'h264_output_path' in locals() and os.path.exists(h264_output_path):
                            os.unlink(h264_output_path)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        # Cleanup on error
                        if 'input_path' in locals() and os.path.exists(input_path):
                            os.unlink(input_path)
                        if 'output_path' in locals() and os.path.exists(output_path):
                            os.unlink(output_path)
                        if 'h264_output_path' in locals() and os.path.exists(h264_output_path):
                            os.unlink(h264_output_path)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This application uses a UNET model to detect potholes in images and videos.

**Features:**
- Real-time image detection
- Batch video processing
- Dynamic model switching (ONNX/PyTorch)
- FPS monitoring
""")
