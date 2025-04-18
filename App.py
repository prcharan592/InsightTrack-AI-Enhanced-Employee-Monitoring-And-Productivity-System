import streamlit as st
import cv2
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import tempfile
from ultralytics import YOLO
import time

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load YOLO model
yolo_model = YOLO("last1.pt")

# Custom CSS for attractive styling with animations
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f0f2f6 0%, #e0e7ff 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .title {
        font-size: 3em;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.2);
        animation: bounce 1.5s infinite;
    }
    @keyframes bounce { 
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    .subtitle {
        font-size: 1.8em;
        color: #3498db;
        margin-top: 20px;
        font-weight: bold;
        animation: slideIn 1s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .metrics-box, .report-box, .uploader {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 8px solid #3498db;
        animation: popIn 0.8s ease-out;
    }
    @keyframes popIn {
        from { transform: scale(0.8); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    .stButton>button {
        background: linear-gradient(45deg, #3498db, #2ecc71);
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(52, 152, 219, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 152, 219, 0); }
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #2ecc71, #3498db);
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .spinner {
        color: #3498db;
        font-size: 1.2em;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        color: #7f8c8d;
        font-size: 1em;
        animation: fadeIn 1s ease-in;
    }
    </style>
""", unsafe_allow_html=True)

def process_video(video_file):
    """Process video with YOLO and collect all frames for analysis"""
    frames_for_gemini = []
    detection_results = {
        'total_persons': 0,
        'total_cabinets': 0,
        'frame_count': 0,
        'frame_detections': []
    }
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_file.getvalue())
        video_path = tmpfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_in_seconds = frame_idx / fps
        results = yolo_model(frame)
        
        frame_person_count = 0
        frame_cabinet_count = 0
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 1:  # person
                    label = "Person"
                    color = (255, 0, 0)
                    frame_person_count += 1
                elif cls_id == 0:  # cabinet
                    label = "Cabinet"
                    color = (0, 255, 0)
                    frame_cabinet_count += 1
                else:
                    continue
                
                bbox = box.xyxy[0]
                x1, y1, x2, y2 = map(int, bbox)
                time_label = f"{time_in_seconds:.2f} sec"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, time_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        detection_results['total_persons'] += frame_person_count
        detection_results['total_cabinets'] += frame_cabinet_count
        
        detection_results['frame_detections'].append({
            'frame_idx': frame_idx,
            'time': time_in_seconds,
            'persons': frame_person_count,
            'cabinets': frame_cabinet_count
        })
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_for_gemini.append(Image.fromarray(frame_rgb))
        
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        metrics_placeholder.markdown(f"""
        <div class="metrics-box">
            <h3>Processing Progress</h3>
            <p><b>Frame:</b> {frame_idx}/{total_frames}</p>
            <p><b>Time:</b> {time_in_seconds:.2f} seconds</p>
            <p><b>Current Frame Detections:</b></p>
            <ul>
                <li>Persons: {frame_person_count}</li>
                <li>Cabinets: {frame_cabinet_count}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        frame_idx += 1
        time.sleep(0.01)  # Small delay for smoother animation
    
    detection_results['frame_count'] = frame_idx
    detection_results['total_time'] = frame_idx/fps
    
    cap.release()
    os.unlink(video_path)
    
    return detection_results, frames_for_gemini

def generate_gemini_report(frames, yolo_results):
    """Generate concise, actionable report using Gemini"""
    avg_persons = sum(frame['persons'] for frame in yolo_results['frame_detections']) / len(yolo_results['frame_detections'])
    avg_cabinets = sum(frame['cabinets'] for frame in yolo_results['frame_detections']) / len(yolo_results['frame_detections'])
    
    analysis_prompt = f"""
    Analyze this workplace video and provide only concise, actionable insights and decisions based on these metrics:
    - Total Frames Analyzed: {yolo_results['frame_count']}
    - Video Duration: {yolo_results['total_time']:.2f} seconds
    - Total Person Detections: {yolo_results['total_persons']}
    - Total Cabinet Detections: {yolo_results['total_cabinets']}
    - Average Persons per Frame: {avg_persons:.2f}
    - Average Cabinets per Frame: {avg_cabinets:.2f}

    Focus on:
    1. Key cabinet usage patterns and optimization recommendations
    2. People movement patterns and potential interaction hotspots
    3. Peak activity periods for resource allocation
    4. Major workplace utilization decisions

    Keep the response brief, under 200 words, and actionable.
    """
    
    content = [analysis_prompt]
    content.extend(frames[:5])  # Use fewer frames for faster processing
    
    try:
        response = gemini_model.generate_content(content)
        
        concise_report = f"""
================================================================================
                InsightTrack: Workplace Analysis Report
================================================================================

Key Insights & Decisions
-------------------
{response.text}

Metrics Summary
---------------
- Duration: {yolo_results['total_time']:.2f} sec
- Frames: {yolo_results['frame_count']}
- Avg Persons/Frame: {avg_persons:.2f}
- Avg Cabinets/Frame: {avg_cabinets:.2f}
- Total Persons: {yolo_results['total_persons']}
- Total Cabinets: {yolo_results['total_cabinets']}

Generated on: {st.session_state.get('current_date', 'April 12, 2025')}
================================================================================
"""
        return concise_report
    except Exception as e:
        st.error(f"Error in Gemini analysis: {str(e)}")
        return None

def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">InsightTrack: AI-Enhanced Employee Monitoring</h1>', unsafe_allow_html=True)
    st.write("Upload a workplace video to gain actionable insights using advanced AI detection and analysis.")
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'], help="Supports MP4 format only", key="uploader")
    
    if uploaded_file is not None:
        if st.button("Analyze Video"):
            with st.spinner('Analyzing video with AI...'):  # Removed unsafe_allow_html
                yolo_results, frames_for_gemini = process_video(uploaded_file)
                
                st.markdown('<h2 class="subtitle">Detection Summary</h2>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metrics-box">
                    <p><b>Total Frames:</b> {yolo_results['frame_count']}</p>
                    <p><b>Duration:</b> {yolo_results['total_time']:.2f} sec</p>
                    <p><b>Persons Detected:</b> {yolo_results['total_persons']}</p>
                    <p><b>Cabinets Detected:</b> {yolo_results['total_cabinets']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner('Generating insights...'):  # Removed unsafe_allow_html
                report = generate_gemini_report(frames_for_gemini, yolo_results)
                
                if report:
                    st.markdown('<h2 class="subtitle">Actionable Insights Report</h2>', unsafe_allow_html=True)
                    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="InsightTrack_Analysis_Report.txt",
                        mime="text/plain",
                        key="download_button"
                    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">© 2025 InsightTrack. All rights reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
