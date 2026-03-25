import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from fpdf import FPDF
import io
import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI 数字化体能评估", layout="wide")

# ==========================================
# --- 2. 初始化 AI 核心 (MediaPipe Pose) ---
# ==========================================
import mediapipe as mp
# 建议改用下面这种写法，更稳健
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

mp_drawing = mp.solutions.drawing_utils # 用于在图片上画骨架

# 定义一个 AI 处理函数
def process_pose_image(image_file):
    if image_file is None:
        return None, []
    
    # 将上传的文件转为 Open CV 格式
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # 运行 AI 模型
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(opencv_image_rgb)

    # 在原图上绘制骨架（为了专业感）
    annotated_image = opencv_image_rgb.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_LANDMARKS)

    return annotated_image, results.pose_landmarks

# ==========================================
# --- 3. 专业诊断逻辑 (基于坐标计算) ---
# ==========================================
def analyze_posture_landmarks(landmarks):
    issues = []
    if landmarks is None:
        return issues

    # 获取关键点
    lm = landmarks.landmark
    
    # 诊断：高低肩 (静态正侧观)
    # 计算左右肩膀关键点（11和12）的Y坐标差
    l_sh = lm[11].y
    r_sh = lm[12].y
    if abs(l_sh - r_sh) > 0.03: # 设定阈值
        issues.append("静态诊断：高低肩明显")
        
    # 诊断：头前引 (静态左侧/右侧观)
    # 计算耳朵（7或8）和肩膀（11或12）的X坐标差
    ear = lm[8].x # 假设使用右耳
    sh = lm[12].x # 假设使用右肩
    # 坐标范围是 0-1，需要注意方向
    if (ear > sh + 0.05):
        issues.append("静态诊断：轻度头前引")

    return issues

# --- 4. 界面标题 ---
st.title("🛡️ AI 数字化体能评估 & 智能诊断系统")
st.caption("AI 自动识别体态问题 | MediaPipe技术支持")

# --- 5. 第一步：基本信息记录 ---
st.header("第一步：基础信息采集")
with st.container(border=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        name = st.text_input("客户姓名", "张三")
    with col2:
        needs = st.multiselect("主要目标", ["增肌", "减脂", "体态改善", "疼痛缓解"])

# ==========================================
# --- 6. 第二步：AI 静态体位评估 (核心升级) ---
# ==========================================
st.header("第二步：AI 静态体位智能评估")
with st.container(border=True):
    st.info("请使用 iPad 相机拍摄客户的自然站立状态（正侧观、左侧观）。点击拍照后，AI 将自动进行诊断。")
    
    # 定义两个相机分栏
    cam_col1, cam_col2 = st.columns(2)
    ai_issues = [] # 用于汇总诊断结果
    
    # 静态正侧观
    with cam_col1:
        st.write("**正侧观** (用于诊断高低肩)")
        img_front = st.camera_input("拍照", key="front")
        
        # 当照片上传后，触发 AI
        if img_front:
            with st.spinner('AI 正在分析正侧体态...'):
                ai_img, landmarks = process_pose_image(img_front)
                st.image(ai_img, caption="AI 骨架追踪预览", use_container_width=True)
                
                # 运行诊断逻辑
                issues = analyze_posture_landmarks(landmarks)
                if issues:
                    st.warning("AI 诊断要点：")
                    for iss in issues:
                        st.write(f"- ❌ {iss}")
                        ai_issues.append(iss)
                else:
                    st.success("AI 诊断：正侧体态未见明显异常")

    # 静态左侧观
    with cam_col2:
        st.write("**左侧观** (用于诊断头前引)")
        img_left = st.camera_input("拍照", key="left")
        
        if img_left:
            with st.spinner('AI 正在分析左侧体态...'):
                ai_img, landmarks = process_pose_image(img_left)
                st.image(ai_img, caption="AI 骨架追踪预览", use_container_width=True)
                
                issues = analyze_posture_landmarks(landmarks)
                if issues:
                    st.warning("AI 诊断要点：")
                    for iss in issues:
                        st.write(f"- ❌ {iss}")
                        ai_issues.append(iss)
                else:
                    st.success("AI 诊断：左侧体态未见明显异常")

# ==========================================
# --- 7. 第三步：动作评估与报告 (逻辑更新) ---
# ==========================================
st.divider()
if st.button("📝 生成包含 AI 诊断的 PDF 报告"):
    st.success("报告已在后台生成！")
    
    # PDF 生成部分
    pdf = FPDF()
    pdf.add_page()
    
    font_path = "simhei.ttf"
    if os.path.exists(font_path):
        pdf.add_font('Chinese', '', font_path)
        pdf.set_font('Chinese', size=16)
    else:
        pdf.set_font('Arial', size=16)

    # 写入内容
    pdf.cell(200, 10, txt=f"AI 体能评估报告: {name}", ln=True, align='C')
    pdf.set_font('Chinese' if os.path.exists(font_path) else 'Arial', size=10)
    pdf.ln(10)
    
    # 目标与 AI 诊断摘要
    pdf.multi_cell(0, 10, txt=f"运动目标: {', '.join(needs)}")
    pdf.ln(5)
    
    pdf.cell(0, 10, txt="--- AI 智能诊断要点 ---", ln=True)
    if ai_issues:
        for iss in ai_issues:
            pdf.cell(0, 10, txt=f"* {iss}", ln=True)
    else:
        pdf.cell(0, 10, txt="* 各项指标良好，继续保持！", ln=True)

    pdf_output = pdf.output(dest='S')
    st.download_button(
        label="📥 下载 PDF 详细报告",
        data=pdf_output,
        file_name=f"{name}_AI_Assessment.pdf",
        mime="application/pdf"
    )
