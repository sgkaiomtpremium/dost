import streamlit as st
import torch
import sys
import os
from PIL import Image
import json
import time
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="dots.ocr - Document Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None

@st.cache_resource
def check_torch_installation():
    """Check if PyTorch is properly installed"""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return True, device, torch.__version__
    except ImportError:
        return False, None, None

@st.cache_resource(show_spinner="🔄 Đang tải model dots.ocr...")
def load_model():
    """Load model once and cache it"""
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        # Sử dụng model ID từ Hugging Face
        model_path = "rednote-hilab/dots.ocr"
        
        # Load processor first
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # Determine torch_dtype based on device availability
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device_map = "auto"
        else:
            torch_dtype = torch.float32
            device_map = "cpu"
        
        # Load model with optimized settings
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        return model, processor
        
    except Exception as e:
        st.error(f"❌ Lỗi khi tải model: {str(e)}")
        st.info("💡 Có thể do model quá lớn hoặc kết nối mạng. Thử lại sau ít phút.")
        return None, None

def get_prompt_template(prompt_type):
    """Get prompt template based on type"""
    prompts = {
        "layout_all": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].
3. Text Extraction & Formatting Rules:
   - Picture: For the 'Picture' category, the text field should be omitted.
   - Formula: Format its text as LaTeX.
   - Table: Format its text as HTML.
   - All Others (Text, Title, etc.): Format their text as Markdown.
4. Constraints:
   - The output text must be the original text from the image, with no translation.
   - All layout elements must be sorted according to human reading order.
5. Final Output: The entire output must be a single JSON object.""",
        
        "ocr_only": """Extract all text content from this document image in reading order. 
Output only the text content without any layout information or formatting.""",
        
        "layout_only": """Detect and output only the layout elements and their bounding boxes from this document image.
Format: JSON with bbox coordinates and element categories only, no text content.""",
        
        "table_extract": """Extract all tables from this document image.
Format each table as HTML and provide bounding box coordinates.""",
        
        "formula_extract": """Extract all mathematical formulas from this document image.
Format each formula as LaTeX and provide bounding box coordinates."""
    }
    return prompts.get(prompt_type, prompts["layout_all"])

def process_document(image, model, processor, prompt_type="layout_all"):
    """Process document with dots.ocr"""
    
    prompt = get_prompt_template(prompt_type)
    
    # Resize image if too large to prevent memory issues
    max_size = (2048, 2048)
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        st.info(f"📏 Đã resize ảnh xuống {image.size} để tối ưu hiệu suất")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    try:
        from qwen_vl_utils import process_vision_info
        
        # Prepare inputs
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Progress bar for generation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate with progress updates
        status_text.text("🤖 Đang phân tích tài liệu...")
        progress_bar.progress(25)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=4000,
                do_sample=False,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        progress_bar.progress(75)
        status_text.text("📝 Đang xử lý kết quả...")
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Hoàn thành!")
        
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return output_text[0]
        
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý: {str(e)}")
        return None

def display_results(result, prompt_type, image_name):
    """Display and format results"""
    if result is None:
        return
        
    st.subheader("📋 Kết quả phân tích")
    
    # Create tabs for different views
    if prompt_type == "layout_all":
        tab1, tab2, tab3 = st.tabs(["📊 JSON View", "📝 Raw Text", "💾 Download"])
        
        with tab1:
            try:
                # Try to parse as JSON
                parsed_result = json.loads(result)
                st.json(parsed_result, expanded=True)
                
                # Summary statistics
                if isinstance(parsed_result, list):
                    st.info(f"📈 Phát hiện {len(parsed_result)} elements")
                    
                    # Count by type
                    type_counts = {}
                    for item in parsed_result:
                        item_type = item.get('category', 'Unknown')
                        type_counts[item_type] = type_counts.get(item_type, 0) + 1
                    
                    st.write("**Thống kê theo loại:**")
                    for item_type, count in type_counts.items():
                        st.write(f"- {item_type}: {count}")
                        
            except json.JSONDecodeError:
                st.warning("⚠️ Kết quả không phải định dạng JSON hợp lệ")
                st.text_area("Raw Output:", result, height=400)
        
        with tab2:
            st.text_area("Raw Output:", result, height=400)
            
        with tab3:
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    parsed_result = json.loads(result)
                    json_str = json.dumps(parsed_result, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Tải xuống JSON",
                        data=json_str,
                        file_name=f"{image_name}_analysis.json",
                        mime="application/json"
                    )
                except:
                    st.download_button(
                        label="💾 Tải xuống Text",
                        data=result,
                        file_name=f"{image_name}_raw.txt",
                        mime="text/plain"
                    )
            
            with col2:
                # Extract only text content for markdown
                try:
                    parsed_result = json.loads(result)
                    text_content = ""
                    if isinstance(parsed_result, list):
                        for item in parsed_result:
                            if 'text' in item and item['text']:
                                text_content += item['text'] + "\n\n"
                    
                    if text_content:
                        st.download_button(
                            label="📄 Tải xuống Markdown",
                            data=text_content,
                            file_name=f"{image_name}_content.md",
                            mime="text/markdown"
                        )
                except:
                    pass
    else:
        # For other prompt types
        st.text_area("Kết quả:", result, height=400)
        
        st.download_button(
            label="💾 Tải xuống kết quả",
            data=result,
            file_name=f"{image_name}_{prompt_type}.txt",
            mime="text/plain"
        )

def main():
    # Check PyTorch first
    torch_ok, device, torch_version = check_torch_installation()
    
    if not torch_ok:
        st.error("❌ PyTorch chưa được cài đặt đúng cách!")
        st.code("""
        # Để fix, thử:
        pip install torch torchvision torchaudio
        """)
        return
    
    # Header
    st.markdown('<h1 class="main-header">📄 dots.ocr</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Phân tích tài liệu đa ngôn ngữ với AI - Document Parser</p>', unsafe_allow_html=True)
    
    # System info
    st.info(f"🔧 PyTorch {torch_version} | Device: {device}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Model info
        st.info("""
        🤖 **Model**: dots.ocr (1.7B parameters)
        🌍 **Languages**: 100+ ngôn ngữ
        ⚡ **Mode**: CPU/GPU Inference
        """)
        
        # Processing mode
        prompt_type = st.selectbox(
            "🎯 Chế độ xử lý:",
            ["layout_all", "ocr_only", "layout_only", "table_extract", "formula_extract"],
            help="""
            • layout_all: Phân tích đầy đủ (layout + text)
            • ocr_only: Chỉ trích xuất text
            • layout_only: Chỉ phát hiện layout
            • table_extract: Trích xuất bảng
            • formula_extract: Trích xuất công thức
            """
        )
        
        # Advanced settings
        with st.expander("🔧 Cài đặt nâng cao"):
            max_tokens = st.slider("Max tokens:", 1000, 8000, 4000)
            temperature = st.slider("Temperature:", 0.1, 1.0, 0.7)
            
        # Performance warning
        if device == "cpu":
            st.warning("""
            ⚠️ **Lưu ý hiệu suất CPU:**
            - Xử lý có thể mất 30-60 giây
            - Ảnh lớn sẽ được resize tự động
            - Khuyến nghị ảnh < 2048px
            """)
        else:
            st.success("🚀 **GPU detected** - Faster processing!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Tải lên tài liệu")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Chọn file hình ảnh:",
            type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
            help="Hỗ trợ: PNG, JPG, JPEG, WebP, BMP (< 200MB)"
        )
        
        # Example images section
        st.markdown("**📋 Hoặc sử dụng ảnh mẫu:**")
        example_choice = st.selectbox(
            "Chọn ảnh mẫu:",
            ["Không chọn", "Document tiếng Anh", "Document tiếng Việt", "Scientific Paper", "Invoice/Form"]
        )
        
        if example_choice != "Không chọn":
            st.info(f"🖼️ Đã chọn: {example_choice} (tính năng sẽ được thêm)")
    
    with col2:
        st.subheader("🎛️ Trạng thái hệ thống")
        
        # System status
        status_container = st.container()
        
        with status_container:
            # Model loading status
            if not st.session_state.model_loaded:
                if st.button("🚀 Tải Model", type="primary", use_container_width=True):
                    with st.spinner("🔄 Đang tải model..."):
                        model, processor = load_model()
                        if model is not None:
                            st.session_state.model = model
                            st.session_state.processor = processor
                            st.session_state.model_loaded = True
                            st.success("✅ Model đã tải thành công!")
                            st.rerun()
                        else:
                            st.error("❌ Không thể tải model")
            else:
                st.success("✅ Model đã sẵn sàng!")
                
                # Model stats
                model_info = st.container()
                with model_info:
                    st.metric("🧠 Model Status", "Ready")
                    st.metric("💾 Device", device.upper())
                    st.metric("🔢 Parameters", "1.7B")
    
    # Processing section
    if uploaded_file is not None and st.session_state.model_loaded:
        st.markdown("---")
        
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Ảnh đã tải lên")
            st.image(image, caption=f"📄 {uploaded_file.name}", use_column_width=True)
            
            # Image info
            st.info(f"""
            📏 **Kích thước**: {image.size[0]} x {image.size[1]} px
            📦 **Dung lượng**: {len(uploaded_file.getvalue()) / 1024:.1f} KB
            🎨 **Format**: {uploaded_file.type}
            """)
        
        with col2:
            st.subheader("🚀 Xử lý tài liệu")
            
            # Process button
            if st.button("🔍 Bắt đầu phân tích", type="primary", use_container_width=True):
                start_time = time.time()
                
                with st.spinner("🤖 Đang phân tích tài liệu..."):
                    result = process_document(
                        image, 
                        st.session_state.model, 
                        st.session_state.processor, 
                        prompt_type
                    )
                
                processing_time = time.time() - start_time
                
                if result:
                    st.success(f"✅ Hoàn thành! Thời gian xử lý: {processing_time:.1f}s")
                    
                    # Display results
                    display_results(result, prompt_type, uploaded_file.name.split('.')[0])
                else:
                    st.error("❌ Không thể xử lý tài liệu")
    
    elif uploaded_file is not None and not st.session_state.model_loaded:
        st.warning("⚠️ Vui lòng tải model trước khi xử lý tài liệu")
    
    elif uploaded_file is None and st.session_state.model_loaded:
        st.info("📤 Vui lòng tải lên một file ảnh để bắt đầu")
    
    # Footer
    st.markdown("---")
    
    # About section
    with st.expander("📚 Giới thiệu về dots.ocr"):
        st.markdown("""
        **dots.ocr** là một mô hình AI tiên tiến cho phân tích tài liệu đa ngôn ngữ với các tính năng:
        
        🎯 **Tính năng chính:**
        - Phát hiện layout tự động (văn bản, bảng, công thức, hình ảnh...)
        - OCR đa ngôn ngữ (100+ ngôn ngữ)
        - Trích xuất và định dạng nội dung thông minh
        - Bảo toàn thứ tự đọc tự nhiên
        
        ⚡ **Ưu điểm:**
        - Hiệu suất SOTA với chỉ 1.7B parameters
        - Kiến trúc đơn giản, thống nhất
        - Hỗ trợ nhiều định dạng xuất (JSON, Markdown, HTML)
        
        🔗 **Nguồn**: [GitHub - rednote-hilab/dots.ocr](https://github.com/rednote-hilab/dots.ocr)
        """)
    
    # Technical info
    with st.expander("🔧 Thông tin kỹ thuật"):
        st.code(f"""
        Python: {sys.version}
        PyTorch: {torch_version}
        Device: {device.upper()}
        Model: dots.ocr (rednote-hilab/dots.ocr)
        Parameters: 1.7B
        """)

if __name__ == "__main__":
    main()
