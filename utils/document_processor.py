import torch
import json
from PIL import Image
import io
import base64
from typing import Dict, List, Optional, Tuple, Any

class DocumentProcessor:
    """Helper class for document processing utilities"""
    
    @staticmethod
    def optimize_image(image: Image.Image, max_size: Tuple[int, int] = (2048, 2048)) -> Image.Image:
        """
        Optimize image for processing
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return image
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """
        Encode PIL image to base64 string
        """
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    @staticmethod
    def parse_layout_result(result: str) -> Optional[Dict]:
        """
        Parse and validate layout detection result
        """
        try:
            parsed = json.loads(result)
            return parsed
        except json.JSONDecodeError:
            return None
    
    @staticmethod
    def extract_text_only(layout_result: List[Dict]) -> str:
        """
        Extract only text content from layout result
        """
        text_content = []
        
        for item in layout_result:
            if isinstance(item, dict) and 'text' in item:
                text = item['text']
                if text and text.strip():
                    text_content.append(text.strip())
        
        return '\n\n'.join(text_content)
    
    @staticmethod
    def get_layout_statistics(layout_result: List[Dict]) -> Dict[str, Any]:
        """
        Get statistics about detected layout elements
        """
        if not layout_result:
            return {}
        
        stats = {
            'total_elements': len(layout_result),
            'element_types': {},
            'has_tables': False,
            'has_formulas': False,
            'text_elements': 0
        }
        
        for item in layout_result:
            if isinstance(item, dict):
                category = item.get('category', 'Unknown')
                stats['element_types'][category] = stats['element_types'].get(category, 0) + 1
                
                if category == 'Table':
                    stats['has_tables'] = True
                elif category == 'Formula':
                    stats['has_formulas'] = True
                elif category in ['Text', 'Title', 'Section-header']:
                    stats['text_elements'] += 1
        
        return stats
    
    @staticmethod
    def format_bbox(bbox: List[float]) -> str:
        """
        Format bounding box coordinates
        """
        if len(bbox) == 4:
            return f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
        return str(bbox)
    
    @staticmethod
    def validate_model_output(output: str) -> Tuple[bool, str]:
        """
        Validate model output and provide feedback
        """
        if not output or not output.strip():
            return False, "Output is empty"
        
        # Check if it's valid JSON for layout_all mode
        try:
            parsed = json.loads(output)
            if isinstance(parsed, list):
                return True, f"Valid JSON with {len(parsed)} elements"
            elif isinstance(parsed, dict):
                return True, "Valid JSON object"
            else:
                return True, "Valid JSON (other format)"
        except json.JSONDecodeError:
            # Not JSON, but might be valid for other modes
            if len(output) > 10:  # Reasonable minimum length
                return True, "Valid text output"
            else:
                return False, "Output too short"

# Memory management utilities
class MemoryManager:
    """Utilities for memory management during inference"""
    
    @staticmethod
    def clear_cache():
        """Clear torch cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # For CPU, we can try garbage collection
        import gc
        gc.collect()
    
    @staticmethod
    def get_memory_info() -> Dict[str, str]:
        """Get current memory usage info"""
        info = {}
        
        if torch.cuda.is_available():
            info['gpu_allocated'] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            info['gpu_cached'] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        else:
            info['device'] = "CPU"
            
        # System memory (if psutil is available)
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['system_memory'] = f"{memory.used / 1024**3:.2f} / {memory.total / 1024**3:.2f} GB"
            info['memory_percent'] = f"{memory.percent:.1f}%"
        except ImportError:
            info['system_memory'] = "N/A (psutil not installed)"
        
        return info

# Prompt templates
class PromptTemplates:
    """Collection of prompt templates for different tasks"""
    
    LAYOUT_ALL = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

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
5. Final Output: The entire output must be a single JSON object."""

    OCR_ONLY = """Extract all text content from this document image in reading order. 
Output only the text content without any layout information or formatting."""

    LAYOUT_ONLY = """Detect and output only the layout elements and their bounding boxes from this document image.
Format: JSON with bbox coordinates and element categories only, no text content."""

    TABLE_EXTRACT = """Extract all tables from this document image.
Format each table as HTML and provide bounding box coordinates."""

    FORMULA_EXTRACT = """Extract all mathematical formulas from this document image.
Format each formula as LaTeX and provide bounding box coordinates."""
    
    @classmethod
    def get_prompt(cls, prompt_type: str) -> str:
        """Get prompt by type"""
        prompts = {
            'layout_all': cls.LAYOUT_ALL,
            'ocr_only': cls.OCR_ONLY,
            'layout_only': cls.LAYOUT_ONLY,
            'table_extract': cls.TABLE_EXTRACT,
            'formula_extract': cls.FORMULA_EXTRACT
        }
        return prompts.get(prompt_type, cls.LAYOUT_ALL)
