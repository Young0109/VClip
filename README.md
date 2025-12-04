🎬 VClip - Intelligent Video Highlight Extraction
VClip is a robust, multi-modal AI pipeline designed to automate the process of repurposing long-form videos into engaging short clips. Unlike traditional tools that rely solely on silence detection or random sampling, VClip employs a Dual-Stream Analysis strategy—evaluating both the Visual Appeal and Semantic Depth of the content to identify true highlights.

✨ Key Features:

🧠 Multi-Modal Intelligence:

Visual Stream: Uses Qwen-VL to score frames based on image quality, emotion, and information density.

Semantic Stream: Uses OpenAI Whisper for transcription and DeepSeek for NLP analysis (sentiment, keyword density, and "Golden Quote" detection).

audio Engineering: Integrated Demucs for professional-grade vocal separation to ensure clean transcriptions.

📱 Smart Reframing: Automatically detects subjects and crops horizontal videos into 9:16 vertical format, ready for TikTok/Reels/Shorts.

⚡ Scalable Architecture: Built on FastAPI with Celery + Redis for asynchronous task processing, capable of handling long video rendering tasks efficiently.
