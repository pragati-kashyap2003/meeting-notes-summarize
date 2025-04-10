import gradio as gr
import whisper
from transformers import pipeline
from gtts import gTTS
import os
import tempfile

# Global model variables for caching
whisper_model = None
summarizer = None

def preload():
    global whisper_model, summarizer
    # Load the tiny whisper model for faster performance
    whisper_model = whisper.load_model("tiny")  
    # Use a stronger summarizer model for better summary quality
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

preload()

def summarize_text(text, chunk_size=1000):
    summaries = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        summary = summarizer(
            chunk,
            max_length=250,
            min_length=80,
            do_sample=False
        )[0]["summary_text"]
        summaries.append(summary)
    return " ".join(summaries)

def process_audio(audio_file):
    global whisper_model, summarizer

    # Step 1: Transcribe
    result = whisper_model.transcribe(audio_file, language="en")
    transcription = result["text"]

    # Step 2: Summarize
    summary = summarize_text(transcription)

    # Step 3: Text-to-speech
    tts = gTTS(summary)
    output_path = os.path.join(tempfile.gettempdir(), "summary.mp3")
    tts.save(output_path)

    return transcription, summary, output_path

# Gradio App
def create_app():
    with gr.Blocks() as app:
        gr.Markdown("## 🗣️ Voice Summarizer Chatbot")
        gr.Markdown("Speak into the mic, and get a summarized version with audio!")

        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="🎙 Record Your Voice")

        with gr.Row():
            transcription_output = gr.Textbox(
                label="📝 Transcription", 
                lines=15, 
                max_lines=30, 
                scale=1
            )
            summary_output = gr.Textbox(
                label="🧠 Summary", 
                lines=10, 
                max_lines=25, 
                scale=1
            )

        summary_audio_output = gr.Audio(label="🔊 Summary Audio")
        submit_btn = gr.Button("Submit")

        submit_btn.click(
            fn=process_audio,
            inputs=audio_input,
            outputs=[transcription_output, summary_output, summary_audio_output]
        )

        gr.Markdown("""---  
**Created by:**  
👩‍💻 Pragati Kashyap (Roll No: 01)  
👩‍💻 Shaina Choudhary (Roll No: 03)  
👨‍💻 Uday Sankar Chintha (Roll No: 17)  
""")

    return app

app = create_app()
app.launch()
