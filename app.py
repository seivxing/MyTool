import streamlit as st
import asyncio
import edge_tts
import pysubs2
from pydub import AudioSegment
import numpy as np
import os
import tempfile
import re
import time

# មុខងារជំនួយសម្រាប់លុប HTML tags ក្នុង SRT
def clean_text(text):
    text = re.sub(r'<[^>]*>', '', text)
    text = text.replace("\\N", " ").replace("\n", " ")
    return text.strip()

async def process_srt_to_mp3(srt_content, voice, rate, pitch, concurrency=5):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:
        tmp_srt.write(srt_content)
        tmp_srt_path = tmp_srt.name

    subs = pysubs2.load(tmp_srt_path, encoding="utf-8")
    
    if not subs:
        raise ValueError("ឯកសារ SRT របស់អ្នកទទេស្អាត ឬខូច។")

    lines_with_text = [line for line in subs if clean_text(line.text)]

    if not lines_with_text:
        combined_audio = AudioSegment.silent(duration=1000)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        combined_audio.export(output_path, format="mp3", bitrate="192k")
        os.remove(tmp_srt_path)
        return output_path

    progress_bar = st.progress(0)
    total_tasks = len(lines_with_text)
    completed = 0

    # --- Diagnostic timers ---
    # tts_time = ផលបូក ពេលវេលារង់ចាំ server edge_tts (បើធ្វើម្តងមួយៗតាមលំដាប់)
    # decode_time = ពេលវេលា decode mp3 -> AudioSegment (local)
    tts_time = 0.0
    decode_time = 0.0
    frame_rate = None
    channels = None
    actual_end_time = 0
    rendered = []

    # ៣. ដំណើរការ TTS ជា CONCURRENT ជំនួសឱ្យ sequential
    #    មូលហេតុ: ការហៅ edge_tts.Communicate ម្តងមួយៗ (await ជាប់លំដាប់) មានន័យថា
    #    ពេលវេលារង់ចាំ server សរុប = ចំនួនបន្ទាត់ x latency ក្នុងមួយ request
    #    (ឧ. ៥០០ បន្ទាត់ x ០.៨វិ = ~៦៦៧ វិនាទី!) ។ ការបើក request ជាច្រើនក្នុងពេលតែមួយ
    #    (កំណត់ដោយ semaphore ដើម្បីកុំឱ្យ server បដិសេធ) កាត់បន្ថយពេលវេលារង់ចាំសរុប
    #    ចុះទៅជិតនឹងពេលវេលារបស់ request ដែលយឺតបំផុតតែមួយប៉ុណ្ណោះ។
    semaphore = asyncio.Semaphore(concurrency)

    async def render_line(line, max_retries=2):
        text = clean_text(line.text)
        last_err = None
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                    t0 = time.perf_counter()
                    await communicate.save(tmp_path)
                    t1 = time.perf_counter()
                # decode នៅក្រៅ semaphore ដើម្បីមិនកាន់កាប់ slot ដោយឥតប្រយោជន៍
                segment = AudioSegment.from_mp3(tmp_path)
                t2 = time.perf_counter()
                try:
                    os.remove(tmp_path)
                except:
                    pass
                return line.start, segment, (t1 - t0), (t2 - t1)
            except Exception as e:
                last_err = e
                await asyncio.sleep(1)
        raise RuntimeError(f"បរាជ័យក្នុងការបំប្លែងបន្ទាត់ '{text[:30]}...': {last_err}")

    tasks = [asyncio.create_task(render_line(line)) for line in lines_with_text]

    t_tts_wall_start = time.perf_counter()
    for coro in asyncio.as_completed(tasks):
        start_ms, segment, tts_dt, decode_dt = await coro
        tts_time += tts_dt
        decode_time += decode_dt

        if frame_rate is None:
            frame_rate = segment.frame_rate
            channels = segment.channels

        rendered.append((start_ms, segment))
        current_end = start_ms + len(segment)
        if current_end > actual_end_time:
            actual_end_time = current_end

        completed += 1
        progress_bar.progress(completed / total_tasks)

    tts_wall_time = time.perf_counter() - t_tts_wall_start

    # ២. លាយសំឡេងទាំងអស់តាមរយៈ numpy buffer តែមួយ (O(n) ជំនួសឱ្យ O(n²))
    #    មូលហេតុ: AudioSegment របស់ pydub គឺ immutable ដូច្នេះរាល់ដងហៅ .overlay()
    #    វានឹងចម្លងសំឡេងទាំងមូល(មុន+ក្រោយ)ជាថ្មីម្តងៗ។ សម្រាប់ SRT វែង/ច្រើនបន្ទាត់
    #    វាកាន់តែយឺតខ្លាំង (quadratic) ។ ការបូកចូល numpy array ដោយផ្ទាល់វិញ គឺលឿនណាស់
    #    ព្រោះនីមួយៗប៉ះតែជួរដែលចាំបាច់ប៉ុណ្ណោះ។
    sample_width = 2  # បង្ខំទៅ 16-bit ដើម្បីឱ្យស្រប
    total_ms = actual_end_time + 1000
    total_samples = int(total_ms * frame_rate / 1000) * channels
    buffer = np.zeros(total_samples, dtype=np.int32)  # int32 ដើម្បីការពារ overflow ពេលបូក

    t_mix_start = time.perf_counter()
    for start_ms, segment in rendered:
        segment = (segment
                   .set_frame_rate(frame_rate)
                   .set_channels(channels)
                   .set_sample_width(sample_width))
        samples = np.array(segment.get_array_of_samples(), dtype=np.int32)

        start_sample = int(start_ms * frame_rate / 1000) * channels
        end_sample = start_sample + len(samples)

        if end_sample > len(buffer):
            buffer = np.concatenate([buffer, np.zeros(end_sample - len(buffer), dtype=np.int32)])

        buffer[start_sample:end_sample] += samples

    # កាត់កុំឱ្យលើសដែនកំណត់ 16-bit (ការពារសំឡេងខូចប្រសិនបើមានការត្រួតគ្នា)
    buffer = np.clip(buffer, -32768, 32767).astype(np.int16)
    mix_time = time.perf_counter() - t_mix_start

    combined_audio = AudioSegment(
        buffer.tobytes(),
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=channels,
    )

    # នាំចេញ File សម្រេច
    t_export_start = time.perf_counter()
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    combined_audio.export(output_path, format="mp3", bitrate="192k")
    export_time = time.perf_counter() - t_export_start
    os.remove(tmp_srt_path)

    # --- បង្ហាញលទ្ធផលរាយការណ៍ពេលវេលា ---
    wall_total = tts_wall_time + mix_time + export_time
    speedup = tts_time / tts_wall_time if tts_wall_time > 0 else 1.0
    st.info(
        f"⏱️ **ការវិភាគពេលវេលា** (សរុបប្រើពេលពិតប្រាកដ {wall_total:.1f}s សម្រាប់ {len(rendered)} បន្ទាត់, "
        f"concurrency={concurrency})\n\n"
        f"- 🌐 TTS (concurrent, ពេលពិតប្រាកដ): **{tts_wall_time:.1f}s** "
        f"(បើហៅម្តងមួយៗវិញ នឹងចំណាយ ~{tts_time:.1f}s → **លឿនជាង ~{speedup:.1f}x**)\n"
        f"- 💽 Decode mp3 សរុប (local/ffmpeg, ជាប់លំដាប់ក្នុង concurrency window): {decode_time:.1f}s\n"
        f"- 🎚️ លាយសំឡេង numpy (local): {mix_time:.1f}s\n"
        f"- 📦 Export mp3: {export_time:.1f}s\n\n"
        f"បើចង់លឿនជាងនេះទៀត សាកបង្កើន concurrency ក្នុង UI ខាងលើ "
        f"(ប៉ុន្តែបើដាក់ខ្ពស់ពេក server edge_tts អាចបដិសេធ/error ច្រើន)។"
    )

    return output_path

# --- ផ្នែក Interface (UI) ---
st.set_page_config(page_title="SRT to Audio Converter (Pro Sync)", page_icon="🎙️", layout="centered")
st.title("🎙️ កម្មវិធីបំប្លែង SRT ទៅជាសំឡេង (Audio Sync)")

uploaded_file = st.file_uploader("សូមជ្រើសរើសឯកសារ .srt របស់អ្នក", type=["srt"])

col1, col2, col3 = st.columns(3)

with col1:
    # កំណត់ Default ជា Sreymom ដោយប្រើ index=1
    voice_option = st.selectbox("ជ្រើសរើសសំឡេង", 
                                ["km-KH-PisethNeural (ប្រុស)", "km-KH-SreymomNeural (ស្រី)"],
                                index=1)
with col2:
    # កំណត់ Default ល្បឿននិយាយ = 45%
    speed = st.slider("ល្បឿននិយាយ (%)", -50, 100, 45, step=5)
with col3:
    # កំណត់ Default កម្រិត Pitch = 18Hz
    pitch_val = st.slider("កម្រិតសំឡេង Pitch (Hz)", -50, 50, 18, step=1)

concurrency = st.slider(
    "⚡ ចំនួន Request ព្រមគ្នា (Concurrency)", 1, 15, 5, step=1,
    help="ខ្ពស់ = លឿនជាង ព្រោះហៅ edge_tts server ច្រើនក្នុងពេលតែមួយ។ "
         "ប៉ុន្តែបើដាក់ខ្ពស់ពេក server អាចបដិសេធសំណើរ (error ច្រើន) - ចាប់ផ្តើមពី 5 ជាមុន។"
)

voice_id = voice_option.split(" ")[0]
rate_str = f"{speed:+d}%"
pitch_str = f"{pitch_val:+d}Hz"

if uploaded_file is not None:
    # ចាប់យកឈ្មោះ File ដើម ហើយប្តូរកន្ទុយទៅជា .mp3
    original_filename = uploaded_file.name
    base_name = os.path.splitext(original_filename)[0]
    output_filename = f"{base_name}.mp3"

    if st.button("ចាប់ផ្តើមបំប្លែង (Start Sync)"):
        with st.spinner("កំពុងបំប្លែង... ប្រព័ន្ធកំពុងដោតសំឡេងចូលវិនាទីនីមួយៗយ៉ាងសុក្រិត..."):
            try:
                srt_bytes = uploaded_file.read()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result_path = loop.run_until_complete(
                    process_srt_to_mp3(srt_bytes, voice_id, rate_str, pitch_str, concurrency=concurrency)
                )
                
                st.success("ការបំប្លែងជោគជ័យ! សំឡេងដើរទាន់អក្សរហើយ!")
                with open(result_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
                    # ដាក់ឈ្មោះថ្មីត្រង់កន្លែង file_name
                    st.download_button("📥 ទាញយកឯកសារ MP3", f, file_name=output_filename)
            except Exception as e:
                st.error(f"មានបញ្ហាបច្ចេកទេស៖ {e}")
