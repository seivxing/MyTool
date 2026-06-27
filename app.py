import streamlit as st
import asyncio
import edge_tts
import pysubs2
from pysubs2 import SSAEvent, SSAFile
from pydub import AudioSegment
import os
import tempfile
import re
import json
import time
import uuid
import shutil
import threading

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# =====================================================================
# PAGE CONFIG  ←  ត្រូវតែជា Streamlit command ដំបូងបំផុត!
# =====================================================================

st.set_page_config(
    page_title="SRT Tool Suite",
    page_icon="🎙️",
    layout="centered",
)


# =====================================================================
# SHARED UTILITIES
# =====================================================================

def clean_text_for_tts(text: str) -> str:
    """Strip HTML/ASS tags and normalise line-breaks for TTS."""
    text = re.sub(r'<[^>]*>', '', text)
    text = text.replace("\\N", " ").replace("\n", " ")
    return text.strip()


def is_valid_srt_filename(filename: str) -> bool:
    """
    Check the uploaded file looks like an .srt file by its name.
    We validate manually (instead of using file_uploader's `type=` filter)
    because iOS Safari/WebKit has a known bug: when `accept` is restricted
    to a custom extension like `.srt` (which has no registered iOS UTI),
    the Files app picker shows the file greyed-out and unselectable.
    Leaving `type=None` on the uploader avoids that, and we just check
    the extension ourselves here instead.
    """
    return filename.lower().endswith(".srt")


# =====================================================================
# BACKGROUND JOB STORE
# ---------------------------------------------------------------------
# Why this exists: iOS Safari aggressively suspends/unloads background
# tabs to save memory. When the user switches tab/app and comes back,
# Safari does a FULL page reload — a brand new Streamlit session, so
# anything kept in st.session_state (progress, results) is lost.
#
# Fix: run the actual long-running work (translate / TTS) in a plain
# background thread that is NOT tied to the browser session, and persist
# its progress + result to disk under a job_id. The job_id itself is
# stored in the URL's query string (st.query_params), which iOS Safari
# *does* preserve across its forced reloads. So after a reload, the app
# reads the job_id back from the URL, looks up its status on disk, and
# resumes showing progress / the finished result — the work itself never
# stopped, since it kept running server-side the whole time.
# =====================================================================

JOBS_DIR = os.path.join(tempfile.gettempdir(), "srt_tool_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)


def _job_dir(job_id: str) -> str:
    path = os.path.join(JOBS_DIR, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def _status_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "status.json")


def write_job_status(job_id: str, **updates) -> None:
    """Atomically merge `updates` into this job's status.json on disk."""
    path = _status_path(job_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    data.update(updates)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def read_job_status(job_id: str):
    path = _status_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_job_file(job_id: str, filename: str, data: bytes) -> None:
    with open(os.path.join(_job_dir(job_id), filename), "wb") as f:
        f.write(data)


def load_job_file(job_id: str, filename: str):
    path = os.path.join(_job_dir(job_id), filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def cleanup_old_jobs(max_age_seconds: int = 6 * 3600) -> None:
    """Best-effort cleanup so /tmp doesn't grow unbounded over many sessions."""
    now = time.time()
    try:
        for name in os.listdir(JOBS_DIR):
            path = os.path.join(JOBS_DIR, name)
            try:
                if os.path.isdir(path) and (now - os.path.getmtime(path)) > max_age_seconds:
                    shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


cleanup_old_jobs()


# =====================================================================
# TTS FUNCTIONS
# =====================================================================

async def process_srt_to_mp3(srt_content, voice, rate, pitch, on_progress=None):
    """
    on_progress(fraction: float) is called after each line instead of
    updating a Streamlit progress bar directly — this lets the function
    run inside a background thread, decoupled from any browser session.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:
        tmp_srt.write(srt_content)
        tmp_srt_path = tmp_srt.name

    subs = pysubs2.load(tmp_srt_path, encoding="utf-8")
    if not subs:
        raise ValueError("ឯកសារ SRT របស់អ្នកទទេស្អាត ឬខូច។")

    base_duration = subs[-1].end + 30000
    combined_audio = AudioSegment.silent(duration=base_duration)
    total_lines = len(subs)
    actual_end_time = 0

    for i, line in enumerate(subs):
        text = clean_text_for_tts(line.text)
        if not text:
            continue
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_seg:
                await communicate.save(tmp_seg.name)
                seg_size = os.path.getsize(tmp_seg.name)
                if seg_size == 0:
                    raise ValueError(
                        f"No audio was received for line {i+1} (subtitle #{line.index if hasattr(line, 'index') else i+1}): "
                        f"'{text[:60]}...'\n"
                        f"Voice='{voice}', Rate='{rate}', Pitch='{pitch}'"
                    )
                segment = AudioSegment.from_mp3(tmp_seg.name)
                combined_audio = combined_audio.overlay(segment, position=line.start)
                current_end = line.start + len(segment)
                if current_end > actual_end_time:
                    actual_end_time = current_end
                try:
                    os.remove(tmp_seg.name)
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError(f"❌ Error at SRT line {i+1}: {e}") from e
        if on_progress:
            on_progress((i + 1) / total_lines)

    combined_audio = combined_audio[:actual_end_time + 1000]
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    combined_audio.export(output_path, format="mp3", bitrate="192k")
    os.remove(tmp_srt_path)
    return output_path


# =====================================================================
# TRANSLATION FUNCTIONS
# =====================================================================

GEMINI_MODELS = [
    "gemini-3.5-flash",               # Gemini 3.5 Flash ★★ (Newest · May 2026 · Best Agentic)
    "gemini-3.1-pro-preview",         # Gemini 3.1 Pro Preview ★ (Best Reasoning)
    "gemini-3.1-flash-lite",          # Gemini 3.1 Flash-Lite (Cost-Efficient)
    "gemini-3-flash-preview",         # Gemini 3 Flash Preview
    "gemini-robotics-er-1.6-preview", # Gemini Robotics-ER 1.6 Preview (Spatial & Physical Reasoning)
    "gemini-2.5-pro",                 # Gemini 2.5 Pro (Advanced Reasoning)
    "gemini-2.5-flash-preview-05-20", # Gemini 2.5 Flash Preview (fallback)
    "gemini-2.5-flash",               # Gemini 2.5 Flash (fallback)
    "gemini-2.5-flash-lite",          # Gemini 2.5 Flash-Lite (fallback)
]
GEMINI_MODEL = GEMINI_MODELS[0]


def is_music_line(text: str) -> bool:
    return bool(re.search(r'[♪♫]', text))


def clean_for_translation(
    text: str,
    remove_ass: bool = True,
    remove_html: bool = True,
    remove_brackets: bool = True,
) -> str:
    if remove_ass:
        text = re.sub(r'\{[^}]*\}', '', text)
    if remove_html:
        text = re.sub(r'<[^>]+>', '', text)
    if remove_brackets:
        text = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    text = text.replace('\\N', '\n').replace('\\n', '\n')
    lines = []
    for ln in text.split('\n'):
        ln = ln.strip().lstrip('-').strip()
        if ln:
            lines.append(ln)
    return ' '.join(lines).strip()


def build_glossary_section(glossary: dict) -> str:
    """Build glossary section for injection into system prompt."""
    if not glossary:
        return ""
    lines = [
        "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "ESTABLISHED GLOSSARY — MANDATORY: USE THESE EXACT TRANSLATIONS EVERY TIME",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "The following terms have been established for this series.",
        "You MUST use these exact Khmer translations consistently — no variation allowed:\n",
    ]
    for eng, kh in glossary.items():
        lines.append(f"  • {eng}  →  {kh}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def build_system_prompt(glossary: dict = None) -> str:
    base = (
        "ខ្ញុំចង់អោយអ្នកបកប្រែអត្តបទ subtitles នេះទៅជាភាសាខ្មែរ "
        "និងកែសម្រួលកន្លែងដែលមិនសំខាន់អោយមានអត្ថន័យអោយបានត្រឹមត្រូវ៖\n\n"

        "1. កន្លែងដែរមានចម្រៀងត្រូវលុបចោលទាំង (ID) ♪....♪, ♫....♫\n\n"

        "2. ធ្វើយ៉ាងណាបកប្រែអោយដូចអ្នកបញ្ចូលសម្លេង (Dubbing) "
        "ដែលមានភាពធម្មជាតិ និងត្រូវតាមបរិបទរឿង។ "
        "ឈ្មោះត្រូវតែសរសេរជាភាសាខ្មែរតែមួយគត់ "
        "និងសរសេរអោយបានត្រឹមត្រូវទៅតាម subtitles ដែលមាននោះ។\n\n"

        "3. កន្លែងដែលមានសញ្ញា - សូមសរសេរជាប់គ្នាកុំចុះបន្ទាត់។\n\n"

        "4. បកប្រែដោយយកចិត្តទុកដាក់ខ្ពស់ ធ្វើអោយបានត្រឹមត្រូវ 100% "
        "មិនអោយមានកំហុស។\n\n"

        "OUTPUT FORMAT (CRITICAL):\n"
        "- Output ONLY a raw JSON object. No markdown fences, no explanation, no preamble.\n"
        '- Exact schema: {"results": [{"id": <integer>, "text": "<khmer_translation>"}]}'
    )

    if glossary:
        base += build_glossary_section(glossary)

    return base


def translate_batch(client, items: list, model: str, glossary: dict = None) -> dict:
    """Translate one batch with glossary injection and auto-retry."""
    numbered = "\n".join(f"[{idx}] {text}" for idx, text in items)
    user_prompt = (
        "Translate every numbered English subtitle line below into natural spoken Khmer "
        "for dubbing. Return raw JSON only — no markdown.\n\n"
        f"{numbered}\n\n"
        '{"results": [{"id": <n>, "text": "<khmer>"}]}'
    )

    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=build_system_prompt(glossary),
                    temperature=0.3,
                    max_output_tokens=4096,
                ),
            )
            raw = response.text.strip()
            raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)
            return {item["id"]: item["text"] for item in data["results"]}

        except Exception as e:
            err_str = str(e)
            is_busy = ("503" in err_str or "UNAVAILABLE" in err_str
                       or "high demand" in err_str or "429" in err_str
                       or "RESOURCE_EXHAUSTED" in err_str)
            if is_busy and attempt < MAX_RETRIES - 1:
                wait = 10 * (2 ** attempt)
                time.sleep(wait)
                continue
            return {idx: "" for idx, _ in items}

    return {idx: "" for idx, _ in items}


def extract_glossary_from_translation(client, model: str,
                                       english_lines: list,
                                       khmer_lines: list,
                                       existing_glossary: dict) -> dict:
    """
    Ask Gemini to extract character names, place names, and key recurring terms
    from this episode's translation and return a glossary dict {en: kh}.
    Merges with existing_glossary (existing entries win to stay stable).
    """
    sample_pairs = []
    step = max(1, len(english_lines) // 60)
    for i in range(0, min(len(english_lines), 300), step):
        if i < len(khmer_lines) and khmer_lines[i].strip():
            sample_pairs.append(f"EN: {english_lines[i]}\nKH: {khmer_lines[i]}")

    combined = "\n\n".join(sample_pairs[:60])

    prompt = (
        "Below are English subtitle lines and their Khmer translations from a drama series.\n"
        "Extract ALL character names, place names, titles, and key recurring terms "
        "(nouns that appear multiple times and need consistency).\n\n"
        "Return ONLY raw JSON — no markdown, no explanation:\n"
        '{"glossary": [{"en": "<english_term>", "kh": "<khmer_translation>"}]}\n\n'
        f"SUBTITLE PAIRS:\n{combined}"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        new_terms = {item["en"]: item["kh"] for item in data.get("glossary", [])}
        # Existing entries win — don't overwrite established terms
        merged = {**new_terms, **existing_glossary}
        return merged
    except Exception:
        return existing_glossary


def translate_srt(srt_bytes: bytes, api_key: str,
                  on_progress=None, on_status=None,
                  remove_ass: bool = True, remove_html: bool = True,
                  remove_brackets: bool = True, model: str = GEMINI_MODEL,
                  glossary: dict = None):
    """
    Full pipeline: parse → filter music → clean → batch-translate → SRT bytes.
    Returns (srt_bytes, updated_glossary).

    on_progress(fraction: float) and on_status(message: str) are plain
    callbacks instead of direct Streamlit widget calls, so this function
    can be run inside a background thread (no Streamlit context needed).
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError(
            "google-genai មិនទាន់ install ទេ។ សូម run:\n"
            "pip install google-genai"
        )

    if glossary is None:
        glossary = {}

    def _status(msg):
        if on_status:
            on_status(msg)

    def _progress(frac):
        if on_progress:
            on_progress(frac)

    client = genai.Client(api_key=api_key)

    # ── Parse SRT ──────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt", mode="wb") as f:
        f.write(srt_bytes)
        tmp_path = f.name
    try:
        try:
            subs = pysubs2.load(tmp_path, encoding="utf-8")
        except UnicodeDecodeError:
            subs = pysubs2.load(tmp_path, encoding="utf-8-sig")
    finally:
        os.remove(tmp_path)

    if not subs:
        raise ValueError("ឯកសារ SRT ទទេស្អាត ឬខូច!")

    # ── Filter music lines & clean text ────────────────────────────
    valid = []
    for sub in subs:
        if is_music_line(sub.text):
            continue
        cleaned = clean_for_translation(
            sub.text,
            remove_ass=remove_ass,
            remove_html=remove_html,
            remove_brackets=remove_brackets,
        )
        if cleaned:
            valid.append((sub, cleaned))

    if not valid:
        raise ValueError("រកមិនឃើញអត្ថបទដើម្បីបកប្រែ!")

    # ── Batch translate ─────────────────────────────────────────────
    BATCH_SIZE = 20
    all_translations: dict = {}
    total = len(valid)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch = [(start + i, valid[start + i][1]) for i in range(end - start)]

        _status(f"⏳ កំពុងបកប្រែ {start + 1}–{end} / {total} បន្ទាត់  ({model})")
        result = translate_batch(client, batch, model, glossary=glossary)
        all_translations.update(result)
        _progress((b + 1) / n_batches)

    # ── Build output SRT ────────────────────────────────────────────
    out = SSAFile()
    english_lines = [text for _, text in valid]
    khmer_lines = []
    for idx, (sub, _) in enumerate(valid):
        khmer = all_translations.get(idx, "").strip()
        khmer_lines.append(khmer)
        if khmer:
            out.append(SSAEvent(start=sub.start, end=sub.end, text=khmer))

    # ── Extract & update glossary ───────────────────────────────────
    _status("🔍 កំពុងទាញ Glossary ចេញពី EP នេះ...")
    updated_glossary = extract_glossary_from_translation(
        client, model, english_lines, khmer_lines, glossary
    )

    _status("✅ ការបកប្រែបានសម្រេច!")
    return out.to_string("srt").encode("utf-8"), updated_glossary


# =====================================================================
# BACKGROUND JOB RUNNERS
# (executed inside threading.Thread — no Streamlit calls in here)
# =====================================================================

def _run_translate_job(job_id, srt_bytes, api_key, remove_ass, remove_html,
                        remove_brackets, model, glossary, out_name):
    def on_progress(frac):
        write_job_status(job_id, progress=round(float(frac), 4))

    def on_status(msg):
        write_job_status(job_id, message=msg)

    try:
        result_bytes, updated_glossary = translate_srt(
            srt_bytes, api_key,
            on_progress=on_progress, on_status=on_status,
            remove_ass=remove_ass, remove_html=remove_html,
            remove_brackets=remove_brackets, model=model, glossary=glossary,
        )
        save_job_file(job_id, "result.srt", result_bytes)
        save_job_file(
            job_id, "glossary.json",
            json.dumps(updated_glossary, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        write_job_status(
            job_id, status="done", progress=1.0,
            message="✅ ការបកប្រែបានសម្រេច!",
            out_name=out_name, glossary_count=len(updated_glossary),
        )
    except Exception as e:
        write_job_status(job_id, status="error", error=str(e))


def _run_tts_job(job_id, srt_bytes, voice, rate, pitch, mp3_name):
    def on_progress(frac):
        write_job_status(job_id, progress=round(float(frac), 4))

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_path = loop.run_until_complete(
            process_srt_to_mp3(srt_bytes, voice, rate, pitch, on_progress=on_progress)
        )
        with open(result_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(result_path)
        save_job_file(job_id, "result.mp3", audio_bytes)
        write_job_status(
            job_id, status="done", progress=1.0,
            message="✅ ការបំប្លែងជោគជ័យ!", out_name=mp3_name,
        )
    except Exception as e:
        write_job_status(job_id, status="error", error=str(e))


# =====================================================================
# SESSION STATE INIT
# =====================================================================

if "glossary" not in st.session_state:
    st.session_state.glossary = {}  # {english: khmer}


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ការកំណត់")
    st.markdown("---")

    gemini_api_key = st.text_input(
        "🔑 Google Gemini API Key",
        "",
        type="password",
        placeholder="AIza...",
        help="យក Key ឥតគិតថ្លៃ: aistudio.google.com",
    )
    if gemini_api_key:
        st.success("✅ API Key បានបញ្ចូល")
    else:
        st.info(
            "**យក API Key ឥតគិតថ្លៃ:**\n\n"
            "➜ [aistudio.google.com](https://aistudio.google.com)\n\n"
            "ចុច **Get API Key** → **Create API Key**"
        )

    st.markdown("---")
    selected_model = st.selectbox(
        "🤖 ជ្រើស AI Model",
        GEMINI_MODELS,
        index=0,
        help="ប្រសិនបើ model ណាមួយ error 503 (server busy) — ប្ដូរ model ផ្សេង។ gemini-3.5-flash ជា model ថ្មីបំផុត (May 2026)。",
    )
    model_labels = {
        "gemini-3.5-flash":               "Gemini 3.5 Flash ★★ (ចុងក្រោម · May 2026 · ល្អបំផុតសម្រាប់ Agentic & Coding)",
        "gemini-3.1-pro-preview":         "Gemini 3.1 Pro Preview ★ (Best Reasoning · 77.1% ARC-AGI-2)",
        "gemini-3.1-flash-lite":          "Gemini 3.1 Flash-Lite ✦ (ថោក · Translation-optimised)",
        "gemini-3-flash-preview":         "Gemini 3 Flash Preview",
        "gemini-robotics-er-1.6-preview": "Gemini Robotics-ER 1.6 Preview 🤖 (Spatial & Physical Reasoning · Apr 2026)",
        "gemini-2.5-pro":                 "Gemini 2.5 Pro (Advanced Reasoning & Coding)",
        "gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash Preview (Best Quality)",
        "gemini-2.5-flash":               "Gemini 2.5 Flash",
        "gemini-2.5-flash-lite":          "Gemini 2.5 Flash-Lite (Fastest · Cheapest)",
    }
    st.caption(f"✅ {model_labels.get(selected_model, selected_model)}")
    st.info("⚡ Auto-retry 5 ដង ប្រសិនបើ server busy (503)")
    st.success("🛰️ ការងាររត់នៅ server — ប្តូរ tab/app បានដោយសុវត្ថិភាព")

    # ── Glossary Panel ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📖 Series Glossary")
    st.caption("ទុក Glossary ឱ្យ consistent រវាង EP ទាំងអស់")

    glossary_count = len(st.session_state.glossary)
    if glossary_count > 0:
        st.success(f"✅ {glossary_count} ពាក្យក្នុង Glossary")
    else:
        st.warning("⚠️ Glossary ទទេ — បកប្រែ EP1 ជាមុន")

    # Import glossary from JSON
    gl_upload = st.file_uploader(
        "📂 Load Glossary (JSON)",
        type=["json"],
        key="glossary_uploader",
        help="Load glossary ពី EP មុន ដើម្បីឱ្យ EP ថ្មីប្រើពាក្យដ៏ដែល",
    )
    if gl_upload:
        try:
            loaded = json.loads(gl_upload.read().decode("utf-8"))
            if isinstance(loaded, dict):
                # Merge: loaded entries win (they're manually curated)
                st.session_state.glossary = {**st.session_state.glossary, **loaded}
                st.success(f"✅ Loaded {len(loaded)} entries")
            else:
                st.error("❌ JSON format ខុស — ត្រូវជា {{en: kh, ...}}")
        except Exception as e:
            st.error(f"❌ Load ខុស: {e}")

    # Export glossary
    if st.session_state.glossary:
        gl_json = json.dumps(
            st.session_state.glossary, ensure_ascii=False, indent=2
        ).encode("utf-8")
        st.download_button(
            "💾 Save Glossary (JSON)",
            data=gl_json,
            file_name="series_glossary.json",
            mime="application/json",
            help="Save ហើយ Load ឡើងវិញ នៅ EP បន្ទាប់",
        )

    if st.button("🗑️ Clear Glossary", key="clear_glossary"):
        st.session_state.glossary = {}
        st.rerun()

    # Show/edit glossary entries
    if st.session_state.glossary:
        with st.expander(f"👁️ មើល / កែ Glossary ({glossary_count} ពាក្យ)"):
            entries = list(st.session_state.glossary.items())
            to_delete = []
            for eng, kh in entries:
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    st.text(eng)
                with col2:
                    new_kh = st.text_input("", kh, key=f"gl_{eng}", label_visibility="collapsed")
                    if new_kh != kh:
                        st.session_state.glossary[eng] = new_kh
                with col3:
                    if st.button("✕", key=f"del_{eng}"):
                        to_delete.append(eng)
            for k in to_delete:
                del st.session_state.glossary[k]
                st.rerun()

    # Manual add entry
    with st.expander("➕ បន្ថែមពាក្យដៃ"):
        new_en = st.text_input("English", key="add_en", placeholder="e.g. Arthur")
        new_kh = st.text_input("ខ្មែរ", key="add_kh", placeholder="e.g. អាតុ")
        if st.button("➕ Add", key="btn_add_gl"):
            if new_en and new_kh:
                st.session_state.glossary[new_en] = new_kh
                st.success(f"✅ Added: {new_en} → {new_kh}")
                st.rerun()
            else:
                st.warning("⚠️ បំពេញ English និង ខ្មែរ")

    st.markdown("---")
    st.caption("SRT Tool Suite · Gemini 3.5 Flash / 3.1 Pro + Edge TTS")


# ── Title ────────────────────────────────────────────────────────────
st.title("🎙️ SRT Tool Suite")
st.caption("បកប្រែ SRT EN→KH  ·  បំប្លែងទៅជាសំឡេង MP3")

# ── Glossary Status Banner ───────────────────────────────────────────
if st.session_state.glossary:
    st.success(
        f"📖 **Series Glossary សកម្ម** — {len(st.session_state.glossary)} ពាក្យ "
        f"នឹងត្រូវបានប្រើ consistent រៀងរាល់ EP ។ "
        f"Save Glossary ក្នុង Sidebar បន្ទាប់ពីបកប្រែ EP ណាមួយ!"
    )
else:
    st.info(
        "📖 **Glossary ទទេ** — បកប្រែ EP1 ជាមុន, "
        "បន្ទាប់មក Glossary នឹង auto-extract ហើយ Save ទុក → Load ឡើងវិញ នៅ EP2, EP3..."
    )

tab_translate, tab_tts = st.tabs(["🌐 បកប្រែ  EN → KH", "🎧 SRT → MP3"])


# =====================================================================
# TAB 1 — TRANSLATION
# =====================================================================

with tab_translate:
    st.subheader("🌐 បកប្រែ SRT EN → KH")

    with st.expander("ℹ️ របៀបប្រើ Glossary System", expanded=False):
        st.markdown(
            """
### 🔄 ការប្រើ Glossary ដើម្បីឱ្យ EP ទាំងអស់ consistent

| ជំហាន | ការប្រព្រឹត្ត |
|--------|----------------|
| **EP1** | Upload SRT → បកប្រែ → Glossary auto-extract → **💾 Save Glossary** |
| **EP2** | **📂 Load Glossary** (ពី Sidebar) → Upload SRT EP2 → បកប្រែ → **💾 Save Glossary** |
| **EP3+** | ធ្វើដូចគ្នា — Glossary ចូររក្សា consistent ពាក្យ ✅ |

### 🏷️ Features
| | |
|--|--|
| 🎵 ♪ ♫ | លុប music lines |
| 🏷️ Tags | `{\\an8}` · `<i>` · `[...]` → auto-remove |
| 📖 Glossary | ឈ្មោះ + ពាក្យ consistent រៀងរាល់ EP |
| ➕ Manual Add | បន្ថែមពាក្យដៃ ឬ កែ Glossary |
| 🛰️ Server job | ដំណើរការនៅ server — ប្តូរ tab/app បានដោយសុវត្ថិភាព |
"""
        )

    job_translate_id = st.query_params.get("job_translate")

    if job_translate_id:
        status = read_job_status(job_translate_id)

        if status is None:
            st.warning("⚠️ រកមិនឃើញការងារនេះទេ (server restart ឬផុតកំណត់)។ សូមចាប់ផ្តើមឡើងវិញ។")
            if st.button("🆕 ការងារថ្មី", key="reset_translate_missing"):
                del st.query_params["job_translate"]
                st.rerun()

        elif status.get("status") == "running":
            st.info(
                "🔄 កំពុងបកប្រែនៅលើ server... \n\n"
                "✅ អ្នកអាចប្តូរទៅ tab ឬ app ផ្សេងបានដោយសុវត្ថិភាព — "
                "ការងារនឹងបន្តដំណើរការ ហើយលទ្ធផលនឹងបង្ហាញពេលអ្នកត្រឡប់មកវិញ។"
            )
            st.progress(status.get("progress", 0.0))
            st.caption(status.get("message", ""))
            # Poll for updates every 3s. This also means: even if iOS forces
            # a real page reload while we're "waiting", the result is the same —
            # we just re-read status.json and show progress again.
            st.markdown('<meta http-equiv="refresh" content="3">', unsafe_allow_html=True)

        elif status.get("status") == "error":
            st.error(f"❌ មានបញ្ហា: {status.get('error', 'Unknown error')}")
            if st.button("🔁 ព្យាយាមម្តងទៀត", key="retry_translate"):
                del st.query_params["job_translate"]
                st.rerun()

        elif status.get("status") == "done":
            out_name = status.get("out_name", "result_KH.srt")
            glossary_count = status.get("glossary_count", 0)
            result_bytes = load_job_file(job_translate_id, "result.srt")
            glossary_bytes = load_job_file(job_translate_id, "glossary.json")

            if glossary_bytes:
                try:
                    st.session_state.glossary = json.loads(glossary_bytes.decode("utf-8"))
                except Exception:
                    pass

            st.success(
                f"🎉 ការបកប្រែជោគជ័យ! "
                f"Glossary ឥឡូវនេះមាន **{glossary_count} ពាក្យ** — "
                f"Save វាក្នុង Sidebar ដើម្បីប្រើ EP បន្ទាប់!"
            )
            col_dl, col_gl = st.columns(2)
            with col_dl:
                if result_bytes:
                    st.download_button(
                        label="📥 ទាញយក SRT ខ្មែរ",
                        data=result_bytes,
                        file_name=out_name,
                        mime="text/plain",
                        key="dl_srt_result",
                    )
            with col_gl:
                if glossary_bytes:
                    st.download_button(
                        label="💾 Save Glossary សម្រាប់ EP បន្ទាប់",
                        data=glossary_bytes,
                        file_name="series_glossary.json",
                        mime="application/json",
                        key="dl_glossary_result",
                    )

            if st.button("🆕 បកប្រែ EP ផ្សេងទៀត", key="new_translate_job"):
                del st.query_params["job_translate"]
                st.rerun()

    else:
        uploaded_srt_raw = st.file_uploader(
            "📂 ជ្រើស .srt (English)",
            key="translate_uploader",
        )

        uploaded_srt = None
        if uploaded_srt_raw is not None:
            if is_valid_srt_filename(uploaded_srt_raw.name):
                uploaded_srt = uploaded_srt_raw
            else:
                st.error(
                    f"❌ ឯកសារ **{uploaded_srt_raw.name}** មិនមែនជា `.srt` ទេ។ "
                    "សូមជ្រើស file ដែលមាន extension `.srt`។"
                )

        if uploaded_srt:
            base = os.path.splitext(uploaded_srt.name)[0]
            out_name = f"{base}_KH.srt"
            st.info(f"📄 **{uploaded_srt.name}**  →  **{out_name}**")

            st.markdown("**🏷️ ជ្រើសរើស Tags ដែលចង់លុប:**")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                opt_ass = st.checkbox("`{\\an8}` ASS/SSA tags", value=True, key="opt_ass")
            with tc2:
                opt_html = st.checkbox("`<i>` `</i>` HTML tags", value=True, key="opt_html")
            with tc3:
                opt_brackets = st.checkbox("`[...]` Bracket text", value=True, key="opt_brackets")

            if st.button("🔄 ចាប់ផ្តើមបកប្រែ", type="primary", key="btn_translate"):
                key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
                if not key:
                    st.error(
                        "❌ សូមដាក់ **Google Gemini API Key** ក្នុង Sidebar។\n\n"
                        "យក Key ឥតគិតថ្លៃ: https://aistudio.google.com"
                    )
                    st.stop()

                job_id = new_job_id()
                write_job_status(
                    job_id, status="running", progress=0.0,
                    message="⏳ កំពុងចាប់ផ្តើម...",
                )
                t = threading.Thread(
                    target=_run_translate_job,
                    args=(
                        job_id, uploaded_srt.read(), key,
                        opt_ass, opt_html, opt_brackets,
                        selected_model, dict(st.session_state.glossary), out_name,
                    ),
                    daemon=True,
                )
                t.start()
                st.query_params["job_translate"] = job_id
                st.rerun()


# =====================================================================
# TAB 2 — TTS
# =====================================================================

with tab_tts:
    st.subheader("🎧 SRT → សំឡេង MP3 (Audio Sync)")

    job_tts_id = st.query_params.get("job_tts")

    if job_tts_id:
        status = read_job_status(job_tts_id)

        if status is None:
            st.warning("⚠️ រកមិនឃើញការងារនេះទេ (server restart ឬផុតកំណត់)។ សូមចាប់ផ្តើមឡើងវិញ។")
            if st.button("🆕 ការងារថ្មី", key="reset_tts_missing"):
                del st.query_params["job_tts"]
                st.rerun()

        elif status.get("status") == "running":
            st.info(
                "🔄 កំពុងបំប្លែងនៅលើ server... \n\n"
                "✅ អ្នកអាចប្តូរទៅ tab ឬ app ផ្សេងបានដោយសុវត្ថិភាព — "
                "ការងារនឹងបន្តដំណើរការ ហើយលទ្ធផលនឹងបង្ហាញពេលអ្នកត្រឡប់មកវិញ។"
            )
            st.progress(status.get("progress", 0.0))
            st.markdown('<meta http-equiv="refresh" content="3">', unsafe_allow_html=True)

        elif status.get("status") == "error":
            st.error(f"❌ មានបញ្ហា: {status.get('error', 'Unknown error')}")
            if st.button("🔁 ព្យាយាមម្តងទៀត", key="retry_tts"):
                del st.query_params["job_tts"]
                st.rerun()

        elif status.get("status") == "done":
            mp3_name = status.get("out_name", "result.mp3")
            audio_bytes = load_job_file(job_tts_id, "result.mp3")
            st.success("ការបំប្លែងជោគជ័យ! សំឡេងដើរទាន់អក្សរហើយ!")
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button(
                    "📥 ទាញយក MP3", audio_bytes, file_name=mp3_name, key="dl_mp3_result",
                )
            if st.button("🆕 បំប្លែងឯកសារផ្សេងទៀត", key="new_tts_job"):
                del st.query_params["job_tts"]
                st.rerun()

    else:
        tts_file_raw = st.file_uploader(
            "📂 ជ្រើស .srt (ខ្មែរ ឬ ភាសាដែលចង់ស្តាប់)",
            key="tts_uploader",
        )

        tts_file = None
        if tts_file_raw is not None:
            if is_valid_srt_filename(tts_file_raw.name):
                tts_file = tts_file_raw
            else:
                st.error(
                    f"❌ ឯកសារ **{tts_file_raw.name}** មិនមែនជា `.srt` ទេ។ "
                    "សូមជ្រើស file ដែលមាន extension `.srt`។"
                )

        c1, c2, c3 = st.columns(3)
        with c1:
            voice_opt = st.selectbox(
                "សំឡេង",
                ["km-KH-PisethNeural (ប្រុស)", "km-KH-SreymomNeural (ស្រី)"],
                index=1,
            )
        with c2:
            speed = st.slider("ល្បឿន (%)", -50, 100, 45, step=5)
        with c3:
            pitch_val = st.slider("Pitch (Hz)", -50, 50, 18, step=1)

        voice_id  = voice_opt.split(" ")[0]
        rate_str  = f"{speed:+d}%"
        pitch_str = f"{pitch_val:+d}Hz"

        if tts_file:
            base_tts = os.path.splitext(tts_file.name)[0]
            mp3_name = f"{base_tts}.mp3"

            if st.button("🎙️ ចាប់ផ្តើមបំប្លែង", type="primary", key="btn_tts"):
                job_id = new_job_id()
                write_job_status(
                    job_id, status="running", progress=0.0,
                    message="⏳ កំពុងបំប្លែង...",
                )
                t = threading.Thread(
                    target=_run_tts_job,
                    args=(job_id, tts_file.read(), voice_id, rate_str, pitch_str, mp3_name),
                    daemon=True,
                )
                t.start()
                st.query_params["job_tts"] = job_id
                st.rerun()
