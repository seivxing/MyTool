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


# =====================================================================
# TTS FUNCTIONS
# =====================================================================

async def process_srt_to_mp3(srt_content, voice, rate, pitch):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:
        tmp_srt.write(srt_content)
        tmp_srt_path = tmp_srt.name

    subs = pysubs2.load(tmp_srt_path, encoding="utf-8")
    if not subs:
        raise ValueError("ឯកសារ SRT របស់អ្នកទទេស្អាត ឬខូច។")

    base_duration = subs[-1].end + 30000
    combined_audio = AudioSegment.silent(duration=base_duration)
    progress_bar = st.progress(0)
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
        progress_bar.progress((i + 1) / total_lines)

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
                                       english_lines: list[str],
                                       khmer_lines: list[str],
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


def translate_srt(srt_bytes: bytes, api_key: str, progress_bar, status_text,
                  remove_ass: bool = True, remove_html: bool = True,
                  remove_brackets: bool = True, model: str = GEMINI_MODEL,
                  glossary: dict = None) -> tuple[bytes, dict]:
    """
    Full pipeline: parse → filter music → clean → batch-translate → SRT bytes.
    Returns (srt_bytes, updated_glossary).
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError(
            "google-genai មិនទាន់ install ទេ។ សូម run:\n"
            "pip install google-genai"
        )

    if glossary is None:
        glossary = {}

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

        status_text.text(
            f"⏳ កំពុងបកប្រែ {start + 1}–{end} / {total} បន្ទាត់  ({model})"
        )
        result = translate_batch(client, batch, model, glossary=glossary)
        all_translations.update(result)
        progress_bar.progress((b + 1) / n_batches)

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
    status_text.text("🔍 កំពុងទាញ Glossary ចេញពី EP នេះ...")
    updated_glossary = extract_glossary_from_translation(
        client, model, english_lines, khmer_lines, glossary
    )

    status_text.text("✅ ការបកប្រែបានសម្រេច!")
    return out.to_string("srt").encode("utf-8"), updated_glossary


# =====================================================================
# SESSION STATE INIT
# =====================================================================

if "glossary" not in st.session_state:
    st.session_state.glossary = {}  # {english: khmer}
if "translation_result" not in st.session_state:
    st.session_state.translation_result = None  # (bytes, out_name, updated_glossary)
if "tts_result" not in st.session_state:
    st.session_state.tts_result = None  # (bytes, mp3_name)


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
"""
        )

    uploaded_srt = st.file_uploader(
        "📂 ជ្រើស .srt (English)",
        type=["srt"],
        key="translate_uploader",
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
            # Clear previous result when starting new translation
            st.session_state.translation_result = None
            key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
            if not key:
                st.error(
                    "❌ សូមដាក់ **Google Gemini API Key** ក្នុង Sidebar។\n\n"
                    "យក Key ឥតគិតថ្លៃ: https://aistudio.google.com"
                )
                st.stop()

            pb = st.progress(0)
            status = st.empty()

            try:
                result_bytes, updated_glossary = translate_srt(
                    uploaded_srt.read(),
                    key,
                    pb,
                    status,
                    remove_ass=opt_ass,
                    remove_html=opt_html,
                    remove_brackets=opt_brackets,
                    model=selected_model,
                    glossary=st.session_state.glossary,
                )

                # Update session glossary and store result persistently
                st.session_state.glossary = updated_glossary
                st.session_state.translation_result = (result_bytes, out_name, updated_glossary)

            except Exception as exc:
                err = str(exc)
                if "API_KEY_INVALID" in err or "invalid" in err.lower():
                    st.error("❌ API Key មិនត្រឹមត្រូវ។ សូមពិនិត្យម្តងទៀត។")
                elif "google-genai" in err:
                    st.error(f"❌ {err}")
                    st.code("pip install google-genai", language="bash")
                else:
                    st.error(f"❌ មានបញ្ហា: {exc}")

        # Show download buttons persistently from session_state
        if st.session_state.translation_result is not None:
            result_bytes, saved_out_name, updated_glossary = st.session_state.translation_result
            st.success(
                f"🎉 ការបកប្រែជោគជ័យ! "
                f"Glossary ឥឡូវនេះមាន **{len(updated_glossary)} ពាក្យ** — "
                f"Save វាក្នុង Sidebar ដើម្បីប្រើ EP បន្ទាប់!"
            )
            col_dl, col_gl = st.columns(2)
            with col_dl:
                st.download_button(
                    label="📥 ទាញយក SRT ខ្មែរ",
                    data=result_bytes,
                    file_name=saved_out_name,
                    mime="text/plain",
                    key="dl_srt_result",
                )
            with col_gl:
                gl_json = json.dumps(
                    updated_glossary, ensure_ascii=False, indent=2
                ).encode("utf-8")
                st.download_button(
                    label="💾 Save Glossary សម្រាប់ EP បន្ទាប់",
                    data=gl_json,
                    file_name="series_glossary.json",
                    mime="application/json",
                    key="dl_glossary_result",
                )


# =====================================================================
# TAB 2 — TTS
# =====================================================================

with tab_tts:
    st.subheader("🎧 SRT → សំឡេង MP3 (Audio Sync)")

    tts_file = st.file_uploader(
        "📂 ជ្រើស .srt (ខ្មែរ ឬ ភាសាដែលចង់ស្តាប់)",
        type=["srt"],
        key="tts_uploader",
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
            # Clear previous result when starting new conversion
            st.session_state.tts_result = None
            with st.spinner("កំពុងបំប្លែង..."):
                try:
                    srt_bytes = tts_file.read()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result_path = loop.run_until_complete(
                        process_srt_to_mp3(srt_bytes, voice_id, rate_str, pitch_str)
                    )
                    with open(result_path, "rb") as f:
                        audio_bytes = f.read()
                    os.remove(result_path)
                    # Store result persistently
                    st.session_state.tts_result = (audio_bytes, mp3_name)
                except Exception as exc:
                    import traceback
                    tb = traceback.format_exc()
                    # Extract line number from traceback
                    line_match = re.findall(r'line (\d+)', tb)
                    line_info = f" (line {line_match[-1]})" if line_match else ""
                    err_msg = str(exc)

                    if "No audio was received" in err_msg or "no audio" in err_msg.lower():
                        st.error(
                            f"❌ មានបញ្ហា{line_info}: No audio was received. "
                            f"Please verify that your parameters are correct.\n\n"
                            f"**សាកល្បង:**\n"
                            f"- ល្បឿន (Rate): `{rate_str}` — សាកល្បងប្ដូរទៅ `+0%`\n"
                            f"- Pitch: `{pitch_str}` — សាកល្បងប្ដូរទៅ `+0Hz`\n"
                            f"- សំឡេង: `{voice_id}` — ប្ដូរទៅ voice ម្យ៉ាងទៀត\n"
                            f"- ពិនិត្យ internet connection ទៅ Microsoft Edge TTS server"
                        )
                    else:
                        st.error(f"❌ មានបញ្ហា{line_info}: {exc}")

                    with st.expander("🔍 Technical Details (Traceback)"):
                        st.code(tb, language="python")

        # Show audio player and download button persistently from session_state
        if st.session_state.tts_result is not None:
            audio_bytes, saved_mp3_name = st.session_state.tts_result
            st.success("ការបំប្លែងជោគជ័យ! សំឡេងដើរទាន់អក្សរហើយ!")
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                "📥 ទាញយក MP3",
                audio_bytes,
                file_name=saved_mp3_name,
                key="dl_mp3_result",
            )
