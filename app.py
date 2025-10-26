import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import html
import random
from pathlib import Path

# -----------------------------
# Konfigurasi path model
MODEL_DIR = "best_model"  # ganti jika folder model berbeda

# label mapping yang diminta user
LABEL_MAP = {
    0: "All-or-nothing",
    1: "Discounting the positives",
    2: "Emotional Reasoning",
    3: "Jumping to Conclusions",
    4: "Labeling",
    5: "Magnification or Minimization",
    6: "Mental filter",
    7: "No Distortion",
    8: "Overgeneralization",
    9: "Personalization and Blame",
    10: "Should statement",
    11: "Should statements"
}

LABEL_EXPLANATIONS = {
    0: {
        "title": "All-or-nothing (Pola pikir hitam-putih)",
        "desc": "Kamu melihat sesuatu hanya dalam dua sisi: sempurna atau gagal total. Tidak ada ruang di tengah. Misalnya, kalau kamu tidak berhasil 100%, kamu merasa semuanya sia-sia.",
        "example": "“Saya selalu gagal dalam segala hal.”",
        "suggestion": "Ingat, hidup tidak selalu hitam dan putih. Coba lihat bagian mana yang sebenarnya berjalan cukup baik, meskipun tidak sempurna."
    },
    1: {
        "title": "Discounting the positives (Meremehkan hal positif)",
        "desc": "Kamu cenderung mengabaikan hal-hal baik yang sudah kamu lakukan. Walaupun kamu berhasil, kamu merasa itu tidak berarti atau hanya kebetulan.",
        "example": "“Itu cuma keberuntungan, bukan karena saya mampu.”",
        "suggestion": "Berikan penghargaan kecil pada dirimu. Sekecil apa pun pencapaianmu, itu tetap berharga dan menunjukkan usaha nyata."
    },
    2: {
        "title": "Emotional Reasoning (Menganggap perasaan sebagai fakta)",
        "desc": "Kamu merasa apa yang kamu rasakan pasti benar. Misalnya, karena kamu merasa gagal, maka kamu percaya kamu memang gagal — padahal belum tentu begitu.",
        "example": "“Saya merasa tidak kompeten, jadi pasti saya memang tidak kompeten.”",
        "suggestion": "Perasaan itu penting, tapi tidak selalu sama dengan kenyataan. Coba tanyakan pada diri sendiri: ‘Apa buktinya?’"
    },
    3: {
        "title": "Jumping to Conclusions (Cepat menyimpulkan tanpa bukti)",
        "desc": "Kamu langsung menarik kesimpulan tanpa cukup bukti. Kadang kamu merasa tahu apa yang orang lain pikirkan, atau yakin tahu apa yang akan terjadi.",
        "example": "“Dia diam, pasti dia marah padaku.”",
        "suggestion": "Sebelum menyimpulkan, coba periksa dulu faktanya. Mungkin ada alasan lain yang belum kamu tahu."
    },
    4: {
        "title": "Labeling (Memberi label negatif pada diri sendiri atau orang lain)",
        "desc": "Kamu menilai dirimu atau orang lain dengan satu kata negatif berdasarkan satu kejadian. Misalnya, karena satu kesalahan kecil, kamu merasa ‘bodoh’.",
        "example": "“Saya bodoh.”",
        "suggestion": "Daripada memberi label, coba jelaskan perilakunya: ‘Saya salah menulis kata,’ bukan ‘Saya bodoh.’ Itu membantu pikiranmu lebih realistis."
    },
    5: {
        "title": "Magnification or Minimization (Melebih-lebihkan atau mengecilkan)",
        "desc": "Kamu membesar-besarkan hal buruk atau mengecilkan hal baik, sehingga pandanganmu jadi tidak seimbang.",
        "example": "“Satu kesalahan = bencana total.”",
        "suggestion": "Tanyakan pada diri sendiri: seberapa besar pengaruh hal ini sebenarnya, dari skala 1 sampai 10?"
    },
    6: {
        "title": "Mental filter (Hanya fokus pada yang negatif)",
        "desc": "Kamu hanya memperhatikan hal-hal buruk dan lupa pada hal-hal baik yang juga terjadi.",
        "example": "“Orang hanya mengkritikku,” padahal ada juga yang memujimu.",
        "suggestion": "Coba tulis dua hal baik yang terjadi hari ini, sekecil apa pun. Itu bisa membantu menyeimbangkan pandanganmu."
    },
    7: {
        "title": "No Distortion (Tidak ada distorsi yang jelas)",
        "desc": "Model tidak menemukan tanda-tanda distorsi kognitif yang menonjol dalam teksmu. Artinya, pernyataanmu terdengar cukup netral dan realistis.",
        "example": "Pernyataan faktual tanpa kata-kata ekstrem atau berlebihan.",
        "suggestion": "Kalau kamu merasa masih ada hal yang mengganggu, coba tulis ulang dengan konteks berbeda untuk melihat sisi lainnya."
    },
    8: {
        "title": "Overgeneralization (Menarik kesimpulan berlebihan)",
        "desc": "Kamu menggunakan satu pengalaman untuk menyimpulkan semua hal serupa akan berakhir sama.",
        "example": "“Saya gagal sekali, berarti saya akan gagal terus.”",
        "suggestion": "Ingat, satu pengalaman tidak menentukan segalanya. Setiap kesempatan baru bisa membawa hasil yang berbeda."
    },
    9: {
        "title": "Personalization and Blame (Menyalahkan diri sendiri atau orang lain)",
        "desc": "Kamu merasa semua yang salah pasti karena dirimu, atau sebaliknya, menyalahkan orang lain tanpa melihat faktor lain.",
        "example": "“Semua ini salahku.” atau “Itu karena dia selalu begitu.”",
        "suggestion": "Coba pikirkan: faktor apa lagi yang mungkin berperan? Tidak semua hal sepenuhnya tanggung jawab satu orang."
    },
    10: {
        "title": "Should statement (Terlalu banyak ‘harus’)",
        "desc": "Kamu sering memberi aturan kaku pada diri sendiri atau orang lain dengan kata ‘harus’. Ini bisa membuatmu merasa bersalah atau kecewa jika tidak terpenuhi.",
        "example": "“Saya harus selalu sukses.”",
        "suggestion": "Ganti ‘harus’ dengan kata yang lebih lembut, seperti ‘Saya ingin’ atau ‘Akan lebih baik jika’."
    },
    11: {
        "title": "Should statements (Pola pikir penuh tuntutan)",
        "desc": "Mirip dengan sebelumnya, tapi lebih sering digunakan berulang-ulang. Kamu merasa ada banyak hal yang ‘seharusnya’ terjadi sesuai ekspektasimu.",
        "example": "“Mereka seharusnya tahu perasaanku.”",
        "suggestion": "Tidak semua orang bisa membaca pikiran kita. Coba ungkapkan kebutuhanmu secara langsung dan jelas."
    }
}

# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def device_of_model():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(text, tokenizer, model, device=None):
    if device is None:
        device = device_of_model()
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    if device.type == 'cuda':
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(probs.argmax())
    # try to get label name from model config
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        label_name = id2label.get(pred, LABEL_MAP.get(pred, str(pred)))
    else:
        label_name = LABEL_MAP.get(pred, str(pred))
    return pred, probs, label_name


# Simple leave-one-out importance scoring per token/word (works for short texts)
def importance_by_loo(text, tokenizer, model, base_prob=None, target_label=1):
    words = text.split()
    if len(words) == 0:
        return []
    if base_prob is None:
        _, probs, _ = predict(text, tokenizer, model)
        base_prob = probs[target_label] if target_label < len(probs) else probs.max()
    scores = []
    for i, _w in enumerate(words):
        # create text with word i removed
        new_words = words[:i] + ["[REMOVED]"] + words[i+1:]
        new_text = " ".join(new_words)
        _, probs_mask, _ = predict(new_text, tokenizer, model)
        prob = probs_mask[target_label] if target_label < len(probs_mask) else probs_mask.max()
        delta = base_prob - prob
        scores.append((i, words[i], float(delta)))
    # normalize scores
    max_abs = max(abs(s) for (_, _, s) in scores) or 1.0
    norm = [(i, w, s / max_abs) for (i, w, s) in scores]
    return norm


# Build highlighted HTML from importance scores
# membuat gradasi opacity sehingga kata "kuat" berwarna merah pekat, "lemah" meredup
def build_highlight_html(text, importance_scores):
    words = text.split()
    spans = []
    for i, w in enumerate(words):
        score = 0.0
        for tup in importance_scores:
            if tup[0] == i:
                score = tup[2]
                break
        intensity = min(max(score, -1.0), 1.0)
        safe_word = html.escape(w)
        if intensity > 0:
            # map 0..1 to alpha 0.08 .. 0.85 (lebih lembut pada nilai kecil)
            alpha = 0.08 + 0.77 * (intensity ** 1.2)
            style = f"background-color: rgba(255,0,0,{alpha}); padding:3px; border-radius:4px;"
        elif intensity < 0:
            # negative (menurunkan bukti) -> biarkan sangat redup/abu-abu ringan
            alpha = 0.04 + 0.3 * (abs(intensity) ** 1.0)
            style = f"background-color: rgba(200,200,200,{alpha}); padding:3px; border-radius:4px;"
        else:
            style = ""
        spans.append(f"<span style=\"{style}\">{safe_word}</span>")
    html_str = " ".join(spans)
    return html_str

# -----------------------------
# Streamlit App UI (no sidebar, lebih user-friendly, pakai emoji)
st.set_page_config(page_title="Deteksi Distorsi Kognitif (ID)", layout="wide")
st.markdown("# 🧠 Deteksi Distorsi Kognitif — Bahasa Indonesia (MVP)")
st.caption("Masukkan teks singkat (curhat / catatan harian). Hasil informatif — bukan diagnosis.")

with st.spinner("Memuat model... ⏳"):
    tokenizer, model = load_model_and_tokenizer()

# top info expander (menggantikan sidebar)
with st.expander("ℹ️ Tentang aplikasi dan cara pakai", expanded=False):
    st.write("Versi MVP — deteksi distorsi kognitif. Jangan gunakan sebagai pengganti profesional kesehatan.")
    st.write("Letakkan model pada folder 'best_model' (config.json, model.safetensors, tokenizer.json, dll.)")
    st.markdown("**Cara jalankan**:\n1. Pasang dependency: pip install -r requirements.txt\n2. Jalankan: streamlit run streamlit_cd_app_improved.py")

# Input area (gunakan session_state key agar contoh/bersihkan bekerja mulus)
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = "Saya selalu gagal dalam segala hal"

text_input = st.text_area("Tulis teksmu di sini ✍️", value=st.session_state['input_text'], height=180, key='input_text')

def clear_input():
    st.session_state['input_text'] = ""

def set_example():
    examples = [
        "Saya selalu gagal dalam segala hal",
        "Tidak ada yang peduli dengan saya",
        "Aku pasti akan membuat kesalahan lagi",
        "Semuanya salah karena aku",
        "Orang lain selalu lebih baik dariku",
        "Aku tidak akan pernah bisa berubah",
        "Semua usahaku sia-sia",
        "Aku tidak pantas bahagia"
    ]
    st.session_state['input_text'] = random.choice(examples)

cols_top = st.columns([1,1,1,2])
with cols_top[0]:
    st.button("🧹 Bersihkan", on_click=clear_input)
with cols_top[1]:
    st.button("🔄 Contoh", on_click=set_example)
with cols_top[2]:
    analyze_btn = st.button("🔎 Analisis")

st.markdown("---")

if analyze_btn and text_input.strip():
    with st.spinner("Menganalisis... 🔎"):
        device = device_of_model()
        pred, probs, label_name = predict(text_input, tokenizer, model, device=device)
        # Nice label from provided LABEL_MAP
        mapped_label = LABEL_MAP.get(pred, label_name)
        prob_str = ", ".join([f"{LABEL_MAP.get(i, str(i))}: {p:.3f}" for i, p in enumerate(probs)])
        # choose target label for highlighting
        if len(probs) == 2:
            target_label = 1
        else:
            target_label = pred
        base_prob = probs[target_label] if target_label < len(probs) else probs.max()
        imp = importance_by_loo(text_input, tokenizer, model, base_prob=base_prob, target_label=target_label)
        html_highlight = build_highlight_html(text_input, imp)

    # Results layout — lebih ramah
    st.markdown(f"### ✅ Hasil Prediksi: **{mapped_label}** (#{pred})")
    cols = st.columns([2,3])
    with cols[0]:
        st.write("**Probabilitas (ringkas per-label):**")
        st.write(prob_str)
        st.write("---")
        st.markdown("### 🧩 Penjelasan Hasil Prediksi")

        explanation = LABEL_EXPLANATIONS.get(pred)

        if explanation:
            # Pilih ikon sesuai kategori
            icon_map = {
                "All-or-nothing": "⚫⚪",
                "Discounting the positives": "➖✨",
                "Emotional Reasoning": "💭💔",
                "Jumping to Conclusions": "🏃‍♂️💨",
                "Labeling": "🏷️",
                "Magnification or Minimization": "🔍",
                "Mental filter": "🪞",
                "No Distortion": "🌿",
                "Overgeneralization": "🔄",
                "Personalization and Blame": "🙇‍♂️",
                "Should statement": "📋",
                "Should statements": "📋📋"
            }
            icon = icon_map.get(explanation['title'].split(" ")[0], "🧠")

            st.markdown(f"""
            <div style="padding: 15px; border-radius: 12px; background-color: #f8f9fa; border: 1px solid #ddd;">
                <h4 style="margin-bottom:8px;">{icon} <b>{explanation['title']}</b></h4>
                <p style="margin-top:0; margin-bottom:10px;">{explanation['desc']}</p>
                <p style="font-style:italic; color:#555;">💬 <b>Contoh:</b> “{explanation['example']}”</p>
                <p style="margin-top:10px;">💡 <b>Saran singkat:</b> {explanation['suggestion']}</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info(f"📘 Hasil prediksi: **{mapped_label}** — penjelasan detail belum tersedia untuk label ini.")
    with cols[1]:
        st.markdown("#### 🔥 Highlight kata/frasal yang berkontribusi")
        st.components.v1.html(f"<div style='font-size:16px; line-height:1.6'>{html_highlight}</div>", height=220)

    st.markdown("---")
    st.markdown("#### 🔧 Debug / Info")
    st.write({
        "pred_label_index": int(pred),
        "pred_label_name": mapped_label,
        "probs": {LABEL_MAP.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    })

# footer kecil
st.markdown("Made with ❤️ — Aplikasi informatif untuk bantu refleksi pribadi. Jangan jadikan pengganti profesional.")
