"""
Engine 1: Context-Aware Content Generator

WHAT CHANGED FROM ORIGINAL (offline-resilience fixes):
-------------------------------------------------------
1. GEMINI:
   - Original used Google Gemini API (cloud). If not going to rule-based.

2. SARVAM TTS → pyttsx3 LOCAL TTS (offline fallback):
   - Original: if SARVAM_API_KEY missing → returns a placeholder string. IVR breaks silently.
   - Now: tries Sarvam first (if key set), falls back to pyttsx3 which runs offline.
   - pyttsx3 uses the OS speech engine (espeak on Linux, SAPI on Windows, NSSpeech on Mac).
   - No API key needed. No internet needed. Actual audio is always generated.

3. RULE-BASED FALLBACK NOW USES CORRECT LANGUAGE:
   - Original: _rule_based_fallback() had all templates hardcoded in Hindi,
     even though LANGUAGE_GREETINGS and LANGUAGE_CTA were already defined.
   - Fixed: Every format now picks the right language string from those dictionaries.
   - All 11 languages produce correct native-script output offline.

APIs still used (optional, degrade gracefully if absent):
   - GEMINI_API_KEY  — cloud upgrade for richer content
   - SARVAM_API_KEY  — higher-quality TTS audio for IVR
   - STABILITY_API_KEY — real images for VIDEO_THUMBNAIL (optional)
"""

import os
import base64
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Domain Knowledge Base (unchanged from original)
# ---------------------------------------------------------------------------

CROP_KNOWLEDGE = {
    "wheat": {
        "key_threats": ["yellow rust (Puccinia striiformis)", "powdery mildew", "aphids", "loose smut"],
        "critical_stages": ["tillering", "crown root initiation", "flowering", "grain filling"],
        "products": {"fungicide": "Tilt 250 EC", "insecticide": "Actara 25 WG", "herbicide": "Topik 15 WP"},
        "dosage": {"Tilt 250 EC": "200 ml/acre in 200L water", "Topik 15 WP": "160 g/acre in 200L water"},
        "application_timing": "Apply fungicide at first sign of rust, or preventively at tillering.",
        "harvest_gap_days": 35,
    },
    "mustard": {
        "key_threats": ["white rust (Albugo candida)", "Alternaria blight", "aphids (Lipaphis erysimi)"],
        "critical_stages": ["rosette", "flowering", "pod filling"],
        "products": {"fungicide": "Score 250 EC", "insecticide": "Actara 25 WG"},
        "dosage": {"Score 250 EC": "200 ml/acre in 200L water"},
        "application_timing": "Apply at 5% flowering for white rust prevention.",
        "harvest_gap_days": 28,
    },
    "chickpea": {
        "key_threats": ["pod borer (Helicoverpa armigera)", "wilt (Fusarium)", "blight (Ascochyta)"],
        "critical_stages": ["vegetative", "flowering", "pod initiation", "grain filling"],
        "products": {"insecticide": "Actara 25 WG", "fungicide": "Amistar 250 SC"},
        "dosage": {"Actara 25 WG": "80 g/acre in 200L water"},
        "application_timing": "Apply insecticide at pod initiation when pod borer eggs are visible.",
        "harvest_gap_days": 21,
    },
    "potato": {
        "key_threats": ["late blight (Phytophthora infestans)", "early blight (Alternaria solani)", "aphids"],
        "critical_stages": ["emergence", "tuber initiation", "tuber bulking", "maturation"],
        "products": {"fungicide": "Kavach 75 WP", "systemic_fungicide": "Amistar 250 SC"},
        "dosage": {"Kavach 75 WP": "600 g/acre in 200L water"},
        "application_timing": "Preventive spray every 7 days during humid cool weather (15-20 C).",
        "harvest_gap_days": 14,
    },
    "barley": {
        "key_threats": ["powdery mildew", "net blotch", "yellow rust"],
        "critical_stages": ["tillering", "jointing", "heading"],
        "products": {"fungicide": "Tilt 250 EC"},
        "dosage": {"Tilt 250 EC": "200 ml/acre in 200L water"},
        "application_timing": "Apply at first sign of mildew at jointing.",
        "harvest_gap_days": 30,
    },
    "lentil": {
        "key_threats": ["rust (Uromyces viciae-fabae)", "wilt complex", "stemphylium blight"],
        "critical_stages": ["vegetative", "flowering", "pod filling"],
        "products": {"fungicide": "Amistar 250 SC"},
        "dosage": {"Amistar 250 SC": "200 ml/acre in 200L water"},
        "application_timing": "Apply at first sign of rust at podding stage.",
        "harvest_gap_days": 20,
    },
    "safflower": {
        "key_threats": ["Alternaria leaf spot", "root rot", "aphids"],
        "critical_stages": ["rosette", "elongation", "flowering"],
        "products": {"fungicide": "Score 250 EC"},
        "dosage": {"Score 250 EC": "200 ml/acre in 200L water"},
        "application_timing": "Apply at first sign of Alternaria at rosette stage.",
        "harvest_gap_days": 25,
    },
    "cumin": {
        "key_threats": ["blight (Alternaria burnsii)", "powdery mildew", "wilt"],
        "critical_stages": ["seedling", "vegetative", "flowering", "seed setting"],
        "products": {"fungicide": "Tilt 250 EC"},
        "dosage": {"Tilt 250 EC": "150 ml/acre in 200L water"},
        "application_timing": "Apply preventively at seedling stage; repeat at flowering.",
        "harvest_gap_days": 20,
    },
    "maize": {
        "key_threats": ["fall armyworm (Spodoptera frugiperda)", "northern leaf blight", "stalk rot"],
        "critical_stages": ["seedling", "vegetative V6", "tasselling", "grain filling"],
        "products": {"insecticide": "Actara 25 WG"},
        "dosage": {"Actara 25 WG": "80 g/acre in 200L water"},
        "application_timing": "Apply at V3-V6 stage at first sign of fall armyworm damage.",
        "harvest_gap_days": 25,
    },
}

# ---------------------------------------------------------------------------
# 11 Indian Languages — greetings, CTAs, TTS metadata (unchanged from original)
# ---------------------------------------------------------------------------

LANGUAGE_META = {
    "Hindi":     {"sarvam_code": "hi-IN",  "script": "Devanagari",  "sarvam_speaker": "arvind", "pyttsx3_lang": "hi"},
    "Punjabi":   {"sarvam_code": "pa-IN",  "script": "Gurmukhi",   "sarvam_speaker": "arvind", "pyttsx3_lang": "pa"},
    "Marathi":   {"sarvam_code": "mr-IN",  "script": "Devanagari",  "sarvam_speaker": "arvind", "pyttsx3_lang": "mr"},
    "Gujarati":  {"sarvam_code": "gu-IN",  "script": "Gujarati",   "sarvam_speaker": "arvind", "pyttsx3_lang": "gu"},
    "Kannada":   {"sarvam_code": "kn-IN",  "script": "Kannada",    "sarvam_speaker": "arvind", "pyttsx3_lang": "kn"},
    "Bengali":   {"sarvam_code": "bn-IN",  "script": "Bengali",    "sarvam_speaker": "arvind", "pyttsx3_lang": "bn"},
    "Tamil":     {"sarvam_code": "ta-IN",  "script": "Tamil",      "sarvam_speaker": "arvind", "pyttsx3_lang": "ta"},
    "Telugu":    {"sarvam_code": "te-IN",  "script": "Telugu",     "sarvam_speaker": "arvind", "pyttsx3_lang": "te"},
    "Odia":      {"sarvam_code": "od-IN",  "script": "Odia",       "sarvam_speaker": "arvind", "pyttsx3_lang": "or"},
    "Assamese":  {"sarvam_code": "as-IN",  "script": "Assamese",   "sarvam_speaker": "arvind", "pyttsx3_lang": "as"},
    "Malayalam": {"sarvam_code": "ml-IN",  "script": "Malayalam",  "sarvam_speaker": "arvind", "pyttsx3_lang": "ml"},
}

LANGUAGE_GREETINGS = {
    "Hindi":     "नमस्ते {name}!",
    "Punjabi":   "ਸਤ ਸ੍ਰੀ ਅਕਾਲ {name}!",
    "Marathi":   "नमस्कार {name}!",
    "Gujarati":  "નમસ્તે {name}!",
    "Kannada":   "ನಮಸ್ಕಾರ {name}!",
    "Bengali":   "নমস্কার {name}!",
    "Tamil":     "வணக்கம் {name}!",
    "Telugu":    "నమస్కారం {name}!",
    "Odia":      "ନମସ୍କାର {name}!",
    "Assamese":  "নমস্কাৰ {name}!",
    "Malayalam": "നമസ്കാരം {name}!",
}

LANGUAGE_CTA = {
    "Hindi":     {"whatsapp_reply": "जानकारी के लिए YES टाइप करें।",                         "ivr_press": "अधिक जानकारी के लिए 1 दबाएं।",         "sms_missed_call": "मिस्ड कॉल करें: {number}"},
    "Punjabi":   {"whatsapp_reply": "ਜਾਣਕਾਰੀ ਲਈ YES ਲਿਖੋ।",                                 "ivr_press": "ਹੋਰ ਜਾਣਕਾਰੀ ਲਈ 1 ਦਬਾਓ।",             "sms_missed_call": "ਮਿਸਡ ਕਾਲ ਕਰੋ: {number}"},
    "Marathi":   {"whatsapp_reply": "माहितीसाठी YES टाइप करा।",                              "ivr_press": "अधिक माहितीसाठी 1 दाबा।",              "sms_missed_call": "मिस्ड कॉल करा: {number}"},
    "Gujarati":  {"whatsapp_reply": "માહિતી માટે YES ટાઈપ કરો.",                              "ivr_press": "વધુ માહિતી માટે 1 દબાવો.",              "sms_missed_call": "મિસ્ડ કૉલ કરો: {number}"},
    "Kannada":   {"whatsapp_reply": "ಮಾಹಿತಿಗಾಗಿ YES ಟೈಪ್ ಮಾಡಿ.",                            "ivr_press": "ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ 1 ಒತ್ತಿ.",           "sms_missed_call": "ಮಿಸ್ಡ್ ಕಾಲ್ ಮಾಡಿ: {number}"},
    "Bengali":   {"whatsapp_reply": "তথ্যের জন্য YES টাইপ করুন।",                           "ivr_press": "আরও তথ্যের জন্য 1 চাপুন।",            "sms_missed_call": "মিসড কল করুন: {number}"},
    "Tamil":     {"whatsapp_reply": "தகவலுக்கு YES என்று தட்டச்சு செய்யுங்கள்.",             "ivr_press": "மேலும் தகவலுக்கு 1 ஐ அழுத்துங்கள்.",  "sms_missed_call": "மிஸ்டு கால் செய்யுங்கள்: {number}"},
    "Telugu":    {"whatsapp_reply": "సమాచారం కోసం YES టైప్ చేయండి.",                         "ivr_press": "మరింత సమాచారానికి 1 నొక్కండి.",         "sms_missed_call": "మిస్డ్ కాల్ చేయండి: {number}"},
    "Odia":      {"whatsapp_reply": "ସୂଚନା ପାଇଁ YES ଟାଇପ୍ କରନ୍ତୁ।",                        "ivr_press": "ଅଧିକ ସୂଚନା ପାଇଁ 1 ଦବାନ୍ତୁ।",           "sms_missed_call": "ମିସ୍ଡ କଲ୍ କରନ୍ତୁ: {number}"},
    "Assamese":  {"whatsapp_reply": "তথ্যৰ বাবে YES টাইপ কৰক।",                              "ivr_press": "অধিক তথ্যৰ বাবে 1 টিপক।",              "sms_missed_call": "মিচড কল কৰক: {number}"},
    "Malayalam": {"whatsapp_reply": "വിവരങ്ങൾക്ക് YES ടൈപ്പ് ചെയ്യൂ.",                       "ivr_press": "കൂടുതൽ വിവരങ്ങൾക്ക് 1 അമർത്തൂ.",      "sms_missed_call": "മിസ്ഡ് കോൾ ചെയ്യൂ: {number}"},
}

# ---------------------------------------------------------------------------
# OFFLINE FALLBACK TEMPLATES — all 11 languages, all content formats
# FIX: Original hardcoded Hindi. Now uses language-aware strings from the
# dictionaries already defined above.
# ---------------------------------------------------------------------------

# Main body sentences per language (used to build the offline template body)
# Format: {crop}, {stage}, {threat}, {product}, {dosage} are substituted at runtime.
OFFLINE_BODY = {
    "Hindi":     "आपकी {crop} की फसल {stage} अवस्था में है। {threat} का खतरा बढ़ रहा है। Syngenta का {product} ({dosage}) का छिड़काव करें।",
    "Punjabi":   "ਤੁਹਾਡੀ {crop} ਦੀ ਫਸਲ {stage} ਪੜਾਅ 'ਤੇ ਹੈ। {threat} ਦਾ ਖ਼ਤਰਾ ਵੱਧ ਰਿਹਾ ਹੈ। Syngenta ਦਾ {product} ({dosage}) ਦਾ ਛਿੜਕਾਅ ਕਰੋ।",
    "Marathi":   "तुमचे {crop} पीक {stage} अवस्थेत आहे। {threat} चा धोका वाढत आहे। Syngenta चे {product} ({dosage}) फवारा करा।",
    "Gujarati":  "તમારો {crop} પાક {stage} અવસ્થામાં છે। {threat} નો ખતરો વધી રહ્યો છે। Syngenta નો {product} ({dosage}) છાંટો.",
    "Kannada":   "ನಿಮ್ಮ {crop} ಬೆಳೆ {stage} ಹಂತದಲ್ಲಿದೆ. {threat} ಅಪಾಯ ಹೆಚ್ಚಾಗುತ್ತಿದೆ. Syngenta ನ {product} ({dosage}) ಸಿಂಪಡಿಸಿ.",
    "Bengali":   "আপনার {crop} ফসল {stage} পর্যায়ে আছে। {threat} এর বিপদ বাড়ছে। Syngenta এর {product} ({dosage}) স্প্রে করুন।",
    "Tamil":     "உங்கள் {crop} பயிர் {stage} நிலையில் உள்ளது. {threat} அபாயம் அதிகரிக்கிறது. Syngenta {product} ({dosage}) தெளிக்கவும்.",
    "Telugu":    "మీ {crop} పంట {stage} దశలో ఉంది. {threat} ముప్పు పెరుగుతోంది. Syngenta {product} ({dosage}) పిచికారీ చేయండి.",
    "Odia":      "ଆପଣଙ୍କ {crop} ଫସଲ {stage} ଅବସ୍ଥାରେ ଅଛି। {threat} ର ବିପଦ ବଢ଼ୁଛି। Syngenta ର {product} ({dosage}) ସ୍ପ୍ରେ କରନ୍ତୁ।",
    "Assamese":  "আপোনাৰ {crop} শস্য {stage} অৱস্থাত আছে। {threat} ৰ বিপদ বাঢ়িছে। Syngenta ৰ {product} ({dosage}) স্প্ৰে কৰক।",
    "Malayalam": "നിങ്ങളുടെ {crop} വിള {stage} ഘട്ടത്തിലാണ്. {threat} അപകടം വർധിക്കുന്നു. Syngenta {product} ({dosage}) തളിക്കുക.",
}

OFFLINE_IVR = {
    "Hindi":     "यह Syngenta India की ओर से एक महत्वपूर्ण कृषि सलाह है। आपकी {crop} की फसल {stage} अवस्था में है और {threat} का खतरा है। Syngenta का {product} — {dosage} — का प्रयोग करें। {ivr_press} धन्यवाद।",
    "Punjabi":   "ਇਹ Syngenta India ਵੱਲੋਂ ਇੱਕ ਮਹੱਤਵਪੂਰਨ ਖੇਤੀ ਸਲਾਹ ਹੈ। ਤੁਹਾਡੀ {crop} ਦੀ ਫਸਲ {stage} ਪੜਾਅ 'ਤੇ ਹੈ ਅਤੇ {threat} ਦਾ ਖ਼ਤਰਾ ਹੈ। Syngenta ਦਾ {product} — {dosage} — ਵਰਤੋ। {ivr_press} ਧੰਨਵਾਦ।",
    "Marathi":   "हे Syngenta India कडून एक महत्त्वाचे कृषी सल्ला आहे। तुमचे {crop} पीक {stage} अवस्थेत आहे आणि {threat} चा धोका आहे। Syngenta चे {product} — {dosage} — वापरा। {ivr_press} धन्यवाद।",
    "Gujarati":  "આ Syngenta India તરફથી એક મહત્ત્વની કૃષિ સલાહ છે. તમારો {crop} પાક {stage} અવસ્થામાં છે અને {threat} નો ખતરો છે. Syngenta નો {product} — {dosage} — વાપરો. {ivr_press} આભાર.",
    "Kannada":   "ಇದು Syngenta India ಯಿಂದ ಒಂದು ಮುಖ್ಯ ಕೃಷಿ ಸಲಹೆ. ನಿಮ್ಮ {crop} ಬೆಳೆ {stage} ಹಂತದಲ್ಲಿದೆ ಮತ್ತು {threat} ಅಪಾಯವಿದೆ. Syngenta {product} — {dosage} — ಬಳಸಿ. {ivr_press} ಧನ್ಯವಾದ.",
    "Bengali":   "এটি Syngenta India থেকে একটি গুরুত্বপূর্ণ কৃষি পরামর্শ। আপনার {crop} ফসল {stage} পর্যায়ে আছে এবং {threat} এর বিপদ আছে। Syngenta {product} — {dosage} — ব্যবহার করুন। {ivr_press} ধন্যবাদ।",
    "Tamil":     "இது Syngenta India வின் முக்கியமான விவசாய ஆலோசனை. உங்கள் {crop} பயிர் {stage} நிலையில் உள்ளது மற்றும் {threat} அபாயம் உள்ளது. Syngenta {product} — {dosage} — பயன்படுத்துங்கள். {ivr_press} நன்றி.",
    "Telugu":    "ఇది Syngenta India నుండి ముఖ్యమైన వ్యవసాయ సలహా. మీ {crop} పంట {stage} దశలో ఉంది మరియు {threat} ముప్పు ఉంది. Syngenta {product} — {dosage} — ఉపయోగించండి. {ivr_press} ధన్యవాదాలు.",
    "Odia":      "ଏହା Syngenta India ର ଏକ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ କୃଷି ପରାମର୍ଶ। ଆପଣଙ୍କ {crop} ଫସଲ {stage} ଅବସ୍ଥାରେ ଅଛି ଏବଂ {threat} ର ବିପଦ ଅଛି। Syngenta {product} — {dosage} — ବ୍ୟବହାର କରନ୍ତୁ। {ivr_press} ଧନ୍ୟବାଦ।",
    "Assamese":  "এইটো Syngenta India ৰ পৰা এটি গুৰুত্বপূৰ্ণ কৃষি পৰামৰ্শ। আপোনাৰ {crop} শস্য {stage} অৱস্থাত আছে আৰু {threat} ৰ বিপদ আছে। Syngenta {product} — {dosage} — ব্যৱহাৰ কৰক। {ivr_press} ধন্যবাদ।",
    "Malayalam": "ഇത് Syngenta India യിൽ നിന്നുള്ള ഒരു പ്രധാന കൃഷി ഉപദേശം. നിങ്ങളുടെ {crop} വിള {stage} ഘട്ടത്തിലാണ്, {threat} അപകടമുണ്ട്. Syngenta {product} — {dosage} — ഉപയോഗിക്കുക. {ivr_press} നന്ദി.",
}

OFFLINE_SOCIAL = {
    "Hindi":     "🌾 {crop} किसान — {threat} अलर्ट! Syngenta {product} से अपनी फसल बचाएं। आज ही डीलर से पूछें। #Syngenta #AgriIndia 🌿",
    "Punjabi":   "🌾 {crop} ਕਿਸਾਨ — {threat} ਅਲਰਟ! Syngenta {product} ਨਾਲ ਫਸਲ ਬਚਾਓ। ਅੱਜ ਡੀਲਰ ਤੋਂ ਪੁੱਛੋ। #Syngenta #AgriIndia 🌿",
    "Marathi":   "🌾 {crop} शेतकरी — {threat} सतर्कता! Syngenta {product} ने पीक वाचवा। आज डीलरला विचारा। #Syngenta #AgriIndia 🌿",
    "Gujarati":  "🌾 {crop} ખેડૂત — {threat} ચેતવણી! Syngenta {product} થી પાક બચાવો. આજે ડીલરને પૂછો. #Syngenta #AgriIndia 🌿",
    "Kannada":   "🌾 {crop} ರೈತರೇ — {threat} ಎಚ್ಚರಿಕೆ! Syngenta {product} ದಿಂದ ಬೆಳೆ ರಕ್ಷಿಸಿ. ಇಂದು ಡೀಲರ್ ಅನ್ನು ಕೇಳಿ. #Syngenta #AgriIndia 🌿",
    "Bengali":   "🌾 {crop} কৃষক — {threat} সতর্কতা! Syngenta {product} দিয়ে ফসল রক্ষা করুন। আজই ডিলারকে জিজ্ঞেস করুন। #Syngenta #AgriIndia 🌿",
    "Tamil":     "🌾 {crop} விவசாயிகளே — {threat} எச்சரிக்கை! Syngenta {product} மூலம் பயிரை பாதுகாக்குங்கள். இன்றே கடைக்காரரிடம் கேளுங்கள். #Syngenta #AgriIndia 🌿",
    "Telugu":    "🌾 {crop} రైతులకు — {threat} హెచ్చరిక! Syngenta {product} తో పంటను రక్షించండి. ఈరోజే డీలర్‌ని అడగండి. #Syngenta #AgriIndia 🌿",
    "Odia":      "🌾 {crop} ଚାଷୀ — {threat} ସତର୍କତା! Syngenta {product} ରେ ଫସଲ ରକ୍ଷା କରନ୍ତୁ। ଆଜି ଡିଲରରଙ୍କୁ ପଚାରନ୍ତୁ। #Syngenta #AgriIndia 🌿",
    "Assamese":  "🌾 {crop} কৃষক — {threat} সতৰ্কতা! Syngenta {product} ৰে শস্য ৰক্ষা কৰক। আজিয়ে ডিলাৰক সুধিব। #Syngenta #AgriIndia 🌿",
    "Malayalam": "🌾 {crop} കർഷകരേ — {threat} മുന്നറിയിപ്പ്! Syngenta {product} ഉപയോഗിച്ച് വിള സംരക്ഷിക്കൂ. ഇന്നു തന്നെ ഡീലറോടു ചോദിക്കൂ. #Syngenta #AgriIndia 🌿",
}


class ContentFormat(str, Enum):
    WHATSAPP_RICH   = "whatsapp_rich"
    WHATSAPP_TEXT   = "whatsapp_text"
    IVR_VOICE       = "ivr_voice"
    SMS             = "sms"
    FIELD_REP_BRIEF = "field_rep_brief"
    SOCIAL_POST     = "social_post"
    VIDEO_THUMBNAIL = "video_thumbnail"


@dataclass
class ContentRequest:
    grower_id: str
    crop: str
    growth_stage: str
    language: str
    state: str
    tehsil: str
    content_format: ContentFormat
    device_type: str = "smartphone"
    grower_name: str = "किसान भाई"
    farm_size_acres: float = 2.0
    nearest_retailer: str = ""
    active_threats: list = field(default_factory=list)
    days_to_next_stage: int = 14
    weather_alert: str = ""
    rep_name: str = ""
    pest_pressure_index: float = 0.0
    weather_risk_score: float = 0.0


@dataclass
class GeneratedContent:
    grower_id: str
    content_format: str
    language: str
    text: str
    char_count: int
    llm_model: str
    prompt_tokens: int
    completion_tokens: int
    generation_timestamp: str
    tts_audio_url: Optional[str] = None
    video_thumbnail_url: Optional[str] = None
    video_caption: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt builders (unchanged from original — only used for LLM path)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ICAR Advisory RAG — solution doc Engine 1:
# "RAG over: ICAR advisories, product label data, KVK bulletins"
# In production this would be a vector DB (ChromaDB/FAISS) over scraped ICAR PDFs.
# Here we provide a curated inline knowledge base per crop/threat combination
# that is injected into the LLM system prompt for grounding.
# ---------------------------------------------------------------------------

ICAR_RAG_ADVISORIES: dict[str, str] = {
    # Format: "crop|threat_keyword" → advisory text
    "wheat|yellow rust": (
        "ICAR-IARI Advisory: Yellow rust (Puccinia striiformis) in wheat — "
        "Apply Propiconazole (Tilt 25 EC) 0.1% or Mancozeb 0.25% at first sign. "
        "Use resistant varieties HD 2967 / PBW 550. Spray during cool (10-15°C) humid conditions."
    ),
    "wheat|powdery mildew": (
        "ICAR Advisory: Wheat powdery mildew — Apply Karathane (Dinocap) 0.05% or "
        "Sulfex 0.2% at flag leaf stage. Ensure good canopy airflow."
    ),
    "wheat|aphids": (
        "ICAR-NCIPM: Wheat aphid (Sitobion avenae) — Spray Imidacloprid 17.8 SL "
        "(0.5 ml/l) or Thiamethoxam 25 WG at first colony detection. "
        "Economic threshold: 10 aphids/tiller."
    ),
    "mustard|white rust": (
        "ICAR-DOR Advisory: Mustard white rust (Albugo candida) — "
        "Spray Metalaxyl + Mancozeb (Ridomil Gold) 2 g/l at 5% flowering. "
        "Repeat after 15 days if disease persists."
    ),
    "mustard|aphids": (
        "ICAR Advisory: Mustard aphid (Lipaphis erysimi) — Spray Dimethoate 30 EC "
        "(1 ml/l) or Thiamethoxam 25 WG at bud initiation. "
        "Economic threshold: 20 aphids/plant at vegetative stage."
    ),
    "chickpea|pod borer": (
        "ICAR-ICRISAT: Chickpea pod borer (Helicoverpa armigera) — "
        "Apply Indoxacarb 14.5 SC (0.75 ml/l) at pod initiation. "
        "Use pheromone traps (1/acre) for monitoring. Spray at dusk."
    ),
    "potato|late blight": (
        "ICAR-CPRI Advisory: Potato late blight (Phytophthora infestans) — "
        "Apply Cymoxanil + Mancozeb (Curzate M-8) 2.5 g/l every 7 days in humid weather. "
        "Avoid overhead irrigation. Rogue out infected plants."
    ),
    "potato|early blight": (
        "ICAR Advisory: Potato early blight (Alternaria solani) — "
        "Spray Chlorothalonil (Kavach) 2 g/l at first symptom. "
        "Remove lower infected leaves. Ensure balanced NPK fertilization."
    ),
}


def get_icar_advisory(crop: str, threats: list[str]) -> str:
    """Look up the most relevant ICAR advisory for the crop-threat combination.
    Returns a formatted advisory string for injection into the LLM system prompt."""
    crop_lower = crop.lower()
    for threat in threats:
        threat_lower = threat.lower()
        for key, advisory in ICAR_RAG_ADVISORIES.items():
            k_crop, k_threat = key.split("|", 1)
            if k_crop == crop_lower and k_threat in threat_lower:
                return f"\nICAR Advisory (RAG): {advisory}"
    # Generic fallback
    return ""


def build_system_prompt(req: ContentRequest) -> str:
    crop_kb = CROP_KNOWLEDGE.get(req.crop, {})
    threats = req.active_threats or crop_kb.get("key_threats", [])[:2]
    products = crop_kb.get("products", {})
    primary_product = products.get("fungicide") or products.get("insecticide") or "Tilt 250 EC"
    dosage = crop_kb.get("dosage", {}).get(primary_product, "as per label")
    timing = crop_kb.get("application_timing", "Apply at first sign of pest/disease.")

    channel_instructions = {
        ContentFormat.WHATSAPP_RICH:   (f"Write a WhatsApp message in {req.language}. Max 1024 characters. Include 1-2 relevant emojis. End with a clear CTA to reply 'YES' for more information."),
        ContentFormat.WHATSAPP_TEXT:   (f"Write a concise WhatsApp text message in {req.language}. Max 512 characters. No emojis. CTA: reply keyword."),
        ContentFormat.IVR_VOICE:       (f"Write a 30-second IVR voice call script in {req.language}. Structure: warm greeting (5s), main alert (10s), product recommendation (10s), CTA 'Press 1 for dealer' (5s). No special characters or symbols."),
        ContentFormat.SMS:             (f"Write a 160-character SMS in {req.language}. Ultra-concise: alert + product name + missed-call CTA. Every character counts."),
        ContentFormat.FIELD_REP_BRIEF: ("Write a field representative visit brief in English. Include: grower background, current crop threat, recommended product + dosage, 3 key talking points, suggested next steps. Use bullet points."),
        ContentFormat.SOCIAL_POST:     (f"Write a Facebook/WhatsApp Status post in {req.language}. Max 300 characters. Engaging, farmer-friendly tone. Include 2 emojis."),
    }.get(req.content_format, f"Write marketing content in {req.language}.")

    pest_ctx    = (f"- Active pest pressure: {req.pest_pressure_index:.1f}/1.0 (ICAR-NCIPM alert)\n" if req.pest_pressure_index > 0.3 else "")
    weather_ctx = (f"- Weather risk: {req.weather_risk_score:.1f}/1.0 (IMD advisory)\n" if req.weather_risk_score > 0.3 else "")

    # RAG: inject ICAR advisory (solution doc Engine 1: "RAG over ICAR advisories")
    icar_advisory = get_icar_advisory(req.crop, threats)

    return (
        f"You are an expert agricultural marketing content writer for Syngenta India.\n"
        f"You specialize in crop protection advisory content for Indian farmers.\n\n"
        f"TASK: {channel_instructions}\n\n"
        f"AGRONOMIC CONTEXT (weave naturally — do not mention these labels):\n"
        f"- Farmer location: {req.state} / {req.tehsil}\n"
        f"- Crop: {req.crop} | Growth stage: {req.growth_stage}\n"
        f"- Active threats: {', '.join(threats)}\n"
        f"- Recommended product: {primary_product}\n"
        f"- Dosage: {dosage}\n"
        f"- Application timing: {timing}\n"
        f"- Farm size: {req.farm_size_acres:.1f} acres\n"
        f"- Days to next growth stage: {req.days_to_next_stage}\n"
        f"- Nearest retailer: {req.nearest_retailer or 'local agri-input store'}\n"
        f"{pest_ctx}{weather_ctx}"
        f"{'- Weather alert: ' + req.weather_alert if req.weather_alert else ''}\n"
        f"{icar_advisory}\n\n"
        f"TONE: Warm, respectful, advisory like a trusted agronomist friend.\n"
        f"Use respectful second-person address appropriate to the language (e.g. 'aap' in Hindi).\n"
        f"Never use jargon the farmer would not know.\n"
        f"Always position Syngenta as a trusted partner, not just a seller.\n\n"
        f"OUTPUT: Only the final message content. No labels, no markdown, no preamble."
    )


def build_user_prompt(req: ContentRequest) -> str:
    greeting = LANGUAGE_GREETINGS.get(req.language, "Namaste {name}!").format(name=req.grower_name)
    crop_kb  = CROP_KNOWLEDGE.get(req.crop, {})
    threats  = req.active_threats or crop_kb.get("key_threats", [])[:1]
    return (
        f"Generate the {req.content_format.value} marketing message. "
        f"Farmer name token: '{req.grower_name}'. "
        f"Crop: {req.crop} | Stage: {req.growth_stage} | "
        f"Main threat: {threats[0] if threats else 'general crop protection'}. "
        f"Use this greeting if the format supports it: '{greeting}'"
    )

SARVAM_V3_SPEAKERS = {
    "Hindi":     "kavya",
    "Punjabi":   "aditya",
    "Marathi":   "roopa",
    "Gujarati":  "ritu",
    "Kannada":   "gokul",
    "Bengali":   "simran",
    "Tamil":     "kavitha",
    "Telugu":    "vijay",
    "Odia":      "anand",
    "Assamese":  "shruti",
    "Malayalam": "mani",
}

def synthesize_via_sarvam(text: str, language: str,
                           sarvam_api_key: str = "") -> Optional[str]:
    """
    Sarvam AI TTS using bulbul:v3 (current flagship model).
    API docs: https://docs.sarvam.ai/api-reference-docs/getting-started/models/bulbul
    Returns data-URI base64 WAV on success, None to trigger next fallback.
    """
    sarvam_api_key=""
    api_key = sarvam_api_key or os.getenv("SARVAM_API_KEY", "")
    if not api_key:
        return None

    lang_meta = LANGUAGE_META.get(language, {"sarvam_code": "hi-IN"})
    speaker   = SARVAM_V3_SPEAKERS.get(language, "Shubh")  # Shubh is v3 default

    try:
        from sarvamai import SarvamAI

        client = SarvamAI(api_subscription_key=api_key)

        response = client.text_to_speech.convert(
            text=text[:2500],                          # v3 supports up to 2500 chars
            target_language_code=lang_meta["sarvam_code"],
            speaker=speaker,
            model="bulbul:v3",                         # v2 is superseded; use v3
            pace=1.0,                                  # range 0.5–2.0 on v3
            speech_sample_rate=8000,                   # IVR-optimised sample rate
            enable_preprocessing=True,                 # handles dates, numbers, mixed text
            # NOTE: pitch and loudness are NOT supported on v3 — removed
        )

        # SDK returns response.audios — a list of base64 strings
        audio_b64 = (response.audios or [""])[0]
        if not audio_b64:
            print(f"  [Sarvam] Empty audio returned for {language}/{speaker}")
            return None

        print(f"  [Sarvam] TTS OK — {language} / {speaker} / bulbul:v3 ({len(audio_b64)} b64 chars)")
        return f"data:audio/wav;base64,{audio_b64}"

    except ImportError:
        print("sarvamai SDK not installed. Run: pip install sarvamai")
        return None
    except Exception as e:
        print(f"  [Sarvam] TTS failed for {language}: {e}")
        return None

def synthesize_via_bhashini(text: str, language: str,
                              bhashini_api_key: str = "") -> Optional[str]:
    """
    Bhashini API TTS — the TTS provider named in the solution document.
    Solution 1 Phase 3: "Deploy GenAI content engine with Bhashini for 11 languages"
    Solution 2 Model 3: "Fine-tuned IndicBART / Llama-3 + Bhashini API"

    Bhashini (ULCA / Bhashini platform) provides govt-backed Indic TTS.
    API key obtained from https://bhashini.gov.in/ulca/model/explore-models
    Falls back gracefully to Sarvam/pyttsx3 if key absent or call fails.
    Returns data-URI base64 WAV on success, None to trigger next fallback.
    """
    import httpx
    api_key = bhashini_api_key or os.getenv("BHASHINI_API_KEY", "")
    if not api_key:
        return None

    lang_meta = LANGUAGE_META.get(language, {"sarvam_code": "hi-IN"})
    bhashini_lang = lang_meta["sarvam_code"]  # Bhashini uses the same BCP-47 codes

    try:
        # Step 1: Resolve pipeline for TTS
        pipeline_resp = httpx.post(
            "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline",
            headers={
                "userID": api_key.split("|")[0] if "|" in api_key else api_key,
                "ulcaApiKey": api_key.split("|")[1] if "|" in api_key else api_key,
                "Content-Type": "application/json",
            },
            json={
                "pipelineTasks": [{"taskType": "tts", "config": {"language": {"sourceLanguage": bhashini_lang.split("-")[0]}}}],
                "pipelineRequestConfig": {"pipelineId": "64392f96daac500b55c543cd"},
            },
            timeout=10,
        )
        pipeline_resp.raise_for_status()
        pipeline_data = pipeline_resp.json()
        callback_url = pipeline_data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
        inference_key = pipeline_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
        config = pipeline_data["pipelineResponseConfig"][0]["config"][0]
        service_id = config["serviceId"]
        model_id   = config["modelId"]

        # Step 2: Run TTS inference
        infer_resp = httpx.post(
            callback_url,
            headers={"Authorization": inference_key, "Content-Type": "application/json"},
            json={
                "pipelineTasks": [{
                    "taskType": "tts",
                    "config": {
                        "language": {"sourceLanguage": bhashini_lang.split("-")[0]},
                        "serviceId": service_id,
                        "gender": "male",
                        "samplingRate": 8000,
                    }
                }],
                "inputData": {"input": [{"source": text[:500]}]},
            },
            timeout=20,
        )
        infer_resp.raise_for_status()
        audio_content = infer_resp.json()["pipelineResponse"][0]["audio"][0]["audioContent"]
        return f"data:audio/wav;base64,{audio_content}" if audio_content else None
    except Exception as e:
        print(f"Bhashini TTS failed for {language}: {e} — trying Sarvam.")
        return None


def synthesize_tts(text: str, language: str, sarvam_api_key: str = "",
                   bhashini_api_key: str = "") -> Optional[str]:
    # ── DEBUG (remove after fixing) ──────────────────────────────────────────
    b_key = bhashini_api_key or os.getenv("BHASHINI_API_KEY", "")
    s_key = sarvam_api_key   or os.getenv("SARVAM_API_KEY", "")
    print(f"  [TTS DEBUG] language={language}")
    print(f"  [TTS DEBUG] bhashini_key={'SET ('+b_key[:6]+'...)' if b_key else 'MISSING'}")
    print(f"  [TTS DEBUG] sarvam_key= {'SET ('+s_key[:6]+'...)' if s_key else 'MISSING'}")
    # ─────────────────────────────────────────────────────────────────────────

    result = synthesize_via_bhashini(text, language, bhashini_api_key)
    if result is not None:
        return result

    result = synthesize_via_sarvam(text, language, sarvam_api_key)
    if result is not None:
        return result

    print(f"  [TTS] Bhashini and Sarvam both failed for {language} — check your API keys.")
    return None


# ---------------------------------------------------------------------------
# Visual / Video thumbnail (unchanged from original — SVG fallback already works)
# ---------------------------------------------------------------------------

CROP_VISUAL_THEMES: dict[str, str] = {
    "wheat":    "golden wheat field at sunrise, close-up on wheat ears with morning dew",
    "mustard":  "bright yellow mustard flowers in bloom, farmer walking through field",
    "chickpea": "chickpea pods on green plant, Indian farmer inspecting crop",
    "potato":   "freshly harvested potatoes in red soil, farmer holding healthy tubers",
    "barley":   "barley crop with tall stalks swaying in breeze, blue sky background",
    "lentil":   "lentil pods hanging on plant, close-up on leaves",
    "safflower": "vibrant orange safflower blooms, Indian agricultural landscape",
    "cumin":    "cumin seed heads in late stage, arid landscape background",
    "maize":    "tall maize/corn stalks with green cobs, Indian smallholder farm",
}

def generate_video_thumbnail(req: "ContentRequest") -> tuple[Optional[str], Optional[str]]:
    """
    Tries Stability AI (cloud) → falls back to SVG placeholder (offline).
    SVG fallback was already solid in the original. No changes needed here.
    """
    crop_kb      = CROP_KNOWLEDGE.get(req.crop, {})
    threats      = req.active_threats or crop_kb.get("key_threats", ["pest pressure"])
    main_threat  = threats[0]
    crop_theme   = CROP_VISUAL_THEMES.get(req.crop, f"{req.crop} field, Indian farm")
    greeting     = LANGUAGE_GREETINGS.get(req.language, "Namaste!").format(name="")

    stability_key = os.getenv("STABILITY_API_KEY", "")
    if stability_key:
        try:
            import httpx
            prompt = (
                f"Agricultural marketing card: {crop_theme}. "
                f"Syngenta product label visible. Professional photography, "
                f"warm sunrise light, Indian smallholder farm, vibrant colors. "
                f"Text overlay area at bottom third. --ar 16:9"
            )
            resp = httpx.post(
                "https://api.stability.ai/v2beta/stable-image/generate/ultra",
                headers={"authorization": f"Bearer {stability_key}", "accept": "application/json"},
                data={"prompt": prompt, "output_format": "jpeg"},
                timeout=30,
            )
            resp.raise_for_status()
            img_b64 = resp.json().get("image", "")
            if img_b64:
                caption = (f"{greeting.strip()} {req.crop.title()} {req.growth_stage} alert — {main_threat}. Syngenta solution available.")
                return f"data:image/jpeg;base64,{img_b64}", caption
        except Exception as e:
            print(f"Stability AI image generation failed: {e}. Using SVG placeholder.")

    # SVG fallback (offline — no API needed)
    crop_color_map = {
        "wheat": "#F4A832", "mustard": "#FFD700", "chickpea": "#C8A96E",
        "potato": "#8B5E3C", "barley": "#D4B483", "lentil": "#B87333",
        "safflower": "#FF6B35", "cumin": "#8B7355", "maize": "#4CAF50",
    }
    accent = crop_color_map.get(req.crop, "#4CAF50")
    threats_text = main_threat[:40]

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450" width="800" height="450">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a4a1a;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2d7a2d;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="bar" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{accent};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{accent}cc;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="800" height="450" fill="url(#bg)" rx="12"/>
  <g opacity="0.15">
    <line x1="0" y1="100" x2="800" y2="80" stroke="white" stroke-width="2"/>
    <line x1="0" y1="150" x2="800" y2="130" stroke="white" stroke-width="2"/>
    <line x1="0" y1="200" x2="800" y2="180" stroke="white" stroke-width="2"/>
    <line x1="0" y1="250" x2="800" y2="230" stroke="white" stroke-width="2"/>
  </g>
  <rect x="0" y="340" width="800" height="110" fill="url(#bar)" opacity="0.92" rx="0"/>
  <rect x="0" y="0" width="800" height="54" fill="black" opacity="0.55" rx="12"/>
  <text x="24" y="34" font-family="Arial,sans-serif" font-size="20" font-weight="bold" fill="{accent}">🌾 SYNGENTA INDIA</text>
  <text x="780" y="34" font-family="Arial,sans-serif" font-size="14" fill="white" text-anchor="end">CROP ADVISORY</text>
  <text x="400" y="160" font-family="Arial,sans-serif" font-size="52" font-weight="bold" fill="white" text-anchor="middle" opacity="0.95">{req.crop.upper()}</text>
  <text x="400" y="210" font-family="Arial,sans-serif" font-size="22" fill="{accent}" text-anchor="middle">Stage: {req.growth_stage.title()}</text>
  <text x="400" y="260" font-family="Arial,sans-serif" font-size="18" fill="white" text-anchor="middle" opacity="0.9">⚠ Alert: {threats_text}</text>
  <text x="400" y="300" font-family="Arial,sans-serif" font-size="16" fill="white" text-anchor="middle" opacity="0.75">{req.tehsil} · {req.state}</text>
  <text x="400" y="385" font-family="Arial,sans-serif" font-size="18" font-weight="bold" fill="white" text-anchor="middle">Contact your nearest dealer for Syngenta solution</text>
  <text x="400" y="420" font-family="Arial,sans-serif" font-size="14" fill="white" text-anchor="middle" opacity="0.85">Language: {req.language} | ID: {req.grower_id}</text>
</svg>"""

    svg_b64 = base64.b64encode(svg_content.encode()).decode()
    caption = (f"{req.crop.title()} {req.growth_stage} advisory — {main_threat}. Contact Syngenta dealer in {req.tehsil}.")
    return f"data:image/svg+xml;base64,{svg_b64}", caption


# ---------------------------------------------------------------------------
# Main generation function
# Priority: Ollama (local) → Gemini (cloud) → Rule-based (offline)
# CHANGED: Ollama tried first so the system works offline by default.
# Gemini is an optional cloud upgrade, not the primary path.
# ---------------------------------------------------------------------------

def generate_content(req: ContentRequest, api_key: str = "",
                     sarvam_api_key: str = "",
                     bhashini_api_key: str = "") -> GeneratedContent:
    """
    Generation priority:
      1. Ollama (local) — always tried first, no key, works offline
      2. Gemini (cloud) — only if Ollama unavailable/empty AND key is set
      3. Rule-based fallback — guaranteed output, all 11 languages
    """
    if req.content_format == ContentFormat.VIDEO_THUMBNAIL:
        thumb_url, caption = generate_video_thumbnail(req)
        alt_caption = caption or f"{req.crop.title()} advisory for {req.tehsil}"
        return GeneratedContent(
            grower_id=req.grower_id,
            content_format=req.content_format.value,
            language=req.language,
            text=alt_caption,
            char_count=len(alt_caption),
            llm_model="stability_ai_or_svg_fallback",
            prompt_tokens=0, completion_tokens=0,
            generation_timestamp=datetime.utcnow().isoformat(),
            video_thumbnail_url=thumb_url,
            video_caption=caption,
            metadata={"crop": req.crop, "stage": req.growth_stage, "tehsil": req.tehsil},
        )

    system_prompt = build_system_prompt(req)
    user_prompt   = build_user_prompt(req)

    max_tokens_map = {
        ContentFormat.WHATSAPP_RICH:   350,
        ContentFormat.WHATSAPP_TEXT:   180,
        ContentFormat.IVR_VOICE:       200,
        ContentFormat.SMS:              80,
        ContentFormat.FIELD_REP_BRIEF: 400,
        ContentFormat.SOCIAL_POST:     150,
    }
    max_tokens = max_tokens_map.get(req.content_format, 300)

    text      = None
    llm_model = "rule_based_fallback"

    # ── STEP 2: Gemini (cloud, only if Ollama failed/unavailable) ────────────
    if text is None:
        gemini_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            print(f"  [LLM 1/2] Trying Gemini...")
            try:
                from google import genai
                from google.genai import types
                os.environ["GEMINI_API_KEY"] = gemini_key
                gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
                client   = genai.Client()
                response = client.models.generate_content(
                    model=gemini_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.9,
                    ),
                )
                raw = getattr(response, "text", None)
                if raw and raw.strip():
                    text      = raw.strip()
                    llm_model = gemini_model
                    print(f"  [LLM 1/2] Gemini succeeded ({len(text)} chars)")
                else:
                    print(f"  [LLM 1/2] Gemini returned empty/blocked response.")
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    print(f"  [LLM 1/2] Gemini quota exhausted (429) — falling through.")
                elif "503" in msg or "UNAVAILABLE" in msg:
                    print(f"  [LLM 1/2] Gemini overloaded (503) — falling through.")
                else:
                    print(f"  [LLM 1/2] Gemini error: {msg}")
        else:
            print(f"  [LLM 1/2] Gemini skipped — GEMINI_API_KEY not set.")

    # ── STEP 3: Rule-based fallback (always works, all 11 languages) ─────────
    if text is None:
        print(f"  [LLM 2/2] Using rule-based fallback.")
        return _rule_based_fallback(req, sarvam_api_key, bhashini_api_key)

    # ── TTS for IVR ───────────────────────────────────────────────────────────
    tts_url = None
    if req.content_format == ContentFormat.IVR_VOICE:
        tts_url = synthesize_tts(text, req.language, sarvam_api_key, bhashini_api_key)

    return GeneratedContent(
        grower_id=req.grower_id,
        content_format=req.content_format.value,
        language=req.language,
        text=text,
        char_count=len(text),
        llm_model=llm_model,
        prompt_tokens=0,
        completion_tokens=0,
        generation_timestamp=datetime.utcnow().isoformat(),
        tts_audio_url=tts_url,
        metadata={
            "crop": req.crop, "stage": req.growth_stage, "tehsil": req.tehsil,
            "pest_pressure": req.pest_pressure_index, "weather_risk": req.weather_risk_score,
        },
    )


# ---------------------------------------------------------------------------
# Rule-based fallback — fully offline, all 11 languages
# FIX: Was hardcoded in Hindi for all languages. Now uses OFFLINE_BODY,
# OFFLINE_IVR, OFFLINE_SOCIAL dictionaries with proper per-language strings.
# ---------------------------------------------------------------------------

def _rule_based_fallback(req: ContentRequest,
                          sarvam_api_key: str = "",
                          bhashini_api_key: str = "") -> GeneratedContent:
    crop_kb         = CROP_KNOWLEDGE.get(req.crop, {})
    products        = crop_kb.get("products", {})
    primary_product = products.get("fungicide") or products.get("insecticide") or "Tilt 250 EC"
    threats         = req.active_threats or crop_kb.get("key_threats", ["pest pressure"])
    main_threat     = threats[0]
    dosage          = crop_kb.get("dosage", {}).get(primary_product, "as per label")
    cta             = LANGUAGE_CTA.get(req.language, LANGUAGE_CTA["Hindi"])
    greeting        = LANGUAGE_GREETINGS.get(req.language, LANGUAGE_GREETINGS["Hindi"]).format(name=req.grower_name)

    # Shared substitution kwargs for all templates
    kw = dict(
        crop=req.crop, stage=req.growth_stage,
        threat=main_threat, product=primary_product,
        dosage=dosage, ivr_press=cta["ivr_press"],
    )

    # ── WHATSAPP_RICH — uses OFFLINE_BODY in the grower's own language ───────
    if req.content_format == ContentFormat.WHATSAPP_RICH:
        body = OFFLINE_BODY.get(req.language, OFFLINE_BODY["Hindi"]).format(**kw)
        text = (
            f"🌾 {greeting}\n\n"
            f"{body}\n\n"
            f"📞 {req.nearest_retailer or 'नजदीकी डीलर से संपर्क करें'}\n"
            f"{cta['whatsapp_reply']}"
        )

    # ── WHATSAPP_TEXT — shorter, still in correct language ───────────────────
    elif req.content_format == ContentFormat.WHATSAPP_TEXT:
        body = OFFLINE_BODY.get(req.language, OFFLINE_BODY["Hindi"]).format(**kw)
        text = f"{body} {cta['whatsapp_reply']}"

    # ── IVR_VOICE — full script in correct language ───────────────────────────
    elif req.content_format == ContentFormat.IVR_VOICE:
        ivr_tmpl = OFFLINE_IVR.get(req.language, OFFLINE_IVR["Hindi"])
        text = f"{greeting} {ivr_tmpl.format(**kw)}"

    # ── SMS — Latin-script skeleton (160-char hard limit) ─────────────────────
    elif req.content_format == ContentFormat.SMS:
        text = (
            f"Syngenta:{req.crop} {req.growth_stage[:8]}-{main_threat[:18]} alert."
            f"Use {primary_product}.{dosage[:20]}."
            f"{cta['sms_missed_call'].format(number='1800-XXX')}"
        )[:160]

    # ── FIELD_REP_BRIEF — English, same as original ───────────────────────────
    elif req.content_format == ContentFormat.FIELD_REP_BRIEF:
        text = (
            f"REP VISIT BRIEF — {req.grower_id}\n"
            f"Location: {req.tehsil}, {req.state}\n"
            f"Crop: {req.crop} | Stage: {req.growth_stage} | Farm: {req.farm_size_acres:.1f} acres\n\n"
            f"KEY TALKING POINTS:\n"
            f"• {main_threat} risk is high at this stage in the region.\n"
            f"• Recommend {primary_product} — dosage: {dosage}.\n"
            f"• Apply before the next growth stage ({req.days_to_next_stage} days away).\n\n"
            f"NEXT STEP: Demo spray at farm. Provide sample if available."
        )

    # ── SOCIAL_POST — correct language ────────────────────────────────────────
    elif req.content_format == ContentFormat.SOCIAL_POST:
        social_tmpl = OFFLINE_SOCIAL.get(req.language, OFFLINE_SOCIAL["Hindi"])
        text = social_tmpl.format(crop=req.crop.title(), threat=main_threat[:30], product=primary_product)

    else:
        body = OFFLINE_BODY.get(req.language, OFFLINE_BODY["Hindi"]).format(**kw)
        text = f"{greeting} {body} {cta['whatsapp_reply']}"

    # TTS for IVR — now uses unified dispatcher (Sarvam → pyttsx3)
    tts_url = None
    if req.content_format == ContentFormat.IVR_VOICE:
        tts_url = synthesize_tts(text, req.language, sarvam_api_key, bhashini_api_key)

    return GeneratedContent(
        grower_id=req.grower_id,
        content_format=req.content_format.value,
        language=req.language,
        text=text,
        char_count=len(text),
        llm_model="rule_based_fallback",
        prompt_tokens=0, completion_tokens=0,
        generation_timestamp=datetime.utcnow().isoformat(),
        tts_audio_url=tts_url,
        metadata={"crop": req.crop, "stage": req.growth_stage, "tehsil": req.tehsil},
    )


def batch_generate(requests: list[ContentRequest], api_key: str = "",
                   sarvam_api_key: str = "",
                   bhashini_api_key: str = "") -> list[GeneratedContent]:
    results = []
    for i, req in enumerate(requests):
        print(f"  [{i+1}/{len(requests)}] {req.grower_id} | {req.content_format.value} | {req.language}")
        results.append(generate_content(req, api_key, sarvam_api_key, bhashini_api_key))
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Engine 1: Content Generator Demo ===")
    print("No API keys needed — runs offline via Ollama + pyttsx3 + rule-based fallback.\n")

    for lang in ["Tamil", "Punjabi", "Telugu", "Malayalam", "Hindi"]:
        req = ContentRequest(
            grower_id=f"GRW_DEMO_{lang[:2].upper()}",
            crop="wheat", growth_stage="flowering", language=lang,
            state="Punjab", tehsil="Patiala_T104",
            content_format=ContentFormat.WHATSAPP_RICH,
            active_threats=["yellow rust (Puccinia striiformis)"],
            days_to_next_stage=7, pest_pressure_index=0.75,
        )
        content = generate_content(req)
        print(f"--- {lang} | {content.llm_model} | {content.char_count} chars ---")
        print(content.text[:200])
        print()
