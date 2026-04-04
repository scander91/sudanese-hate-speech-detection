#!/usr/bin/env python3
"""
Weak Supervision Pipeline v3 — CORRECTED
==========================================
Fixes from v2 data analysis:
1. Removed 'زق','زب' (matched مرتزقة/حزب — 1500+ false positives)
2. Added 'كلب' back with variants (71 missed sentences)
3. Tribal slur now ALWAYS votes HATE (not abstaining on news context)
4. Moved 'دمر' from incitement to reporting (37 false HATE in news)
5. Added political slur → OFFENSIVE standalone LF (87 missed)
6. Tightened lf_no_keywords to require positive neutral signal
7. Added morphological variants (طيزو, ولاد الكلب, بنت كلب, etc.)

No GPU needed. CPU only. ~5-10 min for 40K sentences.
Usage: python3 snorkel_pipeline.py --all
"""

import os, re, json, numpy as np, argparse
from collections import Counter
from datetime import datetime

ABSTAIN, HATE, OFFENSIVE, NEUTRAL = -1, 0, 1, 2
LABEL_NAMES = {HATE: "HATE", OFFENSIVE: "OFFENSIVE", NEUTRAL: "NEUTRAL", ABSTAIN: "ABSTAIN"}
CORPUS_PATH = "data/labeling_corpus/labeling_corpus_25k.json"
OUTPUT_DIR = "data/labeling_corpus/weak_supervision_v3"

# ═══════════════════════════════════════════════════════════
# KEYWORDS — v3 CORRECTED
# ═══════════════════════════════════════════════════════════

KW_TRIBAL_SLURS = [
    'جنجويد', 'جنجويدي', 'جنجاويد', 'جنجويدية',
    'عبيد', 'زرقة', 'يا ازرق',
    'زنوج', 'زنجي', 'يا زنجي',
    'فلاتة', 'فلاتي', 'يا فلاتي',
    'همباته', 'شلاليف',
]

KW_TRIBAL_NAMES = [
    'فوراوي', 'فوراوية', 'مساليتي', 'مساليتية',
    'زغاوي', 'زغاوية', 'نوباوي', 'نوباوية',
    'بجاوي', 'بجاوية', 'جعلي', 'جعلية',
    'شايقي', 'شايقية', 'دنقلاوي', 'دنقلاوية',
    'تعايشي', 'بقاري', 'حمري', 'مسيري', 'رزيقي', 'كباشي',
    'الفور', 'المساليت', 'الزغاوة', 'النوبة', 'البجا',
    'الجعليين', 'الشايقية', 'الدناقلة',
    'التعايشة', 'البقارة', 'المسيرية',
    'الرشايدة', 'الرزيقات', 'الكبابيش', 'الحوازمة', 'المعاليا',
    'البني عامر', 'الهدندوة',
    'غرابة', 'شراقة', 'اولاد البحر', 'اولاد الغرب',
    'عرب جزيرة', 'النهروالبحر', 'افارقة',
]

KW_POLITICAL_SLURS = [
    'كوز', 'كيزان', 'كوزي', 'الكيزان', 'كوزية',
    'اخوانجي', 'اخوانجية',
    'دعامي', 'دعامية', 'حميداتي',
    'جيشجي', 'جيشجية',
    'فلول', 'تمكيني', 'تمكينية', 'طفيلي',
    'مندس', 'مندسين', 'طابور خامس',
    'مرتزق', 'مرتزقة', 'المرتزقة',
    'خائن', 'خونة', 'خيانة',
    'عميل', 'عملاء', 'عمالة',
    'سمسار', 'سماسرة', 'الحمدابي',
]

# v3 FIX: Added 'كلب' back with variants, added morphological forms
KW_DEHUMANIZING = [
    'كلب', 'كلاب', 'ابن الكلب', 'ود الكلب', 'يا كلب',
    'اولاد الكلب', 'ولاد الكلب', 'بنت كلب', 'بنت ستين كلب',
    'حيوان', 'حيوانات', 'بهيمة', 'بهائم',
    'حمار', 'حمير', 'يا حمار',
    'خنزير', 'خنازير',
    'قرد', 'قرود', 'صرصور', 'صراصير',
    'جرذ', 'جرذان', 'حشرة', 'حشرات',
    'ذباب', 'دبانة', 'فار', 'فيران',
    'وحش', 'وحوش', 'شيطان', 'شياطين',
    'ملعون', 'ملاعين',
    'همجي', 'همجية', 'همج', 'بربري', 'بربر', 'متوحش', 'متوحشين',
    'زبالة', 'قمامة', 'قذارة', 'قذر', 'نجس', 'نجاسة',
    'سرطان', 'وباء', 'طاعون',
    'حشاش', 'حشاشين',
    'خروف', 'نعجه', 'كديسة', 'فارة', 'فاره', 'جمار', 'لبوه',
    'بعاتي',
]

# v3 FIX: Moved 'دمر','دمروا' to REPORTING. Kept only clear incitement forms.
KW_VIOLENCE_INCITEMENT = [
    'اقتل', 'اقتلوا', 'اقتلوهم', 'نقتل', 'نقتلهم',
    'اذبح', 'اذبحوا', 'اذبحوهم', 'نذبح',
    'احرق', 'احرقوا', 'احرقوهم', 'نحرق',
    'ابيد', 'ابيدوا', 'ابيدوهم',
    'تطهير عرقي', 'ابادة',
    'اغتصب', 'اغتصبوا',
    'اطرد', 'اطردوا', 'اطردوهم',
    'نضفوهم', 'نظفوا', 'شيلوهم', 'شيلوا',
    'كبوا عليهم', 'خشوا فيهم',
    'ادوهم درس', 'ادوهم قفا',
    'فتك', 'متك', 'ولضم', 'ولضمي', 'سوخوي',
    'امسحوهم', 'اكسحوهم', 'اكنسوهم',
    'خلوا يموتوا', 'سيبوهم يموتوا',
    'لازم نخلص', 'يجب ابادة', 'يجب ابادتهم',
    'دمروهم', 'ندمر',  # Only clear incitement forms
    'هجر', 'هجروهم',
]

# v3 FIX: Removed 'زق','زب' (matched مرتزقة/حزب). Added morphological variants.
KW_PROFANITY = [
    'كسم', 'كسمك', 'كس امك', 'كس ام', 'كسك', 'كسوم',
    'طيزك', 'طيزو', 'طيزه',  # v3: added variants
    'بيضاتك', 'قلقاتك', 'نهودك', 'شطور', 'شطورك', 'لبنك',
    'جملك', 'جملها', 'نخريك',
    'احشك', 'حشاه', 'حشاك', 'حشرو', 'حشروليك', 'حشروه',
    'اتناك', 'ناكك', 'زعط', 'زعطك',
    'مبلول', 'شمبورة', 'شمبوره', 'ماسورة', 'دعاره',
    'زفرة', 'زفارة', 'زوط',
    'خول', 'لوطي', 'بتلاوط', 'عجلاتي',
    'شرموط', 'شرموطة', 'شراميط',
    'عرص', 'عرصة', 'منيك', 'منيكة',
    'ابن الشرموطة', 'ود الشرموطة',
    'ابن الحرام', 'ود الحرام', 'قحبة', 'قحاب',
    'خرا', 'خرى', 'خره', 'تفو',
    'وسخان', 'ابوك', 'امك',
    # v3: REMOVED 'زق','زب','بل' (too many false positives)
]

KW_INSULTS = [
    'اهبل', 'مستهبل', 'تتهابل',
    'ابلده', 'دلاهة', 'هبنقه', 'هبنقات',
    'مسطول', 'تمبرله', 'غشيم',
    'سجم', 'سجمانه', 'عوير', 'عوين',
    'خرمان', 'عرمان', 'كدكه', 'كدكة',
    'عواره', 'كلجة', 'جعان', 'فلقاي',
    'منطط', 'قرقدي', 'قلوط',
    'شليق', 'قنيطة', 'قنيط',
    'شكش', 'شكشة', 'شكشه', 'ابصلعه',
    'خايب', 'باطل', 'سفيه',
    'جعجاع', 'قحاطة', 'تافه', 'تافهين',
    'سندالة', 'سندالتك',
    'ماكويس', 'ماتمام',
    'صاج', 'صيج', 'صيجليك', 'فلنقايات',
    'حقير', 'حقراء', 'وضيع',
    'جبان', 'جبناء', 'خسيس', 'خساس',
    'نذل', 'انذال', 'سافل', 'سفلة', 'غبي', 'جهلة',
    'تهنق', 'تكاكي', 'تهر', 'تملص', 'تتشالق',
    'بتهنجك', 'شاميك', 'شميتك', 'ظعاط',
    'عب', 'خطرات', 'جعبات', 'شفشفة',
    'عرد', 'جغم', 'طيرو', 'بالابجدية', 'ببلك',
    'خمسين', 'عزبا', 'عزبه',
    'خالة', 'خالات', 'مطفجه', 'نعجه ساي',
    'سبهلل', 'كبكابة',
]

KW_THREATS = [
    'تهديد', 'هددوا', 'نهدد', 'توعد', 'توعدوا',
    'انتقام', 'ننتقم', 'انتقموا', 'ثأر', 'نثأر',
    'عقاب', 'نعاقب', 'يوم الحساب',
    'ويل لهم', 'مصيرهم',
    'سندمر', 'ستندم', 'سيندمون',
    'والله نكسر', 'والله نوريك',
]

KW_NEWS = [
    'عاجل', 'بيان', 'بيان صحفي', 'بيان رسمي',
    'اعلن', 'اعلنت', 'صرح', 'صرحت', 'تصريح', 'تصريحات',
    'مصادر', 'مصادر ميدانية', 'مصادر موثوقة',
    'تقرير', 'تقارير', 'مراسل', 'مراسلنا',
    'وكالة', 'مؤتمر صحفي',
    'الناطق الرسمي', 'المتحدث باسم',
    'بحسب', 'وفقا', 'نقلا عن', 'افادت', 'افاد',
    'رصد ومتابعة',
]

KW_HUMANITARIAN = [
    'مساعدات', 'اغاثة', 'اغاثية',
    'مجاعة', 'جوع', 'جوعانين', 'عطش',
    'دواء', 'ادوية', 'مستشفى', 'مستشفيات',
    'ازمة انسانية', 'منظمة', 'منظمات',
    'الصليب الاحمر', 'الهلال الاحمر',
    'الامم المتحدة', 'اليونيسف',
    'وقف اطلاق النار', 'هدنة',
    'مفاوضات', 'تفاوض', 'سلام',
    'نازح', 'نازحين', 'لاجئ', 'لاجئين',
    'معسكر نازحين', 'ايواء', 'مأوى',
]

# v3 FIX: Added 'دمر','دمرت','تدمير' here (moved from incitement)
KW_VIOLENCE_REPORTING = [
    'قتل', 'مقتل', 'قتلى', 'قتيل',
    'شهيد', 'شهداء', 'استشهد',
    'اصابة', 'اصابات', 'جرحى', 'مصاب',
    'قصف', 'قصفت', 'غارة', 'غارات',
    'اشتباك', 'اشتباكات', 'معارك', 'معركة',
    'هجوم', 'هجمات',
    'مجزرة', 'مجازر', 'انتهاكات', 'انتهاك',
    'تعذيب', 'نهب', 'نهبوا', 'سلب',
    'حصار', 'محاصرة', 'اختطاف', 'خطفوا',
    'كمين', 'كمائن',
    'حرق', 'حريق', 'دمار', 'تدمير',
    'دمر', 'دمرت', 'دمروا', 'مدمر',  # v3: moved from incitement
]

KW_WAR_PARTIES = [
    'الدعم السريع', 'قوات الدعم السريع',
    'القوات المسلحة', 'الجيش السوداني',
    'حميدتي', 'البرهان', 'دقلو',
    'المليشيا', 'مليشيا', 'مليشيات',
    'مسيرات', 'طائرات مسيرة', 'بايركتار',
]

KW_DIALECT = [
    'شنو', 'هسع', 'هسة', 'هسي', 'زول', 'زولة',
    'كده', 'ليك', 'ليكي', 'ليهو', 'ليها',
    'بتاع', 'بتاعت', 'بتاعي',
    'ياخ', 'يا زول', 'عشان', 'بالله',
    'داير', 'عايز', 'ماشي', 'جاي', 'راجع',
    'سمح', 'كويس', 'تمام',
    'غايتو', 'تاني', 'برضو', 'كمان',
    'ملاح', 'عيشة', 'كسرة', 'قراصة',
    'ويكة', 'بامية', 'روب', 'جبنة',
    'حوش', 'راكوبة', 'عنقريب',
    'اها', 'ايوا', 'خلاص', 'يلا',
]

KW_RELIGIOUS = [
    'كافر', 'كفار', 'كافرين', 'مرتد', 'مرتدين',
    'منافق', 'منافقين', 'مشرك', 'مشركين',
    'ملحد', 'ملحدين', 'زنديق', 'زنادقة',
]

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def has_any(text, keywords):
    for kw in keywords:
        if kw in text:
            return True
    return False

def count_matches(text, keywords):
    return sum(1 for kw in keywords if kw in text)

# All negative keywords combined (for no-keyword check)
ALL_NEGATIVE_KW = (KW_TRIBAL_SLURS + KW_TRIBAL_NAMES + KW_POLITICAL_SLURS +
    KW_DEHUMANIZING + KW_VIOLENCE_INCITEMENT + KW_PROFANITY +
    KW_INSULTS + KW_THREATS + KW_RELIGIOUS)

ALL_CONTEXT_KW = KW_NEWS + KW_VIOLENCE_REPORTING + KW_WAR_PARTIES + KW_HUMANITARIAN

# ═══════════════════════════════════════════════════════════
# LABELING FUNCTIONS — v3 (42 total)
# ═══════════════════════════════════════════════════════════

# ─── HATE (19) ────────────────────────────────────────────

def lf_tribal_slur_dehumanizing(t):
    if has_any(t, KW_TRIBAL_SLURS) and has_any(t, KW_DEHUMANIZING): return HATE
    return ABSTAIN

def lf_tribal_slur_profanity(t):
    if has_any(t, KW_TRIBAL_SLURS) and has_any(t, KW_PROFANITY): return HATE
    return ABSTAIN

def lf_tribal_slur_violence(t):
    if has_any(t, KW_TRIBAL_SLURS) and has_any(t, KW_VIOLENCE_INCITEMENT): return HATE
    return ABSTAIN

# v3 FIX: tribal slur ALWAYS votes HATE (no longer abstains for news)
def lf_tribal_slur_always(t):
    """Tribal slur present = HATE regardless of context"""
    if has_any(t, KW_TRIBAL_SLURS): return HATE
    return ABSTAIN

def lf_tribal_name_dehumanizing(t):
    if has_any(t, KW_TRIBAL_NAMES) and has_any(t, KW_DEHUMANIZING): return HATE
    return ABSTAIN

def lf_tribal_name_profanity(t):
    if has_any(t, KW_TRIBAL_NAMES) and has_any(t, KW_PROFANITY): return HATE
    return ABSTAIN

def lf_tribal_name_violence(t):
    if has_any(t, KW_TRIBAL_NAMES) and has_any(t, KW_VIOLENCE_INCITEMENT): return HATE
    return ABSTAIN

def lf_political_dehumanizing(t):
    if has_any(t, KW_POLITICAL_SLURS) and has_any(t, KW_DEHUMANIZING): return HATE
    return ABSTAIN

def lf_political_profanity(t):
    if has_any(t, KW_POLITICAL_SLURS) and has_any(t, KW_PROFANITY): return HATE
    return ABSTAIN

def lf_political_violence(t):
    if has_any(t, KW_POLITICAL_SLURS) and has_any(t, KW_VIOLENCE_INCITEMENT): return HATE
    return ABSTAIN

def lf_violence_incitement_alone(t):
    if has_any(t, KW_VIOLENCE_INCITEMENT): return HATE
    return ABSTAIN

def lf_multiple_tribal_slurs(t):
    if count_matches(t, KW_TRIBAL_SLURS) >= 2: return HATE
    return ABSTAIN

def lf_dehumanizing_violence(t):
    if has_any(t, KW_DEHUMANIZING) and has_any(t, KW_VIOLENCE_INCITEMENT): return HATE
    return ABSTAIN

def lf_religious_violence(t):
    if has_any(t, KW_RELIGIOUS) and has_any(t, KW_VIOLENCE_INCITEMENT): return HATE
    return ABSTAIN

def lf_religious_dehumanizing(t):
    if has_any(t, KW_RELIGIOUS) and has_any(t, KW_DEHUMANIZING): return HATE
    return ABSTAIN

def lf_threats_group(t):
    if has_any(t, KW_THREATS) and (has_any(t, KW_TRIBAL_NAMES) or has_any(t, KW_TRIBAL_SLURS) or has_any(t, KW_POLITICAL_SLURS)):
        return HATE
    return ABSTAIN

def lf_multi_hate_signals(t):
    s = sum([has_any(t, KW_TRIBAL_SLURS), has_any(t, KW_DEHUMANIZING),
             has_any(t, KW_VIOLENCE_INCITEMENT), has_any(t, KW_PROFANITY), has_any(t, KW_THREATS)])
    return HATE if s >= 3 else ABSTAIN

# v3 NEW: dehumanizing + political slur = HATE
def lf_klb_with_group(t):
    """Dehumanizing (incl كلب) + any group reference = HATE"""
    if has_any(t, KW_DEHUMANIZING) and (has_any(t, KW_TRIBAL_NAMES) or has_any(t, KW_POLITICAL_SLURS) or has_any(t, KW_WAR_PARTIES)):
        return HATE
    return ABSTAIN

# v3 NEW: profanity + war party = HATE (not just OFFENSIVE)
def lf_profanity_warparty_hate(t):
    """Profanity directed at war parties with insults = HATE"""
    if has_any(t, KW_PROFANITY) and has_any(t, KW_WAR_PARTIES) and has_any(t, KW_INSULTS):
        return HATE
    return ABSTAIN

# ─── OFFENSIVE (9) ────────────────────────────────────────

def lf_profanity_no_group(t):
    if has_any(t, KW_PROFANITY) and not has_any(t, KW_TRIBAL_SLURS) and not has_any(t, KW_TRIBAL_NAMES) and not has_any(t, KW_POLITICAL_SLURS):
        return OFFENSIVE
    return ABSTAIN

def lf_insults_alone(t):
    if has_any(t, KW_INSULTS): return OFFENSIVE
    return ABSTAIN

def lf_threats_no_group(t):
    if has_any(t, KW_THREATS) and not has_any(t, KW_TRIBAL_NAMES) and not has_any(t, KW_TRIBAL_SLURS):
        return OFFENSIVE
    return ABSTAIN

def lf_dehumanizing_no_group(t):
    if has_any(t, KW_DEHUMANIZING) and not has_any(t, KW_TRIBAL_SLURS) and not has_any(t, KW_TRIBAL_NAMES) and not has_any(t, KW_POLITICAL_SLURS):
        return OFFENSIVE
    return ABSTAIN

def lf_profanity_insults(t):
    if has_any(t, KW_PROFANITY) and has_any(t, KW_INSULTS): return OFFENSIVE
    return ABSTAIN

def lf_religious_no_violence(t):
    if has_any(t, KW_RELIGIOUS) and not has_any(t, KW_VIOLENCE_INCITEMENT) and not has_any(t, KW_DEHUMANIZING):
        return OFFENSIVE
    return ABSTAIN

def lf_multi_profanity(t):
    if count_matches(t, KW_PROFANITY) >= 2: return OFFENSIVE
    return ABSTAIN

# v3 FIX: political slur alone = OFFENSIVE (not NEUTRAL)
def lf_political_slur_at_least_offensive(t):
    """Political slurs are at LEAST offensive even without profanity"""
    if has_any(t, KW_POLITICAL_SLURS): return OFFENSIVE
    return ABSTAIN

def lf_insults_overrides_neutral(t):
    if has_any(t, KW_INSULTS) and has_any(t, KW_DIALECT): return OFFENSIVE
    return ABSTAIN

# ─── NEUTRAL (10) ────────────────────────────────────────

def lf_news_no_hate(t):
    if has_any(t, KW_NEWS) and not has_any(t, KW_PROFANITY) and not has_any(t, KW_TRIBAL_SLURS) and not has_any(t, KW_DEHUMANIZING) and not has_any(t, KW_INSULTS) and not has_any(t, KW_POLITICAL_SLURS):
        return NEUTRAL
    return ABSTAIN

def lf_humanitarian_no_hate(t):
    if has_any(t, KW_HUMANITARIAN) and not has_any(t, KW_PROFANITY) and not has_any(t, KW_TRIBAL_SLURS) and not has_any(t, KW_VIOLENCE_INCITEMENT) and not has_any(t, KW_DEHUMANIZING):
        return NEUTRAL
    return ABSTAIN

def lf_reporting_plus_news(t):
    if has_any(t, KW_VIOLENCE_REPORTING) and has_any(t, KW_NEWS) and not has_any(t, KW_TRIBAL_SLURS) and not has_any(t, KW_PROFANITY):
        return NEUTRAL
    return ABSTAIN

def lf_war_party_news(t):
    if has_any(t, KW_WAR_PARTIES) and has_any(t, KW_NEWS) and not has_any(t, KW_PROFANITY) and not has_any(t, KW_TRIBAL_SLURS) and not has_any(t, KW_DEHUMANIZING):
        return NEUTRAL
    return ABSTAIN

# v3 FIX: require dialect markers AND no negative keywords
def lf_dialect_no_hate(t):
    if has_any(t, KW_DIALECT) and not has_any(t, ALL_NEGATIVE_KW):
        return NEUTRAL
    return ABSTAIN

# v3 FIX: tightened — require POSITIVE neutral signal (dialect or humanitarian)
def lf_no_negative_with_positive(t):
    """No negative keywords + has positive signal (dialect/humanitarian) = NEUTRAL"""
    if not has_any(t, ALL_NEGATIVE_KW) and (has_any(t, KW_DIALECT) or has_any(t, KW_HUMANITARIAN)):
        return NEUTRAL
    return ABSTAIN

# v3 FIX: pure no-keywords only for very short sentences
def lf_short_clean(t):
    """Short sentence (5-8 words), no negative keywords = NEUTRAL"""
    if len(t.split()) <= 8 and not has_any(t, ALL_NEGATIVE_KW) and not has_any(t, ALL_CONTEXT_KW):
        return NEUTRAL
    return ABSTAIN

def lf_humanitarian_dialect(t):
    if has_any(t, KW_HUMANITARIAN) and has_any(t, KW_DIALECT) and not has_any(t, ALL_NEGATIVE_KW):
        return NEUTRAL
    return ABSTAIN

def lf_reporting_no_hate(t):
    if has_any(t, KW_VIOLENCE_REPORTING) and not has_any(t, KW_PROFANITY) and not has_any(t, KW_TRIBAL_SLURS) and not has_any(t, KW_DEHUMANIZING) and not has_any(t, KW_VIOLENCE_INCITEMENT) and not has_any(t, KW_INSULTS) and not has_any(t, KW_POLITICAL_SLURS):
        return NEUTRAL
    return ABSTAIN

def lf_tribal_name_news(t):
    if has_any(t, KW_TRIBAL_NAMES) and has_any(t, KW_NEWS) and not has_any(t, KW_PROFANITY) and not has_any(t, KW_DEHUMANIZING) and not has_any(t, KW_VIOLENCE_INCITEMENT):
        return NEUTRAL
    return ABSTAIN

# ─── Conflict resolution (2) ─────────────────────────────

def lf_hate_overrides_news(t):
    hate_n = count_matches(t, KW_TRIBAL_SLURS) + count_matches(t, KW_DEHUMANIZING) + count_matches(t, KW_VIOLENCE_INCITEMENT)
    if has_any(t, KW_NEWS) and hate_n >= 2: return HATE
    return ABSTAIN

def lf_dehumanizing_standalone_offensive(t):
    """Dehumanizing term alone (even without group) = at least OFFENSIVE"""
    if has_any(t, KW_DEHUMANIZING) and not has_any(t, KW_NEWS): return OFFENSIVE
    return ABSTAIN


# ═══════════════════════════════════════════════════════════
# ALL LFs
# ═══════════════════════════════════════════════════════════

ALL_LFS = [
    # HATE (19)
    ("lf_tribal_slur_dehumanizing", lf_tribal_slur_dehumanizing),
    ("lf_tribal_slur_profanity", lf_tribal_slur_profanity),
    ("lf_tribal_slur_violence", lf_tribal_slur_violence),
    ("lf_tribal_slur_always", lf_tribal_slur_always),
    ("lf_tribal_name_dehumanizing", lf_tribal_name_dehumanizing),
    ("lf_tribal_name_profanity", lf_tribal_name_profanity),
    ("lf_tribal_name_violence", lf_tribal_name_violence),
    ("lf_political_dehumanizing", lf_political_dehumanizing),
    ("lf_political_profanity", lf_political_profanity),
    ("lf_political_violence", lf_political_violence),
    ("lf_violence_incitement_alone", lf_violence_incitement_alone),
    ("lf_multiple_tribal_slurs", lf_multiple_tribal_slurs),
    ("lf_dehumanizing_violence", lf_dehumanizing_violence),
    ("lf_religious_violence", lf_religious_violence),
    ("lf_religious_dehumanizing", lf_religious_dehumanizing),
    ("lf_threats_group", lf_threats_group),
    ("lf_multi_hate_signals", lf_multi_hate_signals),
    ("lf_klb_with_group", lf_klb_with_group),
    ("lf_profanity_warparty_hate", lf_profanity_warparty_hate),
    # OFFENSIVE (9)
    ("lf_profanity_no_group", lf_profanity_no_group),
    ("lf_insults_alone", lf_insults_alone),
    ("lf_threats_no_group", lf_threats_no_group),
    ("lf_dehumanizing_no_group", lf_dehumanizing_no_group),
    ("lf_profanity_insults", lf_profanity_insults),
    ("lf_religious_no_violence", lf_religious_no_violence),
    ("lf_multi_profanity", lf_multi_profanity),
    ("lf_political_slur_at_least_offensive", lf_political_slur_at_least_offensive),
    ("lf_insults_overrides_neutral", lf_insults_overrides_neutral),
    # NEUTRAL (10)
    ("lf_news_no_hate", lf_news_no_hate),
    ("lf_humanitarian_no_hate", lf_humanitarian_no_hate),
    ("lf_reporting_plus_news", lf_reporting_plus_news),
    ("lf_war_party_news", lf_war_party_news),
    ("lf_dialect_no_hate", lf_dialect_no_hate),
    ("lf_no_negative_with_positive", lf_no_negative_with_positive),
    ("lf_short_clean", lf_short_clean),
    ("lf_humanitarian_dialect", lf_humanitarian_dialect),
    ("lf_reporting_no_hate", lf_reporting_no_hate),
    ("lf_tribal_name_news", lf_tribal_name_news),
    # CONFLICT (2)
    ("lf_hate_overrides_news", lf_hate_overrides_news),
    ("lf_dehumanizing_standalone_offensive", lf_dehumanizing_standalone_offensive),
]


# ═══════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════

def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["text"] for item in data], data

def apply_lfs(texts, lfs):
    n, m = len(texts), len(lfs)
    L = np.full((n, m), ABSTAIN, dtype=int)
    for j, (name, func) in enumerate(lfs):
        for i, text in enumerate(texts):
            L[i, j] = func(text)
        if (j + 1) % 10 == 0:
            print(f"    {j+1}/{m} done...")
    return L

def analyze_lfs(L, lf_names):
    n, m = L.shape
    print(f"\n  {'LF Name':<42} {'Cov%':>6} {'#Lab':>7} {'HATE':>6} {'OFF':>6} {'NEU':>6}")
    print(f"  {'─'*75}")
    stats = []
    for j in range(m):
        col = L[:, j]
        lab = int(np.sum(col != ABSTAIN))
        cov = 100 * lab / n
        h, o, ne = int(np.sum(col==HATE)), int(np.sum(col==OFFENSIVE)), int(np.sum(col==NEUTRAL))
        stats.append({"name": lf_names[j], "coverage": round(cov,2), "labeled": lab, "hate": h, "offensive": o, "neutral": ne})
        print(f"  {lf_names[j]:<42} {cov:>5.1f}% {lab:>7,} {h:>6,} {o:>6,} {ne:>6,}")

    any_v = int(np.sum(np.any(L != ABSTAIN, axis=1)))
    no_v = n - any_v
    conflicts = sum(1 for i in range(n) if len(set(L[i][L[i]!=ABSTAIN])) > 1)
    print(f"\n  At least 1 vote: {any_v:,} ({100*any_v/n:.1f}%)")
    print(f"  No votes: {no_v:,} ({100*no_v/n:.1f}%)")
    print(f"  Conflicts: {conflicts:,} ({100*conflicts/n:.1f}%)")
    return stats

def majority_vote(L):
    n = L.shape[0]
    labels = np.full(n, NEUTRAL, dtype=int)
    confs = np.zeros(n)
    for i in range(n):
        votes = L[i][L[i] != ABSTAIN]
        if len(votes) == 0:
            labels[i], confs[i] = NEUTRAL, 0.0
        else:
            vc = Counter(votes)
            best = vc.most_common(1)[0]
            labels[i], confs[i] = best[0], best[1] / len(votes)
    return labels, confs

def weighted_vote(L):
    n, m = L.shape
    w = np.array([1.0 / (np.sum(L[:, j] != ABSTAIN) / n + 0.005) for j in range(m)])
    w = w / w.sum() * m
    labels = np.full(n, NEUTRAL, dtype=int)
    confs = np.zeros(n)
    for i in range(n):
        sc = {HATE: 0.0, OFFENSIVE: 0.0, NEUTRAL: 0.0}
        tw = 0.0
        for j in range(m):
            if L[i, j] != ABSTAIN:
                sc[L[i, j]] += w[j]
                tw += w[j]
        if tw == 0:
            labels[i], confs[i] = NEUTRAL, 0.0
        else:
            best = max(sc, key=sc.get)
            labels[i], confs[i] = best, sc[best] / tw
    return labels, confs

def save_results(texts, meta, lmv, cmv, lwv, cwv, outdir):
    os.makedirs(outdir, exist_ok=True)
    results = []
    for i in range(len(texts)):
        item = dict(meta[i])
        item.update({"weak_label_mv": int(lmv[i]), "weak_label_mv_name": LABEL_NAMES[lmv[i]],
                      "weak_conf_mv": round(float(cmv[i]),3),
                      "weak_label_wv": int(lwv[i]), "weak_label_wv_name": LABEL_NAMES[lwv[i]],
                      "weak_conf_wv": round(float(cwv[i]),3)})
        results.append(item)
    with open(os.path.join(outdir, "labeled_corpus_weak.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "labeled_corpus_weak.tsv"), "w", encoding="utf-8") as f:
        f.write("id\ttext\tsource\tkeyword_cat\tweak_label\tlabel_name\tconfidence\thuman_label\n")
        for i, r in enumerate(results):
            t = r["text"].replace("\t"," ").replace("\n"," ")
            f.write(f"{i+1}\t{t}\t{r['source']}\t{r['keyword_category']}\t{r['weak_label_wv']}\t{r['weak_label_wv_name']}\t{r['weak_conf_wv']}\t\n")
    print(f"\n  📁 JSON: {outdir}/labeled_corpus_weak.json")
    print(f"  📁 TSV:  {outdir}/labeled_corpus_weak.tsv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not any([args.analyze, args.label, args.all]): args.all = True

    print(f"\n{'='*85}")
    print(f" WEAK SUPERVISION PIPELINE v3 — CORRECTED")
    print(f" LFs: {len(ALL_LFS)} | Labels: HATE(0) OFFENSIVE(1) NEUTRAL(2)")
    print(f" Fixes: زق/زب removed, كلب added, tribal→HATE always, دمر→reporting")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*85}")

    print(f"\n  Loading corpus: {CORPUS_PATH}")
    texts, meta = load_corpus(CORPUS_PATH)
    print(f"  Loaded {len(texts):,} sentences")

    lf_names = [n for n, _ in ALL_LFS]
    print(f"\n  Applying {len(ALL_LFS)} LFs to {len(texts):,} sentences...")
    L = apply_lfs(texts, ALL_LFS)
    print(f"  Label matrix: {L.shape}")

    if args.analyze or args.all:
        print(f"\n{'='*85}")
        print(f" LF ANALYSIS")
        print(f"{'='*85}")
        stats = analyze_lfs(L, lf_names)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "lf_analysis.json"), "w") as f:
            json.dump(stats, f, indent=2)

    if args.label or args.all:
        lmv, cmv = majority_vote(L)
        lwv, cwv = weighted_vote(L)
        print(f"\n{'='*85}")
        print(f" LABEL DISTRIBUTION")
        print(f"{'='*85}")
        for method, lab, con in [("Majority Vote", lmv, cmv), ("Weighted Vote", lwv, cwv)]:
            print(f"\n  {method}:")
            for v in [HATE, OFFENSIVE, NEUTRAL]:
                c = int(np.sum(lab==v))
                ac = float(np.mean(con[lab==v])) if c > 0 else 0
                print(f"    {LABEL_NAMES[v]:<12} {c:>8,} ({100*c/len(lab):>5.1f}%) avg_conf={ac:.3f}")
        agree = int(np.sum(lmv == lwv))
        print(f"\n  MV vs WV agreement: {agree:,} ({100*agree/len(lmv):.1f}%)")
        save_results(texts, meta, lmv, cmv, lwv, cwv, OUTPUT_DIR)

        lo = int(np.sum(cwv < 0.6))
        mi = int(np.sum((cwv >= 0.6) & (cwv < 0.8)))
        hi = int(np.sum(cwv >= 0.8))
        print(f"\n{'='*85}")
        print(f" HUMAN REVIEW PLAN")
        print(f"{'='*85}")
        print(f"  High conf (>=0.8): {hi:>8,} — spot-check 5-10%")
        print(f"  Medium (0.6-0.8):  {mi:>8,} — review 20-30%")
        print(f"  Low (<0.6):        {lo:>8,} — HUMAN LABELS ALL")
        print(f"  Est. effort:       ~{lo + int(mi*0.25):,} sentences")

    print(f"\n{'='*85}")
    print(f" DONE — output: {OUTPUT_DIR}/")
    print(f"{'='*85}\n")

if __name__ == "__main__":
    main()
