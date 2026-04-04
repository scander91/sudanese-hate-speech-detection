#!/usr/bin/env python3
"""
Build Labeling Corpus - UPDATED with expanded keywords + cleaning
==================================================================
- 680+ keywords across 14 categories
- Sudanese dialect markers for preferring dialectal content
- Non-Sudanese location filter (removes Syrian, Egyptian, etc.)
- MSA news filter (removes عاجل formal news)
- 28,000 sentence stratified sample

Run: python3 build_labeling_corpus.py --step 1   (investigate)
     python3 build_labeling_corpus.py --step 2   (extract sample)
"""

import os
import re
import json
import random
import argparse
from collections import Counter, defaultdict
from datetime import datetime

random.seed(42)

# ─────────────────────────────────────────────────────────────
# FILES
# ─────────────────────────────────────────────────────────────
OLD_CORPUS_FILES = {
    "old_telegram_mukh": "data/raw/Sudanese_txt_cleaned_all/Sudanese_txt_cleaned_telegram_mukh.txt",
    "old_telegram_nabigh": "data/raw/Sudanese_txt_cleaned_all/Sudanese_txt_cleaned_telegram_nabigh.txt",
    "old_telegram_yah": "data/raw/Sudanese_txt_cleaned_all/Sudanese_txt_cleaned_telegram_yah.txt",
    "old_twitter": "data/raw/Sudanese_txt_cleaned_all/Sudanese_txt_cleaned_twitter.txt",
}
WAR_CORPUS = "data/raw/Sudanese_new_only/sudanese_telegram_ALL_cleaned.txt"
OUTPUT_DIR = "data/labeling_corpus"

# ─────────────────────────────────────────────────────────────
# KEYWORD CATEGORIES (680+ keywords, 14 categories)
# ─────────────────────────────────────────────────────────────
KEYWORD_CATEGORIES = {
    "hate_tribal": [
        'جنجويد', 'جنجويدي', 'جنجاويد',
        'عبيد', 'عبد', 'زرقة', 'زرق', 'ازرق',
        'فلاتة', 'فلاتي', 'زنوج', 'زنجي',
        'الفور', 'فوراوي', 'المساليت', 'مساليتي',
        'الزغاوة', 'زغاوي', 'النوبة', 'نوباوي',
        'البجا', 'بجاوي', 'الجعليين', 'جعلي',
        'الشايقية', 'شايقي', 'الدناقلة', 'دنقلاوي',
        'الرشايدة', 'رشيدي', 'التعايشة', 'تعايشي',
        'البقارة', 'بقاري', 'الحمر',
        'المسيرية', 'مسيري', 'الهوسا', 'البرنو', 'البرتي',
        'عرب وزرقة', 'افارقة', 'عرب جزيرة',
        'النهروالبحر', 'همباته', 'شلاليف',
    ],
    "hate_political": [
        'كوز', 'كيزان', 'كوزي', 'الكيزان',
        'اخوانجي', 'اخوانجية', 'الاخوان',
        'فلول', 'فلول النظام', 'تمكيني', 'تمكين',
        'دعامي', 'دعامية', 'الدعامة', 'حميداتي',
        'جيشجي', 'جيشجية',
        'خائن', 'خونة', 'خيانة',
        'عميل', 'عملاء', 'عمالة',
        'مرتزق', 'مرتزقة', 'متمرد', 'متمردين', 'تمرد',
        'انقلابي', 'انقلابيين', 'مندس', 'مندسين',
        'طابور خامس', 'بائع', 'باعوا البلد',
        'سمسار', 'سماسرة', 'سفيه', 'الحمدابي',
    ],
    "dehumanizing": [
        'كلب', 'كلاب', 'ابن الكلب', 'ود الكلب', 'يا كلب',
        'حيوان', 'حيوانات', 'بهيمة', 'بهائم',
        'حمار', 'حمير', 'خنزير', 'خنازير',
        'قرد', 'قرود', 'ثعبان', 'ثعابين', 'حية', 'افعى',
        'حشرة', 'حشرات', 'صرصور', 'صراصير',
        'ذباب', 'دبانة', 'جرذ', 'جرذان', 'فار', 'فيران',
        'فيروس', 'وباء', 'طاعون', 'مرض',
        'قذارة', 'قذر', 'وسخ', 'نجس', 'نجاسة',
        'زبالة', 'قمامة', 'سرطان',
        'وحش', 'وحوش', 'شيطان', 'شياطين', 'ملعون', 'ملاعين',
        'حشاش', 'حشاشين', 'سكران', 'سكارى',
        'همجي', 'همجية', 'همج', 'بربري', 'بربر',
        'متوحش', 'متوحشين',
        'خروف', 'نعجه', 'لبوه', 'كديسة',
        'فارة', 'فاره', 'جمار',
    ],
    "violence_incitement": [
        'اقتل', 'اقتلوا', 'اقتلوهم', 'نقتل', 'نقتلهم',
        'اذبح', 'اذبحوا', 'اذبحوهم', 'نذبح',
        'احرق', 'احرقوا', 'احرقوهم', 'نحرق',
        'دمر', 'دمروا', 'دمروهم', 'ندمر',
        'ابيد', 'ابيدوا', 'ابيدوهم', 'ابادة',
        'اغتصب', 'اغتصبوا',
        'طهر', 'تطهير', 'تطهير عرقي',
        'اطرد', 'اطردوا', 'اطردوهم', 'هجر', 'هجروهم',
        'لازم نخلص', 'لازم ننهي',
        'يجب ابادة', 'يجب ابادتهم',
        'خلوا يموتوا', 'سيبوهم يموتوا',
        'كبوا عليهم', 'خشوا فيهم',
        'ادوهم درس', 'ادوهم قفا',
        'شيلوهم', 'شيلوا', 'نضفوهم', 'نظفوا',
        'فتك', 'متك', 'ولضم', 'ولضمي', 'سوخوي',
    ],
    "violence_reporting": [
        'قتل', 'مقتل', 'قتلى', 'قتيل', 'قتلوا',
        'ذبح', 'مذبحة', 'ذبحوا',
        'حرق', 'حرقوا', 'محروق', 'حريق',
        'دمار', 'تدمير', 'دمرت', 'دمروا', 'مدمر',
        'موت', 'وفاة', 'توفي', 'فارق الحياة',
        'شهيد', 'شهداء', 'استشهد',
        'اصابة', 'اصابات', 'جرحى', 'جريح', 'مصاب',
        'قصف', 'قصفت', 'غارة', 'غارات',
        'اشتباك', 'اشتباكات', 'معارك', 'معركة',
        'هجوم', 'هجمات', 'هاجم', 'هاجمت',
        'مجزرة', 'مجازر', 'ابادة', 'ابادة جماعية',
        'جريمة حرب', 'جرائم حرب', 'انتهاكات', 'انتهاك',
        'تعذيب', 'عذبوا',
        'نزوح', 'نازح', 'نازحين', 'لاجئ', 'لاجئين',
        'تهجير', 'هجروا', 'مهجر', 'تشريد', 'مشردين',
        'اغتصاب', 'اغتصبوا', 'عنف جنسي',
        'نهب', 'نهبوا', 'سلب', 'سرقة', 'سرقوا',
        'حصار', 'محاصرة', 'حاصروا',
        'اختطاف', 'خطفوا', 'مختطف',
    ],
    "war_parties": [
        'الدعم السريع', 'قوات الدعم السريع', 'دعم سريع', 'الدعامة',
        'الجيش السوداني', 'القوات المسلحة', 'القوات المسلحة السودانية',
        'الجيش', 'جيش السودان',
        'حميدتي', 'حميدتى', 'محمد حمدان دقلو',
        'البرهان', 'عبد الفتاح البرهان',
        'دقلو', 'عبد الرحيم دقلو',
        'المليشيا', 'مليشيا', 'مليشيات',
        'الحركة الشعبية', 'حركة تحرير السودان',
        'عبد الواحد', 'مناوي', 'جبريل ابراهيم',
        'عبد العزيز الحلو', 'الحلو',
        'RSF', 'SAF', 'المخابرات', 'الامن', 'جهاز الامن',
    ],
    "locations_conflict": [
        'الخرطوم', 'خرطوم', 'بحري', 'امدرمان', 'ام درمان',
        'شرق النيل', 'كرري',
        'دارفور', 'الجنينة', 'الفاشر', 'نيالا', 'زالنجي',
        'شمال دارفور', 'جنوب دارفور', 'غرب دارفور', 'وسط دارفور',
        'كردفان', 'شمال كردفان', 'جنوب كردفان', 'الابيض', 'كادقلي',
        'الجزيرة', 'ود مدني', 'مدني',
        'بورتسودان', 'بورت سودان', 'كسلا', 'القضارف',
        'مروي', 'عطبرة', 'شندي', 'الدامر',
        'النيل الازرق', 'الدمازين', 'الروصيرص',
        'النيل الابيض', 'كوستي', 'ربك',
        'سنار', 'سنجة', 'الحدود', 'المثلث الحدودي',
    ],
    "neutral_news": [
        'عاجل', 'خبر عاجل', 'اخبار عاجلة',
        'بيان', 'بيان صحفي', 'بيان رسمي',
        'أعلن', 'اعلن', 'اعلنت', 'اعلنوا',
        'صرح', 'صرحت', 'تصريح', 'تصريحات',
        'مصادر', 'مصادر ميدانية', 'مصادر موثوقة',
        'تقرير', 'تقارير', 'مراسل', 'مراسلنا',
        'وكالة', 'وكالات', 'مؤتمر صحفي',
        'الناطق الرسمي', 'المتحدث باسم',
        'بحسب', 'وفقا', 'نقلا عن', 'افادت', 'افاد',
        'رصد', 'رصد ومتابعة', 'متابعة',
    ],
    "revolution_terms": [
        'ثورة', 'الثورة', 'ثورة ديسمبر',
        'تسقط بس', '#تسقط_بس', 'حرية سلام وعدالة',
        'مدنية', 'حكومة مدنية', 'حكم مدني',
        'سلمية', 'سلمية سلمية',
        'اعتصام', 'اعتصام القيادة',
        'موكب', 'مواكب', 'مليونية',
        'مظاهرة', 'مظاهرات', 'تظاهر', 'احتجاج', 'احتجاجات',
        'لجان المقاومة', 'لجان مقاومة', 'مقاومة', 'صمود',
        'انقلاب', 'الانقلاب', 'انقلاب اكتوبر',
        'عسكر', 'حكم عسكري',
        'فترة انتقالية', 'الحكومة الانتقالية',
        'حمدوك', 'عبدالله حمدوك',
        'البشير', 'عمر البشير', 'نظام البشير',
        'المؤتمر الوطني', 'الحزب الحاكم',
    ],
    "profanity": [
        # Sexual profanity
        'كسم', 'كسمك', 'كس امك', 'كس ام',
        'كسك', 'كس', 'كسوم',
        'زب', 'زوط', 'طيزك', 'مسمار',
        'بيضاتك', 'بيض', 'قلقاتك',
        'نهودك', 'شطور', 'شطورك', 'لبنك',
        'جملك', 'جملها', 'نخريك',
        'احشك', 'حشاه', 'حشاك', 'حشرو', 'حشروليك', 'حشروه',
        'اتناك', 'ناكك', 'زعط', 'زعطك',
        'بل', 'مبلول', 'شمبورة', 'شمبوره',
        'ماسورة', 'دعاره', 'زفرة', 'زفارة',
        # Sexual orientation slurs
        'خول', 'لوطي', 'بتلاوط', 'عجلاتي',
        # Common Arabic profanity in Sudan
        'شرموط', 'شرموطة', 'شراميط',
        'عرص', 'عرصة', 'منيك', 'منيكة',
        'ابن الكلب', 'ود الكلب',
        'ابن الشرموطة', 'ود الشرموطة',
        'ابن الحرام', 'ود الحرام', 'قحبة', 'قحاب',
        # Excrement/filth
        'خرا', 'خرى', 'خره', 'تفو',
        'وسخان', 'وسخ', 'زق', 'زقي',
        # Stupidity (Sudanese)
        'اهبل', 'مستهبل', 'تتهابل',
        'ابلده', 'دلاهة', 'هبنقه', 'هبنقات',
        'مسطول', 'تمبرله',
        # Appearance/character (Sudanese)
        'سجم', 'سجمانه', 'عوير', 'عوين',
        'خرمان', 'عرمان', 'كدكه', 'كدكة',
        'عواره', 'كلجة', 'جعان', 'فلقاي',
        'منطط', 'قرقدي', 'قلوط',
        'شليق', 'قنيطة', 'قنيط',
        'شكش', 'شكشة', 'شكشه', 'ابصلعه',
        # General Sudanese insults
        'خايب', 'باطل', 'سفيه',
        'جعجاع', 'قحاطة',
        'تافه', 'تافهين',
        'سندالة', 'سندالتك',
        'ماكويس', 'ماتمام',
        'صاج', 'صيج', 'صيجليك', 'فلنقايات',
        'ابوك', 'امك',
        # Behavioral (Sudanese)
        'تهنق', 'تكاكي', 'تهر', 'تملص', 'تتشالق',
        'بتهنجك', 'شاميك', 'شميتك', 'ظعاط',
        'عب', 'خطرات', 'جعبات', 'شفشفة',
        'عرد', 'جغم', 'طير', 'طيرو',
        'بالابجدية', 'ببلك',
        # Gender-specific Sudanese
        'لبوه', 'خمسين', 'عزبا', 'عزبه',
        'خالة', 'خالات', 'مطفجه', 'نعجه ساي',
        # General insults
        'حقير', 'حقراء', 'وضيع',
        'جبان', 'جبناء', 'خسيس', 'خساس',
        'نذل', 'انذال', 'سافل', 'سفلة', 'غبي', 'جهلة',
    ],
    "religious_sectarian": [
        'كافر', 'كفار', 'كافرين', 'مرتد', 'مرتدين',
        'منافق', 'منافقين', 'مشرك', 'مشركين',
        'صليبي', 'صليبيين', 'علماني', 'علمانيين',
        'ملحد', 'ملحدين', 'زنديق', 'زنادقة',
    ],
    "gender_hate": [
        'عنف ضد المراة', 'عنف اسري',
        'اغتصاب', 'تحرش', 'تحرش جنسي',
        'عار', 'عيب', 'فضيحة',
        'ستات', 'حريم', 'مرة', 'بت كلب',
    ],
    "threats": [
        'نهدد', 'تهديد', 'هددوا', 'توعد', 'توعدوا',
        'انتقام', 'ننتقم', 'انتقموا', 'ثأر', 'نثأر',
        'عقاب', 'نعاقب', 'عاقبوا',
        'حساب', 'نحاسب', 'حاسبوا', 'يوم الحساب',
        'ويل', 'ويل لهم', 'مصير', 'مصيرهم',
        'ندمر', 'سندمر', 'ستندم', 'سيندمون', 'ندمانين',
        'والله نكسر', 'والله نوريك', 'عليك الله',
    ],
    "humanitarian": [
        'مساعدات', 'اغاثة', 'اغاثية',
        'مجاعة', 'جوع', 'جوعانين', 'مياه', 'عطش',
        'دواء', 'ادوية', 'مستشفى', 'مستشفيات',
        'كهرباء', 'انقطاع',
        'انسانية', 'ازمة انسانية', 'منظمة', 'منظمات',
        'الصليب الاحمر', 'الهلال الاحمر',
        'الامم المتحدة', 'اليونيسف',
        'وقف اطلاق النار', 'هدنة',
        'سلام', 'مفاوضات', 'تفاوض',
    ],
}

# ─────────────────────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────────────────────
NON_SUDANESE_LOCATIONS = [
    'ادلب', 'حلب', 'دمشق', 'الجولاني', 'الاسد', 'بشار',
    'سوريا', 'سورية',
    'القاهرة', 'السيسي',
    'طرابلس', 'بنغازي', 'حفتر', 'ليبيا',
    'صنعاء', 'عدن', 'الحوثي', 'الحوثيين',
    'بغداد', 'الموصل',
]

SUDANESE_DIALECT_MARKERS = [
    'شنو', 'هسع', 'هسة', 'زول', 'زولة',
    'كده', 'ده', 'دي', 'ليك', 'ليكي',
    'بتاع', 'ياخ', 'عشان',
    'عدمان', 'سمجان', 'عيان',
    'دقس', 'كركبة', 'كبكبة',
    'ملاح', 'عيشة', 'جبنة',
    'كاواي', 'مرق', 'جاب', 'فات',
    'ماكويس', 'ماتمام',
    'هسي', 'داك', 'ديل',
    'بالله', 'ودّ', 'بت',
]


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def has_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text))


def classify_line(text):
    matches = []
    for category, keywords in KEYWORD_CATEGORIES.items():
        for kw in keywords:
            if kw in text:
                matches.append(category)
                break
    return matches


def is_non_sudanese(text):
    """Check if text contains non-Sudanese locations (Syrian, Egyptian, etc.)"""
    for loc in NON_SUDANESE_LOCATIONS:
        if loc in text:
            return True
    return False


def has_dialect_marker(text):
    """Check if text contains Sudanese dialect markers"""
    for marker in SUDANESE_DIALECT_MARKERS:
        if marker in text:
            return True
    return False


def is_msa_news(text):
    """Check if text looks like MSA breaking news"""
    text_stripped = text.strip()
    if text_stripped.startswith('عاجل'):
        # Check if it's formal news pattern (not dialectal use of عاجل)
        # Formal news: "عاجل | ..." or "عاجل: ..." or starts with عاجل + formal Arabic
        if any(text_stripped.startswith(p) for p in ['عاجل |', 'عاجل:', 'عاجل /', 'عاجل .']):
            return True
        # Also flag if عاجل is first word and no dialect markers present
        if not has_dialect_marker(text):
            return True
    return False


def is_usable_for_labeling(text, min_words=5, max_words=50):
    if not text or not text.strip():
        return False
    text = text.strip()
    if not has_arabic(text):
        return False
    word_count = len(text.split())
    if word_count < min_words or word_count > max_words:
        return False
    return True


def passes_quality_filter(text):
    """Combined quality filter: usable + not non-Sudanese + not MSA news"""
    if not is_usable_for_labeling(text):
        return False
    if is_non_sudanese(text):
        return False
    if is_msa_news(text):
        return False
    return True


# ═══════════════════════════════════════════════════════════════
# STEP 1: Investigate
# ═══════════════════════════════════════════════════════════════
def step1_investigate():
    total_kw = sum(len(v) for v in KEYWORD_CATEGORIES.values())
    print(f"\n{'='*70}")
    print(f" STEP 1: INVESTIGATING WITH EXPANDED KEYWORDS + FILTERS")
    print(f" Categories: {len(KEYWORD_CATEGORIES)} | Keywords: {total_kw}")
    print(f" Non-Sudanese filter: {len(NON_SUDANESE_LOCATIONS)} locations")
    print(f" Dialect markers: {len(SUDANESE_DIALECT_MARKERS)} terms")
    print(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    all_files = {**OLD_CORPUS_FILES, "war_corpus": WAR_CORPUS}
    all_results = {}

    for name, filepath in all_files.items():
        print(f"\n  Analyzing {name}...", flush=True)
        if not os.path.exists(filepath):
            print(f"  ❌ Not found: {filepath}")
            continue

        total = 0
        usable = 0
        filtered_non_sudanese = 0
        filtered_msa_news = 0
        passed_quality = 0
        category_counts = Counter()
        lines_with_keywords = 0
        dialect_marker_count = 0
        seen = set()
        unique_passed = 0

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total += 1
                text = line.strip()
                if not is_usable_for_labeling(text):
                    continue
                usable += 1
                if is_non_sudanese(text):
                    filtered_non_sudanese += 1
                    continue
                if is_msa_news(text):
                    filtered_msa_news += 1
                    continue
                passed_quality += 1

                h = hash(re.sub(r'\s+', ' ', text))
                if h in seen:
                    continue
                seen.add(h)
                unique_passed += 1

                if has_dialect_marker(text):
                    dialect_marker_count += 1

                categories = classify_line(text)
                if categories:
                    lines_with_keywords += 1
                    for cat in categories:
                        category_counts[cat] += 1

        print(f"    Total lines:          {total:>10,}")
        print(f"    Usable (5-50 words):  {usable:>10,}")
        print(f"    Filtered non-Sudan:   {filtered_non_sudanese:>10,}")
        print(f"    Filtered MSA news:    {filtered_msa_news:>10,}")
        print(f"    Passed quality:       {passed_quality:>10,}")
        print(f"    Unique after dedup:   {unique_passed:>10,}")
        print(f"    With dialect markers: {dialect_marker_count:>10,} ({100*dialect_marker_count/unique_passed:.1f}%)" if unique_passed else "")
        print(f"    With keywords:        {lines_with_keywords:>10,} ({100*lines_with_keywords/unique_passed:.1f}%)" if unique_passed else "")

        print(f"\n    KEYWORD CATEGORIES:")
        print(f"    {'Category':<25} {'Count':>8} {'%':>7}")
        print(f"    {'─'*42}")
        for cat, count in category_counts.most_common():
            pct = 100 * count / unique_passed if unique_passed else 0
            print(f"    {cat:<25} {count:>8,} {pct:>6.1f}%")

        all_results[name] = {
            "total": total, "usable": usable,
            "filtered_non_sudanese": filtered_non_sudanese,
            "filtered_msa_news": filtered_msa_news,
            "unique_passed": unique_passed,
            "with_keywords": lines_with_keywords,
            "with_dialect": dialect_marker_count,
            "categories": dict(category_counts),
        }

    # Summary
    print(f"\n{'='*70}")
    print(f" COMBINED SUMMARY (after quality filters)")
    print(f"{'='*70}")
    print(f"\n  {'Source':<25} {'Usable':>10} {'Non-SD':>8} {'MSA':>8} {'Clean':>10} {'KW%':>6}")
    print(f"  {'─'*70}")
    for name, d in all_results.items():
        kpct = 100*d['with_keywords']/d['unique_passed'] if d['unique_passed'] else 0
        print(f"  {name:<25} {d['usable']:>10,} {d['filtered_non_sudanese']:>8,} "
              f"{d['filtered_msa_news']:>8,} {d['unique_passed']:>10,} {kpct:>5.1f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "corpus_investigation.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  📊 Saved to: {OUTPUT_DIR}/corpus_investigation.json")
    print(f"  ⚠️  Run: python3 build_labeling_corpus.py --step 2")


# ═══════════════════════════════════════════════════════════════
# STEP 2: Extract Sample
# ═══════════════════════════════════════════════════════════════
def step2_extract():
    print(f"\n{'='*70}")
    print(f" STEP 2: EXTRACTING ~40,000 SENTENCE SAMPLE (CLEANED)")
    print(f" Filters: non-Sudanese removal + MSA news removal")
    print(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    SAMPLING_CONFIG = {
        "war_corpus": {
            "source": WAR_CORPUS,
            "targets": {
                "hate_tribal": 3800, "hate_political": 2300,
                "dehumanizing": 1200, "violence_incitement": 1000,
                "violence_reporting": 2500, "profanity": 1000,
                "threats": 700, "religious_sectarian": 600,
                "gender_hate": 300, "neutral_news": 2000,
                "humanitarian": 1300, "random_no_keyword": 2300,
            }
        },
        "old_twitter": {
            "source": OLD_CORPUS_FILES["old_twitter"],
            "targets": {
                "hate_tribal": 1500, "hate_political": 1500,
                "revolution_terms": 2000, "profanity": 1000,
                "violence_reporting": 1000, "dehumanizing": 500,
                "violence_incitement": 500, "threats": 500,
                "religious_sectarian": 500, "humanitarian": 500,
                "random_no_keyword": 9000,
            }
        },
        "old_telegram": {
            "sources": [
                OLD_CORPUS_FILES["old_telegram_mukh"],
                OLD_CORPUS_FILES["old_telegram_nabigh"],
                OLD_CORPUS_FILES["old_telegram_yah"],
            ],
            "targets": {
                "hate_tribal": 300, "hate_political": 300,
                "profanity": 200, "dehumanizing": 200,
                "religious_sectarian": 100, "random_no_keyword": 1400,
            }
        },
    }

    total_target = sum(sum(c["targets"].values()) for c in SAMPLING_CONFIG.values())
    print(f"\n  Total target: {total_target:,} sentences")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_selected = []
    global_seen = set()

    for source_name, config in SAMPLING_CONFIG.items():
        print(f"\n  Processing {source_name}...")
        files = [config["source"]] if "source" in config else config["sources"]
        targets = config["targets"]
        buckets = defaultdict(list)
        no_keyword_lines = []

        for filepath in files:
            if not os.path.exists(filepath):
                print(f"    ❌ Not found: {filepath}")
                continue
            print(f"    Reading {os.path.basename(filepath)}...", end="", flush=True)
            fc = 0
            filtered = 0
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if not passes_quality_filter(text):
                        if is_usable_for_labeling(text) and (is_non_sudanese(text) or is_msa_news(text)):
                            filtered += 1
                        continue
                    normalized = re.sub(r'\s+', ' ', text).strip()
                    h = hash(normalized)
                    if h in global_seen:
                        continue
                    global_seen.add(h)
                    fc += 1
                    categories = classify_line(text)
                    if categories:
                        for cat in categories:
                            if cat in targets:
                                buckets[cat].append(normalized)
                    else:
                        no_keyword_lines.append(normalized)
            print(f" ({fc:,} clean, {filtered:,} filtered)")

        buckets["random_no_keyword"] = no_keyword_lines

        source_selected = []
        print(f"\n    Sampling from {source_name}:")
        print(f"    {'Category':<25} {'Available':>10} {'Target':>8} {'Selected':>10}")
        print(f"    {'─'*55}")
        for category, target_count in targets.items():
            available = buckets.get(category, [])
            n = min(target_count, len(available))
            selected = random.sample(available, n) if n > 0 else []
            for text in selected:
                source_selected.append({
                    "text": text, "source": source_name,
                    "keyword_category": category,
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "has_dialect_marker": has_dialect_marker(text),
                })
            s = "✅" if n >= target_count else "⚠️"
            print(f"    {category:<25} {len(available):>10,} {target_count:>8,} {n:>10,} {s}")
        print(f"    {'─'*55}")
        print(f"    {'SUBTOTAL':<25} {'':>10} {sum(targets.values()):>8,} {len(source_selected):>10,}")
        all_selected.extend(source_selected)

    random.shuffle(all_selected)

    # Stats
    print(f"\n{'='*70}")
    print(f" FINAL CORPUS STATISTICS")
    print(f"{'='*70}")
    print(f"\n  Total target:   {total_target:,}")
    print(f"  Total selected: {len(all_selected):,}")

    source_counts = Counter(s["source"] for s in all_selected)
    print(f"\n  BY SOURCE:")
    for src, cnt in source_counts.most_common():
        print(f"    {src:<25} {cnt:>8,} ({100*cnt/len(all_selected):.1f}%)")

    cat_counts = Counter(s["keyword_category"] for s in all_selected)
    print(f"\n  BY KEYWORD CATEGORY:")
    for cat, cnt in cat_counts.most_common():
        print(f"    {cat:<25} {cnt:>8,} ({100*cnt/len(all_selected):.1f}%)")

    dialect_count = sum(1 for s in all_selected if s["has_dialect_marker"])
    print(f"\n  WITH DIALECT MARKERS: {dialect_count:,} ({100*dialect_count/len(all_selected):.1f}%)")

    wc = [s["word_count"] for s in all_selected]
    print(f"\n  WORD COUNT: min={min(wc)}, max={max(wc)}, mean={sum(wc)/len(wc):.1f}, median={sorted(wc)[len(wc)//2]}")

    # Save
    json_path = os.path.join(OUTPUT_DIR, "labeling_corpus_25k.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_selected, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(OUTPUT_DIR, "labeling_corpus_25k.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in all_selected:
            f.write(item["text"] + "\n")

    tsv_path = os.path.join(OUTPUT_DIR, "labeling_corpus_25k.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("id\ttext\tsource\tkeyword_category\tword_count\thas_dialect\tlabel\n")
        for i, item in enumerate(all_selected):
            t = item["text"].replace("\t", " ").replace("\n", " ")
            d = "1" if item["has_dialect_marker"] else "0"
            f.write(f"{i+1}\t{t}\t{item['source']}\t{item['keyword_category']}\t{item['word_count']}\t{d}\t\n")

    preview_path = os.path.join(OUTPUT_DIR, "sample_preview.txt")
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\nSAMPLE PREVIEW - 20 per category\nTotal: {len(all_selected):,}\n{'='*70}\n\n")
        by_cat = defaultdict(list)
        for item in all_selected:
            by_cat[item["keyword_category"]].append(item)
        for cat in sorted(by_cat):
            items = by_cat[cat]
            f.write(f"\n{'─'*60}\nCATEGORY: {cat} ({len(items)} total)\n{'─'*60}\n\n")
            for item in items[:20]:
                dm = "[SD]" if item["has_dialect_marker"] else "[--]"
                f.write(f"{dm} [{item['source']}] [{item['word_count']}w] {item['text'][:200]}\n\n")

    stats = {
        "total_sentences": len(all_selected), "total_target": total_target,
        "by_source": dict(source_counts), "by_category": dict(cat_counts),
        "dialect_marker_count": dialect_count,
        "word_count_stats": {
            "min": min(wc), "max": max(wc),
            "mean": round(sum(wc)/len(wc), 1),
            "median": sorted(wc)[len(wc)//2],
        },
        "keywords": sum(len(v) for v in KEYWORD_CATEGORIES.values()),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, "corpus_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n  📁 JSON:    {json_path}")
    print(f"  📁 Text:    {txt_path}")
    print(f"  📁 TSV:     {tsv_path}")
    print(f"  📁 Preview: {preview_path}")
    print(f"\n{'='*70}\n DONE. Review files in: {OUTPUT_DIR}/\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, choices=[1, 2])
    args = parser.parse_args()
    if args.step == 1:
        step1_investigate()
    elif args.step == 2:
        step2_extract()

if __name__ == "__main__":
    main()
