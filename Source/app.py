import csv, json, pickle, re
from pathlib import Path
from flask import Flask, render_template, request

base_dir = Path(__file__).resolve().parent.parent
# ... / Model-Fjale-Mire-Keq / ...
app = Flask(__name__, 
    template_folder=base_dir / "Web",
    static_folder=base_dir / "Web",
    static_url_path="/static"
)

ALBANIAN_DATASET = base_dir / "Data" / "train_data_sq.csv"
ENGLISH_DATASET = base_dir / "Data" / "train_data_en.csv"

# LANGUAGE_TEXT
with open(base_dir / "Data" / "language_text.json", "r") as file:
    LANGUAGE_TEXT = json.load(file)

# Ngarkimi i modelit dhe vectorizer
with open(base_dir / "Source" / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(base_dir / "Source" / "sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# POSITIVE WORDS
with open(base_dir / "Data" / "positive_words.csv") as file:
	creader = csv.reader(file)
	POSITIVE_WORDS = [i for i in creader][0]

# NEGATIVE WORDS
with open(base_dir / "Data" / "negative_words.csv") as file:
	creader = csv.reader(file)
	NEGATIVE_WORDS = [i for i in creader][0]

# NEGATIONS
with open(base_dir / "Data" / "negation_words.csv") as file:
	creader = csv.reader(file)
	NEGATIONS = [i for i in creader]

# POSITIVE PHRASES
with open(base_dir / "Data" / "positive_phrases.csv") as file:
	creader = csv.reader(file)
	POSITIVE_PHRASES = [i for i in creader]


# NEGATIVE PHRASES
with open(base_dir / "Data" / "negative_phrases.csv") as file:
	creader = csv.reader(file)
	NEGATIVE_PHRASES = [i for i in creader]


# NEUTRAL WORDS
with open(base_dir / "Data" / "neutral_words.csv") as file:
	creader = csv.reader(file)
	NEUTRAL_WORDS = [i for i in creader][0]


# NEUTRAL PHRASES
with open(base_dir / "Data" / "neutral_phrases.csv") as file:
	creader = csv.reader(file)
	NEUTRAL_PHRASES = [i for i in creader]


def build_stat_items(counts, total, label_order, stat_labels):
    return [
        {
            "label_key": label_key,
            "label": stat_labels[label_key],
            "count": counts.get(source_label, 0),
            "percent": round(counts.get(source_label, 0) * 100 / total, 2) if total else 0,
            "css": css_class,
        }
        for source_label, label_key, css_class in label_order
    ]


def read_label_counts(path, field_name, allowed_labels, encoding):
    counts = {label: 0 for label in allowed_labels}
    total = 0

    with open(path, newline="", encoding=encoding) as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = str(row.get(field_name, "")).strip()
            if label in counts:
                counts[label] += 1
                total += 1

    return counts, total


def load_dataset_stats(language):
    text = LANGUAGE_TEXT[language]
    sq_counts, sq_total = read_label_counts(
        ALBANIAN_DATASET,
        "Sentiment",
        ("1", "0", "2"),
        "utf-8-sig",
    )
    en_counts, en_total = read_label_counts(
        ENGLISH_DATASET,
        "sentiment",
        ("1", "0", "2"),
        "utf-8",
    )

    return {
        "sq": {
            "title": text["stats_title_sq"],
            "total": sq_total,
            "items": build_stat_items(
                sq_counts,
                sq_total,
                (
                    ("1", "positive", "positive"),
                    ("0", "negative", "negative"),
                    ("2", "neutral", "neutral"),
                ),
                text["stat_labels"],
            ),
        },
        "en": {
            "title": text["stats_title_en"],
            "total": en_total,
            "items": build_stat_items(
                en_counts,
                en_total,
                (
                    ("1", "positive", "positive"),
                    ("0", "negative", "negative"),
                    ("2", "neutral", "neutral"),
                ),
                text["stat_labels"],
            ),
        },
    }

def normalize_text(text):
    lowered = text.lower()
    cleaned = re.sub(r"[^0-9a-zA-Zçë\s']", " ", lowered)
    return ''.join(re.sub(r"\s+", " ", cleaned).strip())


def keyword_sentiment(text):
    normalized = normalize_text(text)
    if not normalized:
        return None

    tokens = normalized.split()
    if len(tokens) < 2:
        return None
    print(NEUTRAL_WORDS)
    for phrase in NEUTRAL_WORDS:
        if phrase in normalized:
            return 2

    score = 0

    for phrase in NEGATIVE_WORDS:
        if phrase in normalized:
            score -= 4
    for phrase in POSITIVE_WORDS:
        if phrase in normalized:
            score += 3

    for phrase in POSITIVE_PHRASES:
        if " " in phrase and phrase in normalized:
            score += 2
    for phrase in NEGATIVE_PHRASES:
        if " " in phrase and phrase in normalized:
            score -= 2

    has_neutral = any(token in NEUTRAL_WORDS for token in tokens)
    has_positive = False
    has_negative = False

    for index in range(len(tokens)):
        window_two = " ".join(tokens[index:index + 2])
        window_three = " ".join(tokens[index:index + 3])

        if window_three in POSITIVE_PHRASES or window_two in POSITIVE_PHRASES:
            score += 3
            has_positive = True
        if window_three in NEGATIVE_PHRASES or window_two in NEGATIVE_PHRASES:
            score -= 3
            has_negative = True

    for index, token in enumerate(tokens):
        if token in NEUTRAL_WORDS:
            has_neutral = True

        if token in NEGATIONS:
            window = tokens[index + 1:index + 4]
            if any(candidate in POSITIVE_WORDS for candidate in window):
                score -= 2
                has_negative = True
            if any(candidate in NEGATIVE_WORDS for candidate in window):
                score += 2
                has_positive = True

    if has_neutral or (has_positive and has_negative):
        return 2

    if score >= 2:
        return 1
    if score <= -2:
        return 0
    return None


def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    probabilities = model.predict_proba(text_vectorized)[0]
    ml_prediction = int(model.predict(text_vectorized)[0])
    keyword_prediction = keyword_sentiment(text)
    confidence = float(max(probabilities))

    if keyword_prediction is not None:
        if ml_prediction == 2 and keyword_prediction in (0, 1) and confidence < 0.85:
            return keyword_prediction
        if len(text.split()) <= 4 and confidence < 0.60:
            return keyword_prediction
        if confidence < 0.70 and keyword_prediction != ml_prediction:
            return keyword_prediction

    return ml_prediction


def build_stats_sections(language):
    dataset_stats = load_dataset_stats(language)
    text = LANGUAGE_TEXT[language]
    sections = [
        {
            "key": "sq",
            "title": dataset_stats["sq"]["title"],
            "total": dataset_stats["sq"]["total"],
            "items": dataset_stats["sq"]["items"],
            "note": None,
        },
        {
            "key": "en",
            "title": dataset_stats["en"]["title"],
            "total": dataset_stats["en"]["total"],
            "items": dataset_stats["en"]["items"],
            "note": text["stats_note_en"],
        },
    ]
    return sections

@app.route("/", methods=["GET", "POST"])
def home():
    language = request.values.get("language") or request.args.get("lang") or "sq"
    if language not in LANGUAGE_TEXT:
        language = "sq"

    text = LANGUAGE_TEXT[language]
    result = None
    result_key = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("comment", "")

        if user_text.strip() != "":
            prediction = predict_sentiment(user_text)
            result_key = prediction if prediction in text["labels"] else 2
            result = text["labels"].get(prediction, text["labels"]["2"])

    return render_template(
        "index.html",
        language=language,
        texts=text,
        translations=LANGUAGE_TEXT,
        result=result,
        result_key=result_key,
        user_text=user_text,
        stats_sections=build_stats_sections(language),
    )

if __name__ == "__main__":
    app.run(debug=True)
