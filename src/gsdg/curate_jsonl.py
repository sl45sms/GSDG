import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from gsdg.openai_client import InferenceError, OpenAICompatibleClient

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None


LOGGER = logging.getLogger("gsdg.curate_jsonl")

DEFAULT_REVIEW_SYSTEM_PROMPT = (
    "Είσαι ένας εξειδικευμένος αναλυτής δεδομένων και γλωσσολόγος, υπεύθυνος για "
    "τον ποιοτικό έλεγχο και τον καθαρισμό ενός ελληνικού συνόλου δεδομένων "
    "(dataset) που θα χρησιμοποιηθεί για το fine-tuning ενός Large Language Model. "
    "Ο στόχος σου είναι να αξιολογείς ένα ζεύγος 'Ερώτησης' και 'Απάντησης' με βάση "
    "αυστηρά κριτήρια και να αποφασίζεις αν πρέπει να διατηρηθεί (ACCEPT) ή να "
    "απορριφθεί (REJECT), ταξινομώντας το ταυτόχρονα στη σωστή θεματική κατηγορία."
)

CATEGORY_NAMES = (
    "politics",
    "science",
    "medicine",
    "technology",
    "art",
    "history",
    "religion",
    "education",
    "philosophy",
    "sports",
    "business",
    "economics",
    "law",
    "mythology",
    "literature",
    "music",
    "general",
)

LLM_REJECT_LABELS = (
    "REJECT_LANGUAGE",
    "REJECT_TEMPORAL",
    "REJECT_LOW_QUALITY",
)

CATEGORY_ALIASES = {
    "philosophy": "philosophy",
    "philosophical": "philosophy",
    "φιλοσοφία": "philosophy",
    "φιλοσοφια": "philosophy",
    "religion": "religion",
    "religious": "religion",
    "theology": "religion",
    "θεολογία": "religion",
    "θεολογια": "religion",
    "θρησκεία": "religion",
    "θρησκεια": "religion",
    "education": "education",
    "educational": "education",
    "pedagogy": "education",
    "pedagogical": "education",
    "εκπαίδευση": "education",
    "εκπαιδευση": "education",
    "παιδεία": "education",
    "παιδεια": "education",
    "sports": "sports",
    "sport": "sports",
    "athletics": "sports",
    "αθλητισμός": "sports",
    "αθλητισμος": "sports",
    "σπορ": "sports",
    "business": "business",
    "commerce": "business",
    "management": "business",
    "marketing": "business",
    "επιχειρήσεις": "business",
    "επιχειρησεις": "business",
    "επιχείρηση": "business",
    "επιχειρηση": "business",
    "economics": "economics",
    "economic": "economics",
    "finance": "economics",
    "financial": "economics",
    "οικονομικά": "economics",
    "οικονομικα": "economics",
    "οικονομία": "economics",
    "οικονομια": "economics",
    "law": "law",
    "legal": "law",
    "jurisprudence": "law",
    "δίκαιο": "law",
    "δικαιο": "law",
    "νομικά": "law",
    "νομικα": "law",
    "mythology": "mythology",
    "mythological": "mythology",
    "μυθολογία": "mythology",
    "μυθολογια": "mythology",
    "literature": "literature",
    "literary": "literature",
    "λογοτεχνία": "literature",
    "λογοτεχνια": "literature",
    "music": "music",
    "musical": "music",
    "μουσική": "music",
    "μουσικη": "music",
    "general_knowledge": "general",
    "γενικά": "general",
    "γενικα": "general",
    "πολιτική": "politics",
    "πολιτικη": "politics",
    "επιστήμη": "science",
    "επιστημη": "science",
    "ιατρική": "medicine",
    "ιατρικη": "medicine",
    "τεχνολογία": "technology",
    "τεχνολογια": "technology",
    "τέχνη": "art",
    "τεχνη": "art",
    "ιστορία": "history",
    "ιστορια": "history",
    "religion_and_belief": "religion",
    "education_and_pedagogy": "education",
    "sports_and_games": "sports",
    "business_and_management": "business",
    "economics_and_finance": "economics",
    "law_and_justice": "law",
    "myths": "mythology",
    "literature_and_books": "literature",
    "music_and_sound": "music",
}

REJECT_LABEL_ALIASES = {
    "REJECT_LANGUAGE": "REJECT_LANGUAGE",
    "LANGUAGE": "REJECT_LANGUAGE",
    "REJECT_TEMPORAL": "REJECT_TEMPORAL",
    "TEMPORAL": "REJECT_TEMPORAL",
    "REJECT_LOW_QUALITY": "REJECT_LOW_QUALITY",
    "LOW_QUALITY": "REJECT_LOW_QUALITY",
    "LOWQUALITY": "REJECT_LOW_QUALITY",
}

LOW_INFORMATION_ANSWERS = {
    "ναι",
    "οχι",
    "όχι",
    "δεν ξερω",
    "δεν ξέρω",
    "ισως",
    "ίσως",
    "μπορει",
    "μπορεί",
    "κανενα",
    "κανενα απο τα παραπανω",
    "κανένα",
}

AI_REFUSAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ως\s+(?:ai|γλωσσικ(?:ο|ό)\s+μοντ(?:ε|έ)λο)",
        r"δεν\s+έχω\s+τη\s+δυνατότητα",
        r"δεν\s+μπορ(?:ώ|ω)\s+να\s+(?:εκφ(?:έ|ε)ρω|δώσω|απαντήσω)",
        r"i\s+am\s+an\s+ai",
        r"as\s+an\s+ai\s+language\s+model",
        r"i\s+cannot\s+(?:provide|answer|give)",
    )
]

BROKEN_MARKUP_PATTERNS = [
    re.compile(r"<[^>\n]*$"),
    re.compile(r"^[^<\n]*>"),
    re.compile(r"\[[^\]\n]*\]\([^\)\n]*$"),
    re.compile(r"\[[^\]\n]*$"),
    re.compile(r"\*\*[^*\n]*$"),
    re.compile(r"`[^`\n]*$"),
]

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]"
)
REPEATED_CHARACTER_PATTERN = re.compile(r"([^\W\d_])\1{7,}|([^\s])\2{11,}", re.UNICODE)
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_PATTERN = re.compile(r"\w+(?:['’]\w+)?", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?;;\n]+")

CATEGORY_KEYWORDS = {
    "politics": (
        "βουλ",
        "κυβερν",
        "εκλογ",
        "υπουργ",
        "πρωθυπουργ",
        "πρόεδρ",
        "προεδρ",
        "διπλωματ",
        "διεθν",
        "συνταγμ",
        "κράτος",
        "πολιτικ",
    ),
    "science": (
        "φυσικ",
        "χημ",
        "μαθημα",
        "διάστημ",
        "διαστημ",
        "πλανήτ",
        "πλανητ",
        "αστρον",
        "θεωρ",
        "εξίσ",
        "εξισ",
        "πειραμ",
    ),
    "medicine": (
        "ιατρ",
        "υγε",
        "βιολογ",
        "κύτταρ",
        "κυτταρ",
        "νόσ",
        "νοσ",
        "ασθεν",
        "θεραπε",
        "φάρμακ",
        "φαρμακ",
        "σύμπτωμ",
        "συμπτωμ",
    ),
    "technology": (
        "τεχνολογ",
        "πληροφορ",
        "υπολογιστ",
        "λογισμ",
        "αλγόριθ",
        "αλγοριθ",
        "δίκτυ",
        "δικτυ",
        "μηχανικ",
        "ηλεκτρον",
        "προγραμματ",
        "ρομπότ",
        "ρομποτ",
    ),
    "art": (
        "τέχν",
        "τεχν",
        "πολιτισμ",
        "κινηματογρ",
        "ταιν",
        "θέατρ",
        "θεατρ",
        "ζωγραφ",
        "γλυπτ",
        "σκηνοθ",
        "εικαστ",
        "γκαλερ",
        "πολιτιστικ",
    ),
    "history": (
        "ιστορ",
        "αρχαιολογ",
        "αρχαί",
        "αρχαι",
        "βυζαντ",
        "αιώ",
        "αιω",
        "επανάστα",
        "επαναστα",
        "αυτοκρατ",
        "βασιλ",
        "χρονολογ",
    ),
    "religion": (
        "θρησκ",
        "θεολογ",
        "εκκλησ",
        "μοναχ",
        "μοναστ",
        "πίστ",
        "πιστ",
        "χριστιαν",
        "ισλάμ",
        "ισλαμ",
        "κοράν",
        "κοραν",
        "βίβλ",
        "βιβλ",
        "ευαγγέλ",
        "ευαγγελ",
        "δογμ",
        "λατρ",
    ),
    "education": (
        "εκπαιδ",
        "παιδαγωγ",
        "σχολ",
        "μαθητ",
        "διδασκ",
        "διδασκαλ",
        "πανεπιστημ",
        "καθηγητ",
        "μάθημ",
        "μαθημ",
        "μάθηση",
        "μαθηση",
        "πρόγραμμα σπουδ",
        "προγραμμα σπουδ",
        "τάξ",
        "ταξ",
    ),
    "philosophy": (
        "φιλοσοφ",
        "οντολογ",
        "γνωσιολογ",
        "μεταφυσ",
        "ηθικ",
        "λογικ",
        "αριστοτέλ",
        "αριστοτελ",
        "πλάτων",
        "πλατων",
        "σωκράτ",
        "σωκρατ",
        "υπαρξ",
        "νουσ",
        "ψυχ",
        "συνείδησ",
        "συνειδησ",
    ),
    "sports": (
        "αθλητ",
        "άθλημ",
        "αθλημα",
        "ποδόσφ",
        "ποδοσφ",
        "μπάσκετ",
        "μπασκετ",
        "βόλεϊ",
        "βολει",
        "τένις",
        "τενις",
        "ολυμπιακ",
        "αγών",
        "αγων",
        "πρωτάθλημ",
        "πρωταθλημ",
        "προπονητ",
        "γκολ",
        "μαραθών",
        "μαραθων",
    ),
    "business": (
        "επιχειρ",
        "εταιρεί",
        "εταιρει",
        "μάρκετινγκ",
        "μαρκετινγκ",
        "διοίκησ",
        "διοικησ",
        "management",
        "startup",
        "brand",
        "πωλήσ",
        "πωλησ",
        "πελάτ",
        "πελατ",
        "αγορά εργασ",
        "αγορα εργασ",
        "επιχειρηματ",
    ),
    "economics": (
        "οικονομ",
        "πληθωρισ",
        "αεπ",
        "gdp",
        "φορολογ",
        "επιτόκ",
        "επιτοκ",
        "τράπεζ",
        "τραπεζ",
        "χρηματοοικονομ",
        "νομισμ",
        "αγορά",
        "αγορα",
        "επενδ",
        "μισθ",
        "ανεργ",
        "δημοσιονομ",
    ),
    "law": (
        "δίκαι",
        "δικαι",
        "νόμ",
        "νομ",
        "σύνταγμ",
        "συνταγμ",
        "δικαστ",
        "δικηγορ",
        "εισαγγελ",
        "ποινικ",
        "αστικ",
        "συμβόλ",
        "συμβολ",
        "σύμβασ",
        "συμβασ",
        "νομοθεσ",
        "άρθρ",
        "αρθρ",
    ),
    "mythology": (
        "μυθολογ",
        "μύθ",
        "μυθ",
        "θεά",
        "θεα",
        "θεοί",
        "θεοι",
        "ήρω",
        "ηρω",
        "όλυμπ",
        "ολυμπ",
        "ζευς",
        "αθηνά",
        "αθηνα",
        "ηρακλ",
        "οδυσσ",
        "τιτάν",
        "τιταν",
    ),
    "literature": (
        "λογοτεχν",
        "μυθιστόρ",
        "μυθιστορ",
        "μυθιστόρημ",
        "μυθιστορημ",
        "διήγημ",
        "διηγημ",
        "ποίη",
        "ποιη",
        "ποιητ",
        "συγγραφ",
        "πεζογραφ",
        "στίχ",
        "στιχ",
        "βιβλί",
        "βιβλι",
        "μυθοπλασ",
        "δραματουργ",
    ),
    "music": (
        "μουσικ",
        "τραγούδ",
        "τραγουδ",
        "συνθέτ",
        "συνθετ",
        "μελωδ",
        "ρυθμ",
        "όπερ",
        "οπερ",
        "ορχήστρ",
        "ορχηστρ",
        "συμφων",
        "άλμπουμ",
        "αλμπουμ",
        "κιθάρ",
        "κιθαρ",
        "πιάν",
        "πιαν",
    ),
}


class ParsedSample(NamedTuple):
    record: Dict[str, Any]
    question: str
    answer: str
    source_row_index: Optional[int]
    line_number: int


class ReviewDecision(NamedTuple):
    accept: bool
    category: str
    reject_labels: List[str]
    rationale: str
    used_llm: bool


def default_output_dir() -> Path:
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / "synthetics"
    return Path.cwd() / "synthetics"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally filter, review, classify, and de-duplicate generated Greek "
            "Q/A JSONL records into category-specific JSONL outputs."
        ),
    )
    parser.add_argument("input", help="Input JSONL produced by the generator")
    parser.add_argument(
        "--out-dir",
        default=str(default_output_dir()),
        help="Directory where category JSONL files will be written",
    )
    parser.add_argument(
        "--state-db",
        default=None,
        help="SQLite state path for checkpoints and the MinHash-LSH de-duplication index",
    )
    parser.add_argument(
        "--reject-log",
        default=None,
        help="Optional JSONL file where rejected samples are logged with reason codes",
    )
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token for tokenizer downloads when --tokenizer-model is used",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Optional Hugging Face tokenizer model for accurate context-window token counting",
    )
    parser.add_argument(
        "--disable-llm-review",
        action="store_true",
        help="Skip semantic LLM review and classify only with deterministic heuristics",
    )
    parser.add_argument(
        "--skip-healthcheck",
        action="store_true",
        help="Skip the inference server health check",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--review-max-tokens", type=int, default=256)
    parser.add_argument("--review-temperature", type=float, default=0.0)
    parser.add_argument("--min-answer-words", type=int, default=4)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--max-word-ratio", type=float, default=10.0)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--num-perm", type=int, default=64)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_text(text: Any) -> str:
    return normalize_whitespace(unicodedata.normalize("NFKC", str(text)))


def normalize_for_compare(text: str) -> str:
    return normalize_text(text).casefold()


def extract_question_answer_pair(record: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(record, dict):
        return None

    question = record.get("question")
    answer = record.get("answer")
    if question is not None and answer is not None:
        return normalize_text(question), normalize_text(answer)

    messages = record.get("messages")
    if not isinstance(messages, list):
        return None

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue

        content = message.get("content")
        if isinstance(content, dict):
            payload = content
        elif isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
        else:
            continue

        if not isinstance(payload, dict):
            continue

        question = payload.get("question")
        answer = payload.get("answer")
        if question is not None and answer is not None:
            return normalize_text(question), normalize_text(answer)

    return None


def extract_source_row_index(record: Dict[str, Any]) -> Optional[int]:
    meta = record.get("meta")
    if not isinstance(meta, dict):
        return None

    raw_index = meta.get("source_row_index")
    if isinstance(raw_index, bool):
        return None
    if isinstance(raw_index, int):
        return raw_index
    if isinstance(raw_index, str):
        raw_index = raw_index.strip()
        if raw_index.isdigit():
            return int(raw_index)
    return None


def iter_jsonl_records(input_path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with input_path.open("r", encoding="utf-8") as handle:
        line_number = 0
        while True:
            raw_line = handle.readline()
            if not raw_line:
                break

            line_number += 1
            if not raw_line.endswith("\n"):
                LOGGER.info(
                    "Stopping at unfinished trailing line %s in %s; will retry on the next run.",
                    line_number,
                    input_path,
                )
                break

            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {input_path} at line {line_number}")

            yield line_number, record


def is_greek_character(character: str) -> bool:
    return "GREEK" in unicodedata.name(character, "")


def greek_ratio(text: str) -> float:
    alphabetic_characters = [character for character in text if character.isalpha()]
    if not alphabetic_characters:
        return 0.0
    greek_characters = sum(1 for character in alphabetic_characters if is_greek_character(character))
    return greek_characters / len(alphabetic_characters)


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def tokenize_for_context_window(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def split_sentences(text: str) -> List[str]:
    parts = [normalize_for_compare(part) for part in SENTENCE_SPLIT_PATTERN.split(text)]
    return [part for part in parts if len(part.split()) >= 3]


def build_word_ngrams(text: str, size: int = 5) -> Set[str]:
    words = [token.casefold() for token in WORD_PATTERN.findall(normalize_text(text))]
    if len(words) < size:
        if not words:
            return set()
        return {" ".join(words)}
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def prompt_answer_overlap(question: str, answer: str) -> float:
    question_sentences = set(split_sentences(question))
    answer_sentences = split_sentences(answer)

    sentence_overlap = 0.0
    if answer_sentences:
        shared_sentences = sum(1 for sentence in answer_sentences if sentence in question_sentences)
        sentence_overlap = shared_sentences / len(answer_sentences)

    question_ngrams = build_word_ngrams(question)
    answer_ngrams = build_word_ngrams(answer)
    ngram_overlap = 0.0
    if answer_ngrams:
        ngram_overlap = len(question_ngrams & answer_ngrams) / len(answer_ngrams)

    return max(sentence_overlap, ngram_overlap)


def contains_ai_refusal(answer: str) -> bool:
    return any(pattern.search(answer) for pattern in AI_REFUSAL_PATTERNS)


def contains_broken_markup(text: str) -> bool:
    if text.count("<") != text.count(">"):
        return True
    if text.count("[") != text.count("]"):
        return True
    if text.count("(") != text.count(")") and "](" in text:
        return True
    if text.count("**") % 2 != 0 or text.count("`") % 2 != 0:
        return True
    return any(pattern.search(text) for pattern in BROKEN_MARKUP_PATTERNS)


def has_excessive_emojis(text: str) -> bool:
    emoji_count = len(EMOJI_PATTERN.findall(text))
    if emoji_count < 4:
        return False
    total_tokens = max(1, len(tokenize_for_context_window(text)))
    return emoji_count / total_tokens >= 0.2


def looks_like_garbage(text: str) -> bool:
    if REPEATED_CHARACTER_PATTERN.search(text):
        return True
    if has_excessive_emojis(text):
        return True
    if contains_broken_markup(text):
        return True
    return False


def normalize_low_information_text(text: str) -> str:
    normalized = normalize_for_compare(text)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalize_whitespace(normalized)


def deterministic_reject_labels(
    question: str,
    answer: str,
    token_counter: "TokenCounter",
    min_answer_words: int,
    max_total_tokens: int,
    max_word_ratio: float,
) -> List[str]:
    reject_labels: List[str] = []

    question_word_count = count_words(question)
    answer_word_count = count_words(answer)
    total_tokens = token_counter.count(question) + token_counter.count(answer)
    combined_text = f"{question}\n{answer}"

    if answer_word_count < min_answer_words:
        reject_labels.append("REJECT_SHORT_ANSWER")
    elif normalize_low_information_text(answer) in LOW_INFORMATION_ANSWERS:
        reject_labels.append("REJECT_SHORT_ANSWER")

    if total_tokens > max_total_tokens:
        reject_labels.append("REJECT_CONTEXT_WINDOW")

    if question_word_count > 0 and answer_word_count > 0:
        ratio = max(question_word_count, answer_word_count) / max(1, min(question_word_count, answer_word_count))
        if ratio > max_word_ratio:
            reject_labels.append("REJECT_LENGTH_RATIO")

    if looks_like_garbage(combined_text):
        reject_labels.append("REJECT_GARBAGE")

    if prompt_answer_overlap(question, answer) > 0.8:
        reject_labels.append("REJECT_OVERLAP")

    if contains_ai_refusal(answer):
        reject_labels.append("REJECT_AI_REFUSAL")

    if greek_ratio(question) < 0.25 or greek_ratio(answer) < 0.25:
        reject_labels.append("REJECT_LANGUAGE")

    return list(dict.fromkeys(reject_labels))


def heuristic_category(question: str, answer: str) -> str:
    normalized = normalize_for_compare(f"{question}\n{answer}")
    category_scores: Dict[str, int] = {category: 0 for category in CATEGORY_NAMES}

    for category, keywords in CATEGORY_KEYWORDS.items():
        category_scores[category] = sum(1 for keyword in keywords if keyword in normalized)

    ranked = sorted(
        ((score, category) for category, score in category_scores.items() if category != "general"),
        reverse=True,
    )
    if not ranked or ranked[0][0] == 0:
        return "general"
    if len(ranked) == 1:
        return ranked[0][1]

    top_score, top_category = ranked[0]
    second_score = ranked[1][0]
    if top_score >= 2 and top_score > second_score:
        return top_category
    if top_score >= 1 and second_score == 0:
        return top_category
    return "general"


class TokenCounter:
    def __init__(self, tokenizer_model: Optional[str], hf_token: Optional[str]) -> None:
        self.tokenizer = None
        if tokenizer_model:
            if AutoTokenizer is None:
                raise RuntimeError(
                    "--tokenizer-model requires transformers to be installed in this environment"
                )
            tokenizer_kwargs: Dict[str, Any] = {}
            if hf_token:
                tokenizer_kwargs["token"] = hf_token
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, **tokenizer_kwargs)

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(tokenize_for_context_window(text))


def extract_json_object(raw_content: str) -> Dict[str, Any]:
    cleaned = THINK_TAG_PATTERN.sub("", raw_content).strip()
    match = JSON_OBJECT_PATTERN.search(cleaned)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = extract_partial_review_payload(cleaned)
    else:
        parsed = extract_partial_review_payload(cleaned)

    if parsed is None:
        raise InferenceError(f"model did not return a JSON object: {cleaned}")

    if not isinstance(parsed, dict):
        raise InferenceError(f"model JSON must be an object: {parsed}")
    return parsed


def extract_partial_review_payload(cleaned: str) -> Optional[Dict[str, Any]]:
    decision_match = re.search(r'"decision"\s*:\s*"([^"\\]+)"', cleaned, re.IGNORECASE)
    category_match = re.search(r'"category"\s*:\s*"([^"\\]+)"', cleaned, re.IGNORECASE)
    labels_match = re.search(r'"reject_labels"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL | re.IGNORECASE)
    rationale_match = re.search(r'"rationale"\s*:\s*"((?:\\.|[^"\\])*)', cleaned, re.DOTALL | re.IGNORECASE)

    if not decision_match or not category_match or not labels_match:
        return None

    labels = []
    for raw_label in re.findall(r'"((?:\\.|[^"\\])*)"', labels_match.group(1)):
        try:
            labels.append(json.loads('"' + raw_label + '"'))
        except json.JSONDecodeError:
            labels.append(raw_label)

    rationale = ""
    if rationale_match:
        raw_rationale = rationale_match.group(1)
        try:
            rationale = json.loads('"' + raw_rationale + '"')
        except json.JSONDecodeError:
            rationale = raw_rationale.replace('\\n', ' ').replace('\\"', '"').strip()

    return {
        "decision": decision_match.group(1),
        "category": category_match.group(1),
        "reject_labels": labels,
        "rationale": rationale,
    }


def build_review_prompt(question: str, answer: str, strict: bool = False) -> str:
    rationale_instruction = (
        "Το rationale να έχει έως 12 λέξεις."
        if strict
        else "Το rationale να είναι σύντομη αιτιολόγηση στα Ελληνικά με έως 20 λέξεις."
    )
    return (
        "Αξιολόγησε το παρακάτω ζεύγος Ερώτησης/Απάντησης για ελληνικό dataset fine-tuning.\n"
        "Κριτήρια απόρριψης:\n"
        "1. REJECT_LANGUAGE: μη φυσικά, λανθασμένα ή κακής ποιότητας Ελληνικά, ή έντονα machine-translation artifacts.\n"
        "2. REJECT_TEMPORAL: χρονικά εξαρτημένοι ισχυρισμοί που παρουσιάζονται ως σύγχρονα γεγονότα χωρίς χρονικό προσδιορισμό.\n"
        "3. REJECT_LOW_QUALITY: η απάντηση είναι επιφανειακή, άσχετη, ή δεν απαντά ουσιαστικά στην ερώτηση.\n"
        "Κατηγορίες: politics, science, medicine, technology, art, history, religion, education, philosophy, sports, business, economics, law, mythology, literature, music, general.\n"
        "Η τιμή του category ΠΡΕΠΕΙ να είναι ακριβώς μία από τις παραπάνω. Αν κάτι δεν ταιριάζει ακριβώς, χρησιμοποίησε general.\n"
        "Αν το δείγμα είναι αποδεκτό, επίλεξε decision=ACCEPT και reject_labels=[].\n"
        "Απάντησε ΜΟΝΟ με έγκυρο JSON σε μία γραμμή, χωρίς markdown ή άλλο κείμενο.\n"
        f"{rationale_instruction}\n"
        "JSON σχήμα:\n"
        '{"decision":"ACCEPT|REJECT","category":"politics|science|medicine|technology|art|history|religion|education|philosophy|sports|business|economics|law|mythology|literature|music|general","reject_labels":["REJECT_LANGUAGE|REJECT_TEMPORAL|REJECT_LOW_QUALITY"],"rationale":"σύντομη αιτιολόγηση στα Ελληνικά"}\n\n'
        f"Ερώτηση:\n{question}\n\n"
        f"Απάντηση:\n{answer}"
    )


def normalize_review_category(category: str, question: str, answer: str) -> str:
    normalized_category = normalize_text(category).casefold().replace(" ", "_")
    if normalized_category in CATEGORY_NAMES:
        return normalized_category
    if normalized_category in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized_category]
    fallback_category = heuristic_category(question, answer)
    LOGGER.warning(
        "LLM returned unsupported category %r; falling back to %s",
        category,
        fallback_category,
    )
    return fallback_category


def normalize_review_reject_labels(raw_labels: Any) -> List[str]:
    if not isinstance(raw_labels, list):
        raise InferenceError(f"invalid reject_labels payload: {raw_labels}")

    cleaned_labels = []
    for raw_label in raw_labels:
        label = normalize_text(raw_label).upper().replace("-", "_").replace(" ", "_")
        resolved_label = REJECT_LABEL_ALIASES.get(label)
        if resolved_label is None:
            LOGGER.warning("Ignoring unsupported review reject label %r", raw_label)
            continue
        cleaned_labels.append(resolved_label)
    return list(dict.fromkeys(cleaned_labels))


def parse_review_decision(parsed: Dict[str, Any], question: str, answer: str) -> ReviewDecision:
    decision = normalize_text(parsed.get("decision", "")).upper()
    category = normalize_review_category(parsed.get("category", ""), question, answer)
    rationale = normalize_text(parsed.get("rationale", ""))

    if decision not in {"ACCEPT", "REJECT"}:
        raise InferenceError(f"invalid review decision: {parsed}")

    cleaned_labels = normalize_review_reject_labels(parsed.get("reject_labels", []))
    if decision == "REJECT" and not cleaned_labels:
        cleaned_labels = ["REJECT_LOW_QUALITY"]

    return ReviewDecision(
        accept=decision == "ACCEPT",
        category=category,
        reject_labels=cleaned_labels,
        rationale=rationale,
        used_llm=True,
    )


def run_llm_review(
    client: OpenAICompatibleClient,
    question: str,
    answer: str,
    review_max_tokens: int,
    review_temperature: float,
) -> ReviewDecision:
    prompts = [
        build_review_prompt(question, answer, strict=False),
        build_review_prompt(question, answer, strict=True),
    ]
    last_error = None

    for user_prompt in prompts:
        raw_response = client.create_chat_completion(
            system_prompt=DEFAULT_REVIEW_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=review_temperature,
            max_tokens=review_max_tokens,
            enable_thinking=False,
        )

        try:
            parsed = extract_json_object(raw_response)
            review = parse_review_decision(parsed, question, answer)
            return ReviewDecision(
                accept=review.accept,
                category=review.category,
                reject_labels=review.reject_labels,
                rationale=review.rationale,
                used_llm=True,
            )
        except InferenceError as exc:
            last_error = exc

    if last_error is None:
        raise InferenceError("LLM review failed without an explicit parser error")
    raise last_error


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_dedupe_text(question: str, answer: str) -> str:
    return normalize_for_compare(f"question: {question}\nanswer: {answer}")


def build_minhash_shingles(text: str, ngram_size: int = 3) -> Set[str]:
    words = [token.casefold() for token in WORD_PATTERN.findall(text)]
    if len(words) < ngram_size:
        if not words:
            return {""}
        return {" ".join(words)}
    return {" ".join(words[index : index + ngram_size]) for index in range(len(words) - ngram_size + 1)}


def compute_minhash_signature(shingles: Set[str], num_perm: int) -> List[int]:
    if not shingles:
        shingles = {""}

    signature = [(1 << 64) - 1] * num_perm
    for shingle in shingles:
        shingle_bytes = shingle.encode("utf-8")
        for permutation in range(num_perm):
            permutation_bytes = permutation.to_bytes(2, byteorder="big", signed=False)
            digest = hashlib.blake2b(shingle_bytes + permutation_bytes, digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="big", signed=False)
            if value < signature[permutation]:
                signature[permutation] = value
    return signature


def signature_band_keys(signature: List[int], bands: int) -> List[Tuple[int, str]]:
    rows_per_band = len(signature) // bands
    keys = []
    for band in range(bands):
        start = band * rows_per_band
        chunk = signature[start : start + rows_per_band]
        encoded = ",".join(str(value) for value in chunk).encode("utf-8")
        keys.append((band, hashlib.blake2b(encoded, digest_size=12).hexdigest()))
    return keys


def jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


class DeduplicationStore:
    def __init__(
        self,
        db_path: Path,
        num_perm: int,
        bands: int,
        threshold: float,
    ) -> None:
        self.db_path = db_path
        self.num_perm = num_perm
        self.bands = bands
        self.threshold = threshold
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize_schema()
        self._validate_or_store_config()

    def close(self) -> None:
        self.connection.close()

    def _initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS accepted_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT NOT NULL UNIQUE,
                normalized_text TEXT NOT NULL,
                category TEXT NOT NULL,
                signature_json TEXT NOT NULL,
                source_row_index INTEGER,
                input_path TEXT,
                output_path TEXT
            );

            CREATE TABLE IF NOT EXISTS lsh_buckets (
                band INTEGER NOT NULL,
                bucket_key TEXT NOT NULL,
                sample_id INTEGER NOT NULL,
                PRIMARY KEY (band, bucket_key, sample_id),
                FOREIGN KEY(sample_id) REFERENCES accepted_samples(id)
            );

            CREATE INDEX IF NOT EXISTS idx_lsh_band_bucket ON lsh_buckets (band, bucket_key);
            """
        )
        self.connection.commit()

    def _validate_or_store_config(self) -> None:
        raw_config = self._get_state_value("dedupe_config")
        expected = {
            "num_perm": self.num_perm,
            "bands": self.bands,
            "threshold": self.threshold,
        }
        if raw_config is None:
            self._set_state_value("dedupe_config", json.dumps(expected, sort_keys=True))
            self.connection.commit()
            return

        stored_config = json.loads(raw_config)
        if stored_config != expected:
            raise ValueError(
                "State DB was created with different MinHash/LSH settings. "
                f"Expected {expected}, found {stored_config}."
            )

    def _get_state_value(self, key: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT value FROM run_state WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def _set_state_value(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO run_state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def count_samples(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS total FROM accepted_samples").fetchone()
        return int(row["total"])

    def checkpoint_key(self, input_path: Path) -> str:
        return f"checkpoint::{input_path.resolve()}"

    def load_checkpoint(self, input_path: Path) -> Tuple[int, int]:
        raw_value = self._get_state_value(self.checkpoint_key(input_path))
        if raw_value is None:
            return -1, 0
        payload = json.loads(raw_value)
        return int(payload.get("last_source_row_index", -1)), int(payload.get("last_line_number", 0))

    def update_checkpoint(
        self,
        input_path: Path,
        source_row_index: Optional[int],
        line_number: int,
    ) -> None:
        current_source_row_index, current_line_number = self.load_checkpoint(input_path)
        payload = {
            "last_source_row_index": max(
                current_source_row_index,
                current_source_row_index if source_row_index is None else source_row_index,
            ),
            "last_line_number": max(current_line_number, line_number),
        }
        self._set_state_value(self.checkpoint_key(input_path), json.dumps(payload, sort_keys=True))

    def has_exact_duplicate(self, text_hash: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM accepted_samples WHERE text_hash = ?",
            (text_hash,),
        ).fetchone()
        return row is not None

    def find_near_duplicate(self, normalized_text: str) -> Optional[sqlite3.Row]:
        shingles = build_minhash_shingles(normalized_text)
        signature = compute_minhash_signature(shingles, self.num_perm)
        bucket_keys = signature_band_keys(signature, self.bands)

        candidate_ids: Set[int] = set()
        for band, bucket_key in bucket_keys:
            rows = self.connection.execute(
                "SELECT sample_id FROM lsh_buckets WHERE band = ? AND bucket_key = ?",
                (band, bucket_key),
            ).fetchall()
            candidate_ids.update(int(row["sample_id"]) for row in rows)

        for candidate_id in candidate_ids:
            row = self.connection.execute(
                "SELECT * FROM accepted_samples WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                continue
            candidate_shingles = build_minhash_shingles(str(row["normalized_text"]))
            if jaccard_similarity(shingles, candidate_shingles) >= self.threshold:
                return row
        return None

    def add_sample(
        self,
        normalized_text: str,
        text_hash: str,
        category: str,
        source_row_index: Optional[int],
        input_path: Path,
        output_path: Path,
    ) -> None:
        shingles = build_minhash_shingles(normalized_text)
        signature = compute_minhash_signature(shingles, self.num_perm)
        cursor = self.connection.execute(
            "INSERT INTO accepted_samples(text_hash, normalized_text, category, signature_json, source_row_index, input_path, output_path) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                text_hash,
                normalized_text,
                category,
                json.dumps(signature),
                source_row_index,
                str(input_path.resolve()),
                str(output_path.resolve()),
            ),
        )
        sample_id = int(cursor.lastrowid)
        for band, bucket_key in signature_band_keys(signature, self.bands):
            self.connection.execute(
                "INSERT OR IGNORE INTO lsh_buckets(band, bucket_key, sample_id) VALUES(?, ?, ?)",
                (band, bucket_key, sample_id),
            )


def maybe_bootstrap_existing_outputs(store: DeduplicationStore, output_directory: Path) -> None:
    if store.count_samples() > 0:
        return

    bootstrapped = 0
    for category in CATEGORY_NAMES:
        output_path = output_directory / f"{category}.jsonl"
        if not output_path.is_file():
            continue

        with output_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in existing output {output_path} at line {line_number}: {exc.msg}"
                    ) from exc

                qa_pair = extract_question_answer_pair(record)
                if qa_pair is None:
                    continue

                question, answer = qa_pair
                normalized_text = build_dedupe_text(question, answer)
                text_hash = sha256_text(normalized_text)
                if store.has_exact_duplicate(text_hash):
                    continue

                store.add_sample(
                    normalized_text=normalized_text,
                    text_hash=text_hash,
                    category=category,
                    source_row_index=extract_source_row_index(record),
                    input_path=output_path,
                    output_path=output_path,
                )
                bootstrapped += 1

    if bootstrapped:
        store.connection.commit()
        LOGGER.info("Bootstrapped %s existing accepted samples into the de-duplication index", bootstrapped)


def append_jsonl_line(output_path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_directory(output_path)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def write_reject_log(
    reject_log_path: Optional[Path],
    sample: ParsedSample,
    reject_labels: List[str],
    rationale: str,
    used_llm: bool,
) -> None:
    if reject_log_path is None:
        return

    payload = {
        "line_number": sample.line_number,
        "source_row_index": sample.source_row_index,
        "question": sample.question,
        "answer": sample.answer,
        "reject_labels": reject_labels,
        "rationale": rationale,
        "used_llm": used_llm,
    }
    append_jsonl_line(reject_log_path, payload)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_directory(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    temp_path.replace(path)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_perm <= 0:
        raise ValueError("--num-perm must be > 0")
    if args.bands <= 0:
        raise ValueError("--bands must be > 0")
    if args.num_perm % args.bands != 0:
        raise ValueError("--num-perm must be divisible by --bands")
    if not 0.0 < args.near_duplicate_threshold <= 1.0:
        raise ValueError("--near-duplicate-threshold must be in the range (0, 1]")
    if args.min_answer_words <= 0:
        raise ValueError("--min-answer-words must be > 0")
    if args.max_total_tokens <= 0:
        raise ValueError("--max-total-tokens must be > 0")
    if args.max_word_ratio <= 1.0:
        raise ValueError("--max-word-ratio must be > 1")


def build_sample(record: Dict[str, Any], line_number: int) -> Optional[ParsedSample]:
    qa_pair = extract_question_answer_pair(record)
    if qa_pair is None:
        return None
    question, answer = qa_pair
    return ParsedSample(
        record=record,
        question=question,
        answer=answer,
        source_row_index=extract_source_row_index(record),
        line_number=line_number,
    )


def should_skip_sample(
    sample: ParsedSample,
    last_source_row_index: int,
    last_line_number: int,
) -> bool:
    if sample.source_row_index is not None:
        return sample.source_row_index <= last_source_row_index
    return sample.line_number <= last_line_number


def summarize_rationale(reject_labels: List[str]) -> str:
    return ", ".join(reject_labels)


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    state_db_path = Path(args.state_db) if args.state_db else output_directory / ".curation_state.sqlite3"
    ensure_parent_directory(state_db_path)

    reject_log_path = Path(args.reject_log) if args.reject_log else None
    if reject_log_path is not None:
        ensure_parent_directory(reject_log_path)

    token_counter = TokenCounter(args.tokenizer_model, args.hf_token)

    client = None
    if not args.disable_llm_review:
        client = OpenAICompatibleClient(
            api_base=args.api_base,
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            api_key=args.api_key,
        )
        if not args.skip_healthcheck:
            LOGGER.info("Running inference server health check")
            client.healthcheck()

    store = DeduplicationStore(
        db_path=state_db_path,
        num_perm=args.num_perm,
        bands=args.bands,
        threshold=args.near_duplicate_threshold,
    )

    maybe_bootstrap_existing_outputs(store, output_directory)

    last_source_row_index, last_line_number = store.load_checkpoint(input_path)
    LOGGER.info(
        "Resuming %s from source_row_index=%s line=%s",
        input_path,
        last_source_row_index,
        last_line_number,
    )

    accepted = 0
    rejected = 0
    skipped = 0
    llm_reviewed = 0

    try:
        for line_number, record in iter_jsonl_records(input_path):
            sample = build_sample(record, line_number)
            if sample is None:
                skipped += 1
                store.update_checkpoint(input_path, None, line_number)
                store.connection.commit()
                LOGGER.warning("Skipping line %s because no question/answer pair was found", line_number)
                continue

            if should_skip_sample(sample, last_source_row_index, last_line_number):
                continue

            reject_labels = deterministic_reject_labels(
                question=sample.question,
                answer=sample.answer,
                token_counter=token_counter,
                min_answer_words=args.min_answer_words,
                max_total_tokens=args.max_total_tokens,
                max_word_ratio=args.max_word_ratio,
            )

            if reject_labels:
                rejected += 1
                write_reject_log(
                    reject_log_path=reject_log_path,
                    sample=sample,
                    reject_labels=reject_labels,
                    rationale=summarize_rationale(reject_labels),
                    used_llm=False,
                )
                store.update_checkpoint(input_path, sample.source_row_index, line_number)
                store.connection.commit()
                last_source_row_index, last_line_number = store.load_checkpoint(input_path)
                continue

            if client is not None:
                try:
                    review = run_llm_review(
                        client=client,
                        question=sample.question,
                        answer=sample.answer,
                        review_max_tokens=args.review_max_tokens,
                        review_temperature=args.review_temperature,
                    )
                    llm_reviewed += 1
                except InferenceError as exc:
                    LOGGER.warning(
                        "LLM review failed at line %s (source_row_index=%s); falling back to heuristic classification: %s",
                        line_number,
                        sample.source_row_index,
                        exc,
                    )
                    review = ReviewDecision(
                        accept=True,
                        category=heuristic_category(sample.question, sample.answer),
                        reject_labels=[],
                        rationale="LLM review fallback after malformed output",
                        used_llm=False,
                    )
            else:
                review = ReviewDecision(
                    accept=True,
                    category=heuristic_category(sample.question, sample.answer),
                    reject_labels=[],
                    rationale="Deterministic-only mode",
                    used_llm=False,
                )

            if not review.accept:
                rejected += 1
                write_reject_log(
                    reject_log_path=reject_log_path,
                    sample=sample,
                    reject_labels=review.reject_labels or ["REJECT_LOW_QUALITY"],
                    rationale=review.rationale,
                    used_llm=review.used_llm,
                )
                store.update_checkpoint(input_path, sample.source_row_index, line_number)
                store.connection.commit()
                last_source_row_index, last_line_number = store.load_checkpoint(input_path)
                continue

            normalized_text = build_dedupe_text(sample.question, sample.answer)
            text_hash = sha256_text(normalized_text)
            if store.has_exact_duplicate(text_hash):
                skipped += 1
                store.update_checkpoint(input_path, sample.source_row_index, line_number)
                store.connection.commit()
                last_source_row_index, last_line_number = store.load_checkpoint(input_path)
                continue

            near_duplicate = store.find_near_duplicate(normalized_text)
            if near_duplicate is not None:
                skipped += 1
                store.update_checkpoint(input_path, sample.source_row_index, line_number)
                store.connection.commit()
                last_source_row_index, last_line_number = store.load_checkpoint(input_path)
                continue

            output_path = output_directory / f"{review.category}.jsonl"
            append_jsonl_line(output_path, sample.record)
            store.add_sample(
                normalized_text=normalized_text,
                text_hash=text_hash,
                category=review.category,
                source_row_index=sample.source_row_index,
                input_path=input_path,
                output_path=output_path,
            )
            store.update_checkpoint(input_path, sample.source_row_index, line_number)
            store.connection.commit()

            accepted += 1
            last_source_row_index, last_line_number = store.load_checkpoint(input_path)
    finally:
        summary_path = output_directory / ".curation_summary.json"
        atomic_write_json(
            summary_path,
            {
                "accepted": accepted,
                "rejected": rejected,
                "skipped": skipped,
                "llm_reviewed": llm_reviewed,
                "input": str(input_path.resolve()),
                "last_checkpoint": {
                    "source_row_index": last_source_row_index,
                    "line_number": last_line_number,
                },
                "state_db": str(state_db_path.resolve()),
            },
        )
        store.close()

    LOGGER.info(
        "Finished curation: accepted=%s rejected=%s skipped=%s llm_reviewed=%s out_dir=%s",
        accepted,
        rejected,
        skipped,
        llm_reviewed,
        output_directory,
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())