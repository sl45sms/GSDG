import json


SYSTEM_PROMPT = (
    "Παράγεις συνθετικά ζεύγη ερώτησης και απάντησης στα Ελληνικά. "
    "Ακολουθείς αυστηρά τη ζητούμενη μορφή εξόδου."
)


def build_user_prompt(source_text: str) -> str:
    return (
        "Με βάση το παρακάτω κείμενο, δημιούργησε ΜΙΑ ερώτηση κατανόησης "
        "και την απάντησή της στα Ελληνικά.\n\n"
        "Κείμενο:\n"
        '"""\n'
        f"{source_text}\n"
        '"""\n\n'
        "Απάντησε ΜΟΝΟ με JSON της μορφής:\n"
        '{"question": "...", "answer": "..."}'
    )


def build_chatml_record(
    *,
    user_prompt: str,
    question: str,
    answer: str,
    dataset_name: str,
    split_name: str,
    row_id: str,
    source_fields: str,
) -> dict:
    assistant_content = json.dumps(
        {"question": question, "answer": answer},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "meta": {
            "dataset": dataset_name,
            "split": split_name,
            "row_id": row_id,
            "source_fields": source_fields,
        },
    }
