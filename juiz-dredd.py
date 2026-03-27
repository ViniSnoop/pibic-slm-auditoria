import json
import os
import re
import time
import ollama # type: ignore


# =========================
# Configuração
# =========================

JUIZ_MODEL = "gemma3:12b"  # ajuste se necessário

DIR_ENTRADA = "saidas_experimento"
DIR_SAIDA = "saidas_juiz"
os.makedirs(DIR_SAIDA, exist_ok=True)

# Seus 5 arquivos (1 por modelo gerador)
INPUT_FILES = [
    "resultados_BATCH_mistral_7b.jsonl",
    "resultados_BATCH_llama3.1_8b.jsonl",
    "resultados_BATCH_qwen2.5_7b.jsonl",
    "resultados_BATCH_ministral-3_3b.jsonl",
    "resultados_BATCH_gemma3_4b.jsonl",
]

# INPUT_FILES = ["resultados_experimento_v2_3_modelos.jsonl"]
# INPUT_FILES = ["resultados_NEGATIVO_neggen-v1.jsonl"]


SEED = 66
TEMPERATURE = 0.0
MAX_TENTATIVAS = 2  # 1 tentativa + 1 retry

# =========================
# Prompt do juiz (scores only)
# =========================

JUDGE_PROMPT = """
TASK: Evaluate an information security maturity checklist response.

You will receive:
1) ORIGINAL QUESTION (includes the required context)
2) MODEL ANSWER (checklist with 6 levels)

Rate the MODEL ANSWER from 1 to 5 on each dimension:

DIM1 FORMAT (1-5)
- 5: Exactly 6 non-empty lines; each line starts with “[ ] Level X:”; levels 0-5 all present; no extra text; each line has 1-3 sentences.
- 4: Same as 5 but with ONE minor deviation (e.g., one line slightly longer/shorter, extra text or one small formatting inconsistency).
- 3: Multiple format issues, but still recognizable as a 0-5 checklist.
- 2: Major format issues (missing levels or extra levels, wrong labels) that hinder use.
- 1: Not a valid checklist format.
If any level is missing (0-5) OR any extra level exists OR there is any extra text outside the 6 lines, then FORMAT must be 2 or 1 (never 3-5).

DIM2 CONCEPTS (1-5)
- 5: Clear, realistic maturity progression 0→5; uses correct infosec concepts (policies, controls, governance, risk, audit, monitoring); no contradictions or invented standards.
- 4: Mostly correct progression and concepts, with minor gaps or mild vagueness.
- 3: Generally plausible but shallow, partially generic, or with noticeable conceptual gaps.
- 2: Many conceptual problems, weak maturity logic, or confusing security concepts.
- 1: Incorrect/confused maturity logic or severe hallucinations.

DIM3 CONTEXT (1-5)
- 5: Strongly tailored to the requested context with specific terms and examples; no mixing of contexts.
- 4: Context is present and mostly specific, but still somewhat generic in places.
- 3: Mentions the context but examples remain generic (“the organization”, “the data”) with limited tailoring.
- 2: Context is weak, partially wrong, or mixes contexts.
- 1: Context is wrong or heavily confused.

DIM4 CLARITY (1-5)
- 5: Clear writing; levels are distinct (no repetition); examples are concrete; concise enough to be usable.
- 4: Clear overall with minor redundancy or minor imprecision.
- 3: Understandable but repetitive or vague; examples not very concrete.
- 2: Hard to follow, redundant, or unclear distinctions between levels.
- 1: Very unclear or incoherent.

IMPORTANT
- Assign scores independently per dimension (a 5 in FORMAT does not imply 5 in CONCEPTS, etc.).
- If any dimension does not meet the requirements for 5, do not give 5 for that dimension.
- Return ONLY the 4 scores below. No justification. No extra text.

EXAMPLES (for calibration only)

Example A (should score high):

ORIGINAL QUESTION:
About information security governance and policies, hospital context

MODEL ANSWER:
[ ] Level 0: No formal governance or policies; staff handle patient data and EMR access informally, and there is no consistent approval for access to clinical records.
[ ] Level 1: Basic rules exist but are inconsistent; unit managers approve access ad hoc, and staff receive informal guidance on handling patient records and privacy.
[ ] Level 2: A baseline governance plan exists and is partially documented; roles for EMR access are defined in some departments, and basic onboarding training covers patient-data handling but is irregular.
[ ] Level 3: Documented policies are formally approved by leadership; RBAC for EMR is defined by clinical role (e.g., physician, nursing, admin), and access requests, exceptions, and terminations follow a standard workflow.
[ ] Level 4: Policies are reviewed on a defined cadence; EMR access is logged and routinely audited (including privileged access), monitoring alerts are in place for suspicious record access, and periodic internal checks validate adherence across clinical systems.
[ ] Level 5: Governance is metrics-driven and continuously improved; KPIs cover access violations, audit findings, and incident trends, policies are updated after incidents and risk reviews, and security governance integrates compliance obligations for patient privacy and clinical operations.
EXPECTED SCORES:
FORMAT: 5
CONCEPTS: 4
CONTEXT: 5
CLARITY: 4

Example B (should score mid / “mixed”):

ORIGINAL QUESTION:
About information security governance and policies, hospital context

MODEL ANSWER:
( ) Level 0: No formal governance; each unit decides how to handle patient data, and access to records is granted informally.
( ) Level 1: Some basic rules exist (often verbal); staff are told to “be careful” with patient information, but enforcement depends on the supervisor.
( ) Level 2: A simple written policy exists, but it is generic; it mentions protecting patient data and controlling access, yet it does not define roles, approval flows, or review frequency.
( ) Level 3: Policies are approved and communicated; access to systems is requested through a standard process, but audits and monitoring are still occasional and mostly reactive.
( ) Level 4: Reviews happen periodically; audits check access and compliance for key systems, but metrics are limited and improvements are not consistently tracked.
( ) Level 5: Continuous improvement is intended; incidents sometimes lead to updates, but governance is not fully metrics-driven and still varies between departments.
EXPECTED SCORES:
FORMAT: 4
CONCEPTS: 3
CONTEXT: 3
CLARITY: 3

NOW EVALUATE THE INPUT BELOW.

FORMAT TO RETURN:
FORMAT: 1-5
CONCEPTS: 1-5
CONTEXT: 1-5
CLARITY: 1-5

ORIGINAL QUESTION:
{original_question}

MODEL ANSWER:
{model_answer}

""".strip()

# =========================
# Parsing / validação
# =========================

LINE_PATTERNS = {
    "FORMAT": re.compile(r"^FORMAT:\s*([1-5])\s*$", re.IGNORECASE),
    "CONCEPTS": re.compile(r"^CONCEPTS:\s*([1-5])\s*$", re.IGNORECASE),
    "CONTEXT": re.compile(r"^CONTEXT:\s*([1-5])\s*$", re.IGNORECASE),
    "CLARITY": re.compile(r"^CLARITY:\s*([1-5])\s*$", re.IGNORECASE),
}

def parse_scores(text: str):
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) != 4:
        return None, f"Esperava 4 linhas, veio {len(lines)}."

    scores = {}
    for ln in lines:
        ok = False
        for key, pat in LINE_PATTERNS.items():
            m = pat.match(ln)
            if m:
                scores[key] = int(m.group(1))
                ok = True
                break
        if not ok:
            return None, f"Linha fora do padrão: {ln}"

    missing = [k for k in LINE_PATTERNS.keys() if k not in scores]
    if missing:
        return None, f"Faltando chaves: {missing}"

    return scores, None

# =========================
# IO helpers
# =========================

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_num, json.loads(line)

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def output_name_for_input(input_filename: str):
    base = os.path.splitext(os.path.basename(input_filename))[0]
    juiz = JUIZ_MODEL.replace(":", "_").replace("/", "_")
    return f"julgamento__{base}__juiz_{juiz}.jsonl"

# =========================
# Juiz
# =========================

def judge_one(original_question: str, model_answer: str):
    prompt = JUDGE_PROMPT.format(
        original_question=original_question,
        model_answer=model_answer
    )

    last_raw = None
    last_err = None
    last_dt = None

    for attempt in range(1, MAX_TENTATIVAS + 1):
        t0 = time.time()
        resp = ollama.generate(
            model=JUIZ_MODEL,
            prompt=prompt,
            options={
                "temperature": TEMPERATURE,
                "seed": SEED,
                "top_k": 1,
                "top_p": 1,
            }
        )
        dt = time.time() - t0
        last_dt = dt

        raw = (resp.get("response") or "").strip()
        last_raw = raw

        scores, err = parse_scores(raw)
        if scores is not None:
            return scores, raw, dt, attempt, None

        last_err = err

        # retry com lembrete de formato (sem adicionar justificativa)
        prompt = (
            prompt
            + "\n\nREMINDER: Output MUST be exactly 4 lines:\n"
              "FORMAT: [1-5]\n"
              "CONCEPTS: [1-5]\n"
              "CONTEXT: [1-5]\n"
              "CLARITY: [1-5]\n"
              "No extra text."
        )

    return None, last_raw, last_dt, MAX_TENTATIVAS, last_err

# =========================
# Execução
# =========================

def main():
    print(f"Juiz: {JUIZ_MODEL}")
    print(f"Entrada dir: {DIR_ENTRADA}")
    print(f"Saída dir: {DIR_SAIDA}")

    for input_file in INPUT_FILES:
        in_path = os.path.join(DIR_ENTRADA, input_file)
        if not os.path.exists(in_path):
            print(f"[SKIP] Não achei: {in_path}")
            continue

        out_file = output_name_for_input(input_file)
        out_path = os.path.join(DIR_SAIDA, out_file)

        print(f"\n=== Julgando arquivo: {input_file} ===")
        print(f"Saída: {out_file}")

        total = ok = fail = 0

        for line_num, item in iter_jsonl(in_path):
            total += 1

            # Compatível com seu pipeline: "pergunta_final" e "resposta_ia"
            original_question = item.get("pergunta_final") or item.get("pergunta") or ""
            model_answer = item.get("resposta_ia") or ""

            if not original_question or not model_answer:
                fail += 1
                append_jsonl(out_path, {
                    "status": "error_missing_fields",
                    "line_num": line_num,
                    "input_file": input_file,
                    "error": "Campos ausentes: precisa de pergunta_final/pergunta e resposta_ia.",
                    "input_ref": {
                        "modelo_gerador": item.get("modelo"),
                        "timestamp_gerador": item.get("timestamp"),
                    }
                })
                continue

            scores, raw, dt, attempt, err = judge_one(original_question, model_answer)

            if scores is None:
                fail += 1
                append_jsonl(out_path, {
                    "status": "error_parse",
                    "line_num": line_num,
                    "input_file": input_file,
                    "attempts": attempt,
                    "raw_judge_output": raw,
                    "error": err,
                    "input_ref": {
                        "modelo_gerador": item.get("modelo"),
                        "timestamp_gerador": item.get("timestamp"),
                        "pergunta_final": original_question,
                    }
                })
            else:
                ok += 1
                append_jsonl(out_path, {
                    "status": "ok",
                    "line_num": line_num,
                    "input_file": input_file,
                    "judge_model": JUIZ_MODEL,
                    "attempt": attempt,
                    "tempo_execucao_s": round(dt, 4) if dt is not None else None,
                    "scores": scores,
                    "raw_judge_output": raw,
                    "input_ref": {
                        "modelo_gerador": item.get("modelo"),
                        "timestamp_gerador": item.get("timestamp"),
                        "pergunta_final": original_question,
                    }
                })

            if total % 25 == 0:
                print(f"[PROGRESS] {input_file}: {total} | ok={ok} | fail={fail}")

        print(f"[DONE] {input_file}: total={total} | ok={ok} | fail={fail}")
        print(f"Output: {out_path}")

    print("\nFinalizado: todos os arquivos processados.")

if __name__ == "__main__":
    main()
