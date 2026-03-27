import time
import ollama
import json
import os
import random


# =========================
# Configurações do Experimento
# =========================

modelos = ['gemma3:4b', 'ministral-3:3b', 'qwen2.5:7b', "llama3.1:8b", "mistral:7b"]

ARQUIVO_PERGUNTAS = "questions-correcao.json"
DIR_SAIDA = "saidas_experimento"
os.makedirs(DIR_SAIDA, exist_ok=True)

# Contextos (ordem importa para o round-robin)
contexts = [
    "hospital context",
    "software enterprise context",
    "bank institution context",
    "commerce enterprise context",
    "government agency context"
]

# MODO:
# - "TESTE": pega uma amostra pequena (N_TESTE) e usa contexto round-robin nessa amostra
# - "BATCH": roda todas as perguntas e aplica contexto round-robin por índice global
MODO = "BATCH"  # troque para "BATCH" quando for rodar o lote real

N_TESTE = 5           # usado apenas no modo TESTE
SEED_EXPERIMENTO = 66 # usado apenas para escolher a amostra no modo TESTE (BATCH não usa random)


# =========================
# Prompt do sistema (CoT / instrução)
# =========================

cot = """
TASK: Create a maturity assessment checklist.

PROCESS (internal only, DO NOT write this):
1. Identify the topic and the context.
2. Generate 6 maturity level descriptions (0 to 5), each with 1-3 sentences and a specific example.

OUTPUT FORMAT (this is what you must return):

[ ] Level 0: Would not know how to answer - [add 1-3 sentences with specific example]
[ ] Level 1: Has, but informally - [add 1-3 sentences with specific example]
[ ] Level 2: Has a basic, undocumented plan - [add 1-3 sentences with specific example]
[ ] Level 3: Yes, with a clear and documented definition - [add 1-3 sentences with specific examples]
[ ] Level 4: Yes, documented and regularly reviewed - [add 1-3 sentences with specific examples]
[ ] Level 5: Yes, documented, reviewed, and continuously improved - [add 1-3 sentences with specific examples]

CRITICAL: Return ONLY the 6 checkboxes above. NO titles, NO explanations, NO steps.
""".strip()


# =========================
# Utilitários
# =========================

def sanitize_model_name(model_name: str) -> str:
    return model_name.replace(":", "_").replace("/", "_")

def load_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("questions.json deve ser uma LISTA de strings (perguntas).")

    data = [q.strip() for q in data if q and q.strip()]
    if not data:
        raise ValueError("questions.json está vazio após limpeza.")
    return data


# =========================
# Carrega perguntas
# =========================

perguntas = load_questions(ARQUIVO_PERGUNTAS)
print(f"Carregadas {len(perguntas)} perguntas de {ARQUIVO_PERGUNTAS}.")


# =========================
# Plano de execução (sem lixo residual)
# =========================

plano_execucao = []  # lista de tuplas: (pergunta_base, contexto, pergunta_final)

if MODO == "TESTE":
    random.seed(SEED_EXPERIMENTO)
    perguntas_amostra = random.sample(perguntas, k=min(N_TESTE, len(perguntas)))

    # Contexto fixo por índice dentro da amostra (round-robin)
    for i, pergunta_base in enumerate(perguntas_amostra):
        contexto = contexts[i % len(contexts)]
        pergunta_final = f"{pergunta_base}, {contexto}"
        plano_execucao.append((pergunta_base, contexto, pergunta_final))

elif MODO == "BATCH":
    # Sem random: 1 contexto fixo por pergunta em ordem (round-robin)
    for i, pergunta_base in enumerate(perguntas):
        contexto = contexts[i % len(contexts)]
        pergunta_final = f"{pergunta_base}, {contexto}"
        plano_execucao.append((pergunta_base, contexto, pergunta_final))

else:
    raise ValueError("MODO inválido. Use 'TESTE' ou 'BATCH'.")

print(f"Plano de execução montado: {len(plano_execucao)} itens (MODO={MODO}).")


# =========================
# Loop de Execução
# =========================

for modelo in modelos:
    nome_arquivo = f"resultados_{MODO}_{sanitize_model_name(modelo)}.jsonl"
    caminho_saida = os.path.join(DIR_SAIDA, nome_arquivo)

    print(f"\n=== Modelo: {modelo} | Rodando {len(plano_execucao)} perguntas | Saída: {caminho_saida} ===")

    for pergunta_base, contexto, pergunta_final in plano_execucao:
        prompt_completo = cot + "\n\nQuestion: " + pergunta_final

        inicio = time.time()
        resp = ollama.generate(
            model=modelo,
            prompt=prompt_completo,
            options={
                "temperature": 0.0,
                "seed": SEED_EXPERIMENTO,  # ok manter no log (mesmo no BATCH)
                # Para você revisar/testar:
                "top_k": 1,
                "top_p": 1,
            }
        )
        fim = time.time()
        duracao = fim - inicio

        resultado = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "modo": MODO,
            "modelo": modelo,

            "pergunta_base": pergunta_base,
            "contexto": contexto,
            "pergunta_final": pergunta_final,

            "prompt_system": cot,
            "resposta_ia": resp.get("response", ""),
            "tempo_execucao_s": round(duracao, 4),
            "parametros": {
                "temperature": 0.0,
                "seed": SEED_EXPERIMENTO,
                "top_k": 1,
                "top_p": 1,
            }
        }

        with open(caminho_saida, "a", encoding="utf-8") as f:
            f.write(json.dumps(resultado, ensure_ascii=False) + "\n")

        print(f"[OK] {modelo} - {pergunta_base[:35]}... + {contexto} ({duracao:.2f}s)")

print("\nColeta finalizada!")
print(f"Arquivos gerados em: {DIR_SAIDA}")