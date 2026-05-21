import asyncio
import json
import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate
from ragas import SingleTurnSample, EvaluationDataset
from langchain_openai import ChatOpenAI
from ragas.metrics import (
    _Faithfulness,
    _ResponseRelevancy,
    _LLMContextPrecisionWithReference,
    _LLMContextRecall,
)
from ragas.run_config import RunConfig

oai_llm = ChatOpenAI(model="gpt-4o-mini")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def print_eval_res(eval_result):
    scores = eval_result.scores
    eval_str = ' | Q | '
    for k in scores[0].keys():
        eval_str = eval_str + str(k) + ' | '
    print(eval_str)
    for i, score in enumerate(scores):
        eval_str = ' | ' + str(i + 1) + ' | '
        for k in score.keys():
            eval_str = eval_str + str(score[k]) + ' | '
        print(eval_str)
    res = eval_result.to_pandas()
    means = res.mean(numeric_only=True).to_dict()
    print("\n📈 Averages:")
    for k, v in means.items():
        print(f"- {k}: {v:.3f}")

async def evaluate_rag_system(test_path="../seed/qna_test.json"):
    test_data = load_jsonl(test_path)
    results = []

    for item in test_data:
        question = item["question"]
        reference_answer = item["answer"]
        url = 'http://localhost:8000/ask'
        myobj = {'question': question}

        try:
            response = requests.post(url, json=myobj, timeout=30)
            response.raise_for_status()
            if not response.text.strip():
                print(f"[WARN] Empty response for question: {question!r}")
                continue
            res = response.json()
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP {response.status_code} for question: {question!r}\n  {e}")
            continue
        except requests.exceptions.JSONDecodeError:
            print(f"[ERROR] Non-JSON response for question: {question!r}")
            print(f"  Raw body: {response.text[:300]!r}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed for question: {question!r}\n  {e}")
            continue

        answer = res['answer']
        contexts = res['contexts']

        results.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference_answer
        ))

    if not results:
        print("[ABORT] No successful responses — check that your server is running on :8000")
        return

    ds = EvaluationDataset(results)
    metrics = [
        _Faithfulness(llm=oai_llm),
        _ResponseRelevancy(llm=oai_llm),
        _LLMContextPrecisionWithReference(llm=oai_llm),
        _LLMContextRecall(llm=oai_llm),
    ]
    run_config = RunConfig(max_workers=16, timeout=30)
    eval_result = evaluate(dataset=ds, metrics=metrics, run_config=run_config)
    print("RAGAS Evals Results")
    print_eval_res(eval_result)

if __name__ == "__main__":
    asyncio.run(evaluate_rag_system())

    