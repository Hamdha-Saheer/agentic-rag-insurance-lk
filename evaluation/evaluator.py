import os
# Bypass the RAGAS OpenAI validation check
os.environ["OPENAI_API_KEY"] = "mock-key-to-bypass-internal-ragas-openai-initialization"

import sys
import json
import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import answer_similarity
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Ensure the parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.rag_pipeline import rag_query
from agents.domain_router import route_domain

def load_ground_truth(path: str = 'evaluation/ground_truth.json') -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _print_results(df: pd.DataFrame, metric_name: str, output_path: str):
    mean_val = df[metric_name].mean()
    print('\n' + '='*55)
    print(f' EVALUATION RESULTS — {metric_name.upper()}')
    print('='*55)
    print(f'   {metric_name:<22} : {mean_val:.4f}')
    print(f'   Total questions        : {len(df)}')
    print(f'   Saved to               : {output_path}')
    print('\n   By category:')
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        val    = cat_df[metric_name].mean()
        v_str  = f'{val:.3f}' if not pd.isna(val) else 'nan'
        print(f'      {cat:<22} n={len(cat_df)}  {metric_name}: {v_str}')
    print('='*55)

# ── PROGRESSIVE GENERATION LOOP ──
def run_evaluation(ground_truth_path: str = 'evaluation/ground_truth.json'):
    gt_data = load_ground_truth(ground_truth_path)
    
    suffix = "web" if config.USE_WEB_SEARCH else "local"
    raw_path = f'evaluation/results/raw_answers_{suffix}.csv'
    score_path = f'evaluation/results/similarity_{suffix}.csv'
    
    print(f'=== STARTING GENERATION PASS FOR: [{suffix.upper()} MODE] ===')
    print(f'Generating answers for {len(gt_data)} questions using config settings...\n')

    original_max_tokens = config.MAX_TOKENS
    original_top_k      = config.TOP_K_DOCS

    config.MAX_TOKENS     = 512
    config.TOP_K_DOCS     = 5      

    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        raw_df = pd.read_csv(raw_path)
        eval_data = {
            'question': raw_df['question'].tolist(),
            'answer': raw_df['answer'].tolist(),
            'ground_truth': raw_df['ground_truth'].tolist()
        }
        categories = raw_df['category'].tolist()
        print(f' 🔄 Resuming Generation. Found {len(eval_data["question"])} / {len(gt_data)} existing answers.')
    else:
        os.makedirs('evaluation/results', exist_ok=True)
        eval_data = {'question': [], 'answer': [], 'ground_truth': []}
        categories = []

    try:
        for i, item in enumerate(gt_data):
            q = item['question']
            if q in eval_data['question']:
                continue
                
            print(f'   [{i+1}/{len(gt_data)}] {q[:60]}...')
            domain = route_domain(q)
            
            try:
                result = rag_query(q, domain)
                ans_text = result['answer']
            except Exception as e:
                print(f'\n🛑 API Rate limit caught during generation loop at item {i+1}!')
                print(' Progress up to this point is securely written to your disk.')
                raise e

            eval_data['question'].append(q)
            eval_data['answer'].append(ans_text)
            eval_data['ground_truth'].append(item['ground_truth'])
            categories.append(item.get('category', 'General'))
            
            pd.DataFrame({
                'question':  eval_data['question'],
                'answer':    eval_data['answer'],
                'ground_truth': eval_data['ground_truth'],
                'category':  categories
            }).to_csv(raw_path, index=False)
            
            time.sleep(15)
            
    finally:
        config.MAX_TOKENS     = original_max_tokens
        config.TOP_K_DOCS     = original_top_k

    print(f'\n   ✅ All raw answers successfully generated and cached -> {raw_path}')
    run_scoring_only(raw_path, score_path)

# ── PROGRESSIVE SCORING LOOP (Configured for Answer Similarity) ──
def run_scoring_only(csv_path: str, output_path: str):
    if not os.path.exists(csv_path):
        print(f'ERROR: {csv_path} not found.')
        return

    metric_name = 'answer_similarity'
    df          = pd.read_csv(csv_path)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        scored_df = pd.read_csv(output_path)
        scored_df = scored_df[scored_df[metric_name].notna()]
        scored_questions = scored_df['question'].tolist()
        print(f' 🔄 Resuming Scoring. Loaded {len(scored_questions)} / {len(df)} scored items.')
    else:
        scored_df = pd.DataFrame()
        scored_questions = []

    to_score_df = df[~df['question'].isin(scored_questions)]

    if to_score_df.empty:
        print(' All questions successfully evaluated.')
        _print_results(scored_df, metric_name, output_path)
        return

    # Setup core LLM clients
    groq_llm = LangchainLLMWrapper(ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.LLM_MODEL,
        temperature=0
    ))
    
    # FIXED: Added model_kwargs={'device': 'cpu'} directly to prevent the meta tensor error
    hf_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
    )

    # Inject clients into answer_similarity components
    answer_similarity.llm = groq_llm
    if hasattr(answer_similarity, 'generator') and answer_similarity.generator is not None:
        answer_similarity.generator.llm = groq_llm

    print(f'\n Starting strict sequential scoring pass for remaining {len(to_score_df)} rows using Semantic Similarity...')

    for idx, row in to_score_df.iterrows():
        q_text = row['question']
        a_text = str(row['answer'])
        if len(a_text) > 1200:
            a_text = a_text[:1200] + "..."
            
        gt_text = row['ground_truth']
        cat_text = row.get('category', 'General')

        print(f' -> Scoring item {len(scored_questions) + 1}/{len(df)}: "{q_text[:45]}..."')

        single_item = {
            'question': [q_text],
            'answer': [a_text],
            'contexts': [[a_text]],
            'ground_truth': [gt_text]
        }
        dataset = Dataset.from_dict(single_item)

        try:
            result = evaluate(
                dataset,
                metrics=[answer_similarity],
                llm=groq_llm,
                embeddings=hf_embeddings,
                run_config=RunConfig(max_workers=1, timeout=120)
            )
            
            row_df = result.to_pandas()
            row_df['category'] = cat_text

            if not scored_df.empty:
                scored_df = pd.concat([scored_df, row_df], ignore_index=True)
            else:
                scored_df = row_df

            scored_df.to_csv(output_path, index=False)
            scored_questions.append(q_text)
            print(f'    Row Score: {row_df[metric_name].values[0]:.4f} | Progress written safely.')
            
            # Strict 15-second scoring cooldown
            time.sleep(15)

        except Exception as e:
            print(f'\n [PAUSED] API limit restriction encountered: {e}')
            print(' Progress saved. Please wait for token window recovery before executing again.')
            return

    print(f'\n Evaluation completed successfully!')
    _print_results(scored_df, metric_name, output_path)

if __name__ == '__main__':
    print('='*55)
    print('   RAGAS CONFIGURABLE PAIRS EVALUATOR')
    print('='*55 + '\n')
    
    suffix = "web" if config.USE_WEB_SEARCH else "local"
    target_raw = f'evaluation/results/raw_answers_{suffix}.csv'

    if os.path.exists(target_raw) and os.path.getsize(target_raw) > 0:
        print(f' Found existing answers file -> {target_raw}')
        run_evaluation()
    else:
        print(f' No baseline answers found for [{suffix.upper()}] -> Starting fresh generation pass.\n')
        run_evaluation()
