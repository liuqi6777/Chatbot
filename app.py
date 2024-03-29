import os
import sys
import sqlite3
import torch
import torch.nn.functional as F

from typing import List
from itertools import chain
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModel, OpenAIGPTLMHeadModel, BertTokenizer, HfArgumentParser
from pyserini.search.lucene import LuceneSearcher


torch.manual_seed(42)


@dataclass
class Arguments:
    do_retrieval: bool = False
    do_generate: bool = False
    
    device: str = 'cpu'
    
    max_history: int = 2
    max_length: int = 30
    min_length: int = 1
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 0.9
    no_sample: bool = False


def retrieval(utterance: str, searcher, database) -> List[str]:
    # for u in utterances:
    #     hits = searcher.search(u)
    hits = searcher.search(utterance)
    sql = f"select response from responses where query_id = {hits[0].docid}"
    res = database.execute(sql)
    candidates = [x[0] for x in res.fetchall()]
    return candidates


def rerank(candidates: List[str], utterance: str, tokenizer, model) -> str:
    cands_input = tokenizer(candidates, padding=True, return_tensors='pt')
    uttr_input = tokenizer(utterance, padding=True, return_tensors='pt')

    with torch.no_grad():
        cands_embeddings = model(**cands_input)
        uttr_embeddings = model(**uttr_input)
        
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Perform pooling. In this case, mean pooling.
    cands_embeddings = mean_pooling(
        cands_embeddings, cands_input['attention_mask'])
    uttr_embeddings = mean_pooling(
        uttr_embeddings, uttr_input['attention_mask'])

    score = cands_embeddings @ uttr_embeddings.T
    top = torch.argmax(score)
    response = candidates[top]
    idx = -1 if top == len(candidates) - 1 else top
    return response, score[top].item(), idx


SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def build_input_from_segments(history, reply, tokenizer, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, pad, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS)
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if i %
                                 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:])
                                          for _ in s]
    return instance, sequence


def sample_sequence(history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(
            history, current_output, tokenizer, with_eos=False)
        input_ids = torch.tensor(
            instance["input_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(
            instance["token_type_ids"], dtype=torch.long, device=args.device).unsqueeze(0)

        outputs = model(input_ids, token_type_ids=token_type_ids)
        logits = outputs.logits
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[
            1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def generate(historys, tokenizer, model, args):
    with torch.no_grad():
        historys = historys[-(2 * args.max_history + 1):]
        history_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in historys]
        out_ids = sample_sequence(history_ids, tokenizer, model, args)
        response = tokenizer.decode(out_ids, skip_special_tokens=True)
        return response


if __name__ == '__main__':

    parser = HfArgumentParser((Arguments, ))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]
        
    assert args.do_retrieval or args.do_generate, "must select one of retrieval mode or generate mode, or both"
    
    if args.do_retrieval:
        searcher = LuceneSearcher('index/queries/')
        searcher.set_language('zh')
        con = sqlite3.connect('index/responses.sqlite')
        cur = con.cursor()
        
        reranker_tokenizer = AutoTokenizer.from_pretrained(
            'uer/sbert-base-chinese-nli')
        reranker = AutoModel.from_pretrained('uer/sbert-base-chinese-nli')
        reranker.eval()

    if args.do_generate:
        generator = OpenAIGPTLMHeadModel.from_pretrained(
            "thu-coai/CDial-GPT_LCCC-large", ignore_mismatched_sizes=True)
        generator_tokenizer = BertTokenizer.from_pretrained(
            "thu-coai/CDial-GPT_LCCC-large")
        generator.eval()


    historys = []
    while True:
        raw_text = input(">>> ")
        if not raw_text:
            break
        historys.append(raw_text)
        
        candidates = []

        if args.do_retrieval:
            try:
                retrieval_cands = retrieval(raw_text, searcher, cur)
            except:
                retrieval_cands = []
            finally:
                candidates += retrieval_cands

        if args.do_generate:
            generate_cands = generate(historys, generator_tokenizer, generator, args)
            candidates.append(generate_cands)
        
        if len(candidates) > 1:
            response, score, _ = rerank(candidates=candidates, utterance=raw_text, tokenizer=reranker_tokenizer, model=reranker)
        else:
            response = candidates[0]
        
        print(response.replace(" ", ""))

    cur.close()
    con.close()
