import json
import os
import sys
import logging
import re
from typing import Dict, List, Optional, Tuple
import random


# Add mediQ src to path
sys.path.append("/media/ssd-3t/isviridov/alphaevolve/mediQ/src")

# Default data path
DEFAULT_DATA_PATH = "/media/ssd-3t/isviridov/alphaevolve/mediQ/data/all_train_convo.jsonl"

# Models specification (must match served model names). Set by run.sh.
# Четыре env-переменные: две модели и два URL.
#   MEDIQ_EXPERT_MODEL_NAME  – модель врача (и QG)
#   MEDIQ_PATIENT_MODEL_NAME – модель пациента
#   MEDIQ_EXPERT_URL         – base_url для врача/QG
#   MEDIQ_PATIENT_URL        – base_url для пациента
MEDIQ_EXPERT_MODEL = os.environ.get("MEDIQ_EXPERT_MODEL_NAME")
MEDIQ_EXPERT_MODEL_QG = MEDIQ_EXPERT_MODEL
MEDIQ_PATIENT_MODEL = os.environ.get("MEDIQ_PATIENT_MODEL_NAME", MEDIQ_EXPERT_MODEL)

MEDIQ_EXPERT_URL = os.environ.get("MEDIQ_EXPERT_URL")
MEDIQ_PATIENT_URL = os.environ.get("MEDIQ_PATIENT_URL", MEDIQ_EXPERT_URL)

# Other fixed params
MAX_CASES = 50

PROGRAM_PARAM_RANGES = {
    "self_consistency": (1, 3),
    "max_questions": (1, 5),
}

# ---------------------------------------------------------------------------
# Three-part batch composition: anchor + hard buffer + random
#   MEDIQ_BATCH_MODE      = "composed" | "random"
#   MEDIQ_ANCHOR_SIZE     = number of stable anchor samples per batch
#   MEDIQ_HARD_BUFFER_SIZE = max hard-buffer slots (filled by weighted sampling)
#   MEDIQ_ANCHOR_REFRESH  = anchor set lifetime in generations
# ---------------------------------------------------------------------------
BATCH_MODE = os.environ.get("MEDIQ_BATCH_MODE", "composed")
BATCH_STATE_FILE = os.environ.get(
    "MEDIQ_BATCH_STATE_FILE",
    os.path.join(os.path.dirname(DEFAULT_DATA_PATH), "batch_state.json"),
)
ANCHOR_SIZE = int(os.environ.get("MEDIQ_ANCHOR_SIZE", "10"))
HARD_BUFFER_MAX_SIZE = int(os.environ.get("MEDIQ_HARD_BUFFER_SIZE", "15"))
ANCHOR_REFRESH_INTERVAL = int(os.environ.get("MEDIQ_ANCHOR_REFRESH", "5"))
ANCHOR_BASE_SEED = 42
HARD_BUFFER_BASE_SEED = 123456


class BatchComposer:
    """Composes evaluation batches from three parts for stable cross-generation comparison.

    1. **Anchor set** — small fixed subset, refreshed every ``ANCHOR_REFRESH_INTERVAL``
       generations.  Provides a stable baseline for cross-generation fitness comparison.
    2. **Hard buffer** — dynamically maintained set of samples with high error rates
       across recent evaluations.  Keeps selective pressure on genuinely difficult cases.
    3. **Random portion** — fresh random samples drawn per-generation for diversity.

    All candidates within the same generation receive identical batches (the batch
    is deterministic given ``iteration`` and the current ``sample_stats``).
    """

    def __init__(
        self,
        total_data_size: int,
        batch_size: int = MAX_CASES,
        state_path: str = BATCH_STATE_FILE,
    ):
        self.total_data_size = total_data_size
        self.batch_size = batch_size
        self.state_path = state_path

    # -- state persistence ----------------------------------------------------

    @staticmethod
    def _default_state() -> dict:
        return {"sample_stats": {}}

    def _read_state(self) -> dict:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_state()

    def _write_state(self, state: dict) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, self.state_path)

    # -- batch part builders ---------------------------------------------------

    def _compute_anchors(self, iteration: int) -> List[int]:
        anchor_epoch = iteration // ANCHOR_REFRESH_INTERVAL
        rng = random.Random(ANCHOR_BASE_SEED + anchor_epoch)
        return sorted(
            rng.sample(range(self.total_data_size), k=min(ANCHOR_SIZE, self.total_data_size))
        )

    def _compute_hard_buffer(
        self, sample_stats: dict, exclude: set, iteration: int,
    ) -> List[int]:
        indices: List[int] = []
        weights: List[float] = []
        for idx_str, s in sample_stats.items():
            idx = int(idx_str)
            if idx in exclude or idx >= self.total_data_size:
                continue
            evals = s.get("evals", 0)
            if evals < 1:
                continue
            error_rate = s.get("errors", 0) / evals
            if error_rate > 0:
                indices.append(idx)
                weights.append(error_rate)
        if not indices:
            return []
        k = min(HARD_BUFFER_MAX_SIZE, len(indices))
        rng = random.Random(HARD_BUFFER_BASE_SEED + iteration)
        selected: List[int] = []
        for _ in range(k):
            chosen = rng.choices(indices, weights=weights, k=1)[0]
            pos = indices.index(chosen)
            indices.pop(pos)
            weights.pop(pos)
            selected.append(chosen)
            if not indices:
                break
        return selected

    # -- public API ------------------------------------------------------------

    def compose_batch(self, iteration: int) -> List[int]:
        """Return a list of sample indices forming the evaluation batch."""
        import fcntl

        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        lock_path = self.state_path + ".lock"

        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                state = self._read_state()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

        anchor_indices = self._compute_anchors(iteration)
        anchor_set = set(anchor_indices)

        hard_indices = self._compute_hard_buffer(
            state.get("sample_stats", {}), anchor_set, iteration,
        )
        hard_set = set(hard_indices)

        used = anchor_set | hard_set
        remaining = [i for i in range(self.total_data_size) if i not in used]
        random_needed = max(0, self.batch_size - len(anchor_indices) - len(hard_indices))
        rng = random.Random(iteration)
        random_indices = rng.sample(remaining, k=min(random_needed, len(remaining)))

        batch = anchor_indices + hard_indices + random_indices

        log_info(
            f"[BatchComposer] iter={iteration} "
            f"anchor={len(anchor_indices)} hard={len(hard_indices)} "
            f"random={len(random_indices)} total={len(batch)}"
        )
        return batch

    def update_results(
        self, batch_indices: List[int], correct_mask: List[bool], iteration: int,
    ) -> None:
        """Increment per-sample error/eval counters."""
        import fcntl

        lock_path = self.state_path + ".lock"
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = self._read_state()
                stats = state.setdefault("sample_stats", {})

                for idx, correct in zip(batch_indices, correct_mask):
                    key = str(idx)
                    if key not in stats:
                        stats[key] = {"errors": 0, "evals": 0}
                    stats[key]["evals"] += 1
                    if not correct:
                        stats[key]["errors"] += 1

                self._write_state(state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

def log_info(message, logger_name="detail_logger", print_to_std=False, type="info"):
    logger = logging.getLogger(logger_name)
    if type == "error": return logger.error(message)
    if logger: logger.info(message)
    if print_to_std: print(message + "\n")


def log_info_expert(message, logger="detail_logger", print_to_std=False):
    if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(logger)
    if logger: logger.info(message)
    if print_to_std: print(message + "\n")


global models
models = {}

_expert_responses: List[str] = []


class ModelCacheHTTP:
    """ModelCache that uses vLLM server via OpenAI-compatible HTTP API."""
    
    def __init__(self, model_name, base_url, **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        self.args = kwargs
        self._init_client()
    
    def _init_client(self):
        import httpx
        from openai import OpenAI
        proxy_url = self.args.get("proxy_url")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.args.get("api_key"),
            http_client=httpx.Client(proxy=proxy_url) if proxy_url else None,
        )
        log_info(f"[ModelCacheHTTP] Initialized client for {self.model_name} at {self.base_url}"
                 + (f" via proxy {proxy_url}" if proxy_url else ""))
    
    def generate(self, messages):
        log_info(f"[{self.model_name}][INPUT]: {messages}")

        temperature = self.args.get("temperature")
        max_tokens = self.args.get("max_tokens")
        top_p = self.args.get("top_p")
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presense_penalty", 0)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            if not response.choices:
                raise RuntimeError(f"[PROVIDER ERROR, not a mutant bug] Empty choices in response: {response}")
            response_text = response.choices[0].message.content
        except Exception as e:
            log_info(f"[{self.model_name}][ERROR]: {e}", type="error")
            raise

        log_info(
            f"[{self.model_name}][OUTPUT]: "
            f"{{'response_text': '{response_text[:100]}...'}}"
        )
        return response_text


def get_response(messages, model_name, **kwargs):
    is_patient = model_name == MEDIQ_PATIENT_MODEL
    if is_patient:
        base_url = MEDIQ_PATIENT_URL
        api_key = os.environ.get("MEDIQ_PATIENT_API_KEY")
        proxy_url = os.environ.get("MEDIQ_PATIENT_PROXY_URL")
    else:
        base_url = MEDIQ_EXPERT_URL
        api_key = os.environ.get("MEDIQ_EXPERT_API_KEY")
        proxy_url = None

    model_cache = models.get(model_name, None)
    if model_cache is None:
        log_info(f"[get_response] Initializing HTTP client for {model_name} at {base_url}")
        model_cache = ModelCacheHTTP(model_name, base_url=base_url, api_key=api_key, proxy_url=proxy_url, **kwargs)
        models[model_name] = model_cache
    
    result = model_cache.generate(messages)
    if not is_patient and isinstance(result, str):
        _expert_responses.append(result)
    return result


def shutdown_models():
    global models
    for mc in models.values():
        mc.model = None
    models.clear()

    
def load_data(filename):
    with open(filename, "r") as json_file:
        json_list = list(json_file)
    data = [json.loads(line) for line in json_list]

    # Normalize any precomputed facts / atomic_facts:
    # - strip leading numbering like "1.", "2)", "3 -", etc.
    # - strip leading dash markers
    for sample in data:
        for key in ("facts", "atomic_facts"):
            facts = sample.get(key)
            if isinstance(facts, list):
                cleaned_facts: list[str] = []
                for s in facts:
                    if not isinstance(s, str):
                        continue
                    s = s.strip()
                    if not s:
                        continue
                    # Remove leading dash markers
                    s = s.lstrip("-").strip()
                    # Remove leading numbering like "1.", "2)", "3 -", etc.
                    s = re.sub(r"^[0-9]+\s*[\.\-\):]*\s*", "", s)
                    if s:
                        cleaned_facts.append(s)
                sample[key] = cleaned_facts
    return data


class Patient:
    def __init__(self, args, sample):
        self.args = args
        # Assuming 'context' is a list or a long string of historical or background information
        if isinstance(sample['context'], list) and len(sample['context']) > 0:
            if 'initial_info' in sample: self.initial_info = sample['initial_info']
            else: self.initial_info = sample['context'][0]  # Taking the first item if it's a list
            self.context_list = sample['context']
            self.context_para = " ".join(sample['context'])
        elif isinstance(sample['context'], str):
            # Assuming sentences are separated by periods, taking the first sentence
            if 'initial_info' in sample: self.initial_info = sample['initial_info']
            else: self.initial_info = sample['context'].split(". ")[0]
            temp = sample['context'].split(". ")
            self.context_list = [temp[i]+'.' if i!=len(temp)-1 and not temp[i].endswith('.') else temp[i] for i in range(len(temp))]
            self.context_para = sample['context']
        else:
            if 'initial_info' in sample: self.initial_info = sample['initial_info']
            else: self.initial_info = ""  # Default fallback
            self.context_list = []
            self.context_para = 'None'
        
        self.model_name = MEDIQ_PATIENT_MODEL
        self.history = []  # To track the interaction history of questions and answers
        # To store atomic facts after initial processing, you can choose to
        # store this somewhere locally to avoid repeated processing
        self.facts = sample.get('facts') or sample.get('atomic_facts')

        # Use the same token budget as the expert unless explicitly overridden
        # in respond() (e.g., for long fact extraction).
        self.max_length = args.max_tokens

    def update_state(self, question, answer):
        # Update the internal history with the new question and the corresponding answer
        self.history.append({"question": question, "answer": answer})

    def get_state(self):
        # Return the initial context and the history of interactions
        return {
            "initial_info": self.initial_info,
            "interaction_history": self.history
        }
    
    def get_questions(self):
        # Return the list of questions asked so far
        return [qa["question"] for qa in self.history]
    
    def get_answers(self):
        # Return the list of answers provided so far
        return [qa["answer"] for qa in self.history]
    
    def get_response(self, messages, max_length=None):
        """
        Call the MedIQ LLM for the patient.

        max_length controls the completion token budget for this call.
        If not provided, falls back to self.max_length, which is tied to
        args.max_tokens.
        """
        if max_length is None:
            max_length = self.max_length

        # Important: ModelCacheHTTP expects `max_tokens`, not `max_length`.
        # We also forward temperature / top_p / top_logprobs so that patient
        # and expert use consistent generation settings.
        return get_response(
            messages,
            self.model_name,
            max_tokens=max_length,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_logprobs=self.args.top_logprobs,
        )
    
    def respond(self, question):
        raise NotImplementedError


class FactSelectPatient(Patient):
    def respond(self, question):
        if not self.facts:
            system_prompt = "You are a truthful medical assistant that understands the patient's information."
            user_prompt = f"Break the following patient information into a list of independent atomic facts, with one piece of information in each statement. Each fact should only include the smallest unit of information, but should be self-contained.\n\"{self.context_para}\"\nResponse with the list of atomic facts and nothing else. Write each fact on a new line starting with a dash (-). No numbered lists allowed."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response_text = self.get_response(messages, max_length=self.max_length)
            response_text = [s.strip().lstrip('- ').strip() for s in response_text.splitlines() if s.strip()]
            self.facts = response_text
        
        facts_prompt = "\n".join(self.facts)
        system_prompt = "You are a truthful medical assistant that understands the patient's information, and you are trying to answer questions from a medical doctor about the patient given a list of factual statements describing the patient. Please return the facts that answer the doctor's question verbatim without any additional information. Do not prefix the answer with numbers. If none of the facts answer the question, simply say \"The patient cannot answer this question, please do not ask this question again.\""
        prompt = f"List of facts:\n{facts_prompt}\n\nDoctor's question: \"{question}\"\n\nStatements that answer the question:"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = self.get_response(messages)
        
        self.update_state(question, response)
        return response


def run_patient_interaction(expert_class, patient_class, args, sample):
    expert_system = expert_class(args, sample["question"], sample["options"])
    patient_system = patient_class(args, sample)  # Assuming the patient_system is initialized with the sample which includes necessary context
    temp_additional_info = []  # To store optional data like confidence scores

    while len(patient_system.get_questions()) < args.max_questions:
        patient_state = patient_system.get_state()
        response_dict = expert_system.respond(patient_state)
        
        # Optional return values for analysis, e.g., confidence score, logprobs
        temp_additional_info.append({k: v for k, v in response_dict.items() if k not in ["type", "letter_choice", "question"]})

        if response_dict["type"] == "question":
            # Patient generates an answer based on the last question asked, and add to memory
            patient_response = patient_system.respond(response_dict["question"])
            log_info(f"[Patient System]: {patient_response}")

        elif response_dict["type"] == "choice":
            expert_decision = response_dict["letter_choice"]
            sample_info = {
                "initial_info": patient_system.initial_info,
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "question": sample["question"],
                "options": sample["options"],
                "context": sample["context"],
                "facts": patient_system.facts, # if the FactSelectPatient patient module is used, this will store the atomic facts the patient used to answer questions for reproducibility
            }
            return expert_decision, patient_system.get_questions(), patient_system.get_answers(), temp_additional_info, sample_info
        
        else:
            raise ValueError("Invalid response type from expert_system.")
    
    # If max questions are reached and no final decision has been made
    log_info(f"==================== Max Interaction Length ({args.max_questions} turns) Reached --> Force Final Answer ====================")
    patient_state = patient_system.get_state()
    log_info("[Force final choice] requesting final letter from model (one call).")
    stuck_response = expert_system.force_final_choice(patient_state)

    sample_info = {
        "initial_info": patient_system.initial_info,
        "correct_answer": sample["answer"],
        "correct_answer_idx": sample["answer_idx"],
        "question": sample["question"],
        "options": sample["options"],
        "context": sample["context"],
        "facts": patient_system.facts,  # if the FactSelectPatient patient module is used, this will store the atomic facts the patient used to answer questions for reproducibility
    }

    return (
        stuck_response,
        patient_system.get_questions(),
        patient_system.get_answers(),
        temp_additional_info,
        sample_info,
    )

class Expert:
    """
    Expert system skeleton
    """
    def __init__(self, args, inquiry, options):
        # Initialize the expert with necessary parameters and the initial context or inquiry
        self.args = args
        self.inquiry = inquiry
        self.options = options

    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        raise NotImplementedError

    def force_final_choice(self, patient_state):
        """When max_questions reached and letter_choice is None, get a final letter in one extra call."""
        raise NotImplementedError

    def ask_question(self, patient_state, prev_messages, task_prompt, question_generation_function):
        # Generate a question based on the current patient state
        kwargs = {
            "patient_state": patient_state,
            "task_prompt": task_prompt,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "messages": prev_messages,
            "model_name": MEDIQ_EXPERT_MODEL_QG,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
        }
        return question_generation_function(**kwargs)
    
    def get_abstain_kwargs(self, patient_state):
        kwargs = {
            "max_depth": self.args.max_questions,
            "patient_state": patient_state,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "abstain_threshold": self.args.abstain_threshold,
            "self_consistency": self.args.self_consistency,
            "model_name": MEDIQ_EXPERT_MODEL,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
        }
        return kwargs


def parse_choice(response_text, options_dict):
    if response_text.strip() in ["A", "B", "C", "D"]:
        return response_text.strip()
    for response_line in response_text.split("\n"):
        for op_letter, op_text in options_dict.items():
            if op_text.lower() in response_line.lower():
                log_info(f"....Found {op_text} in response line: {response_line}")
                return op_letter
        for op_letter in options_dict.keys():
            if op_letter in [token for token in re.sub(r"[,.;@#()?!'/&:$]+\ *", " ", response_line).split(' ')]:
                log_info(f"....Found {op_letter} in response line: {response_line}")
                return op_letter
    log_info("can't parse choice: {}".format(response_text), type="error")
    return None


def expert_response_choice(messages, options_dict, **kwargs):
    """
    Get intermediate answer choice regardless of abstention decision
    """
    log_info(f"++++++++++++++++++++ Start of Multiple Chocie Decision [py:expert_response_choice()] ++++++++++++++++++++")
    log_info(f"[<CHOICE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")
    response_text = get_response(messages, **kwargs)
    if not response_text: 
        log_info("[<CHOICE LM RES>]: " + "No response.")
        return "No response.", None
    log_info("[<CHOICE LM RES>]: " + response_text)

    letter_choice = parse_choice(response_text, options_dict)
    if letter_choice:
        log_info("[<CHOICE PARSED>]: " + letter_choice)
    else:
        log_info("[<CHOICE PARSED>]: " + "FAILED TO PARSE.")
    
    return response_text, letter_choice


def parse_confidence_score(response_text):
    # parse the probability
    float_regex = re.compile(r'\d+\.\d+')
    scores = re.findall(float_regex, response_text)

    if len(scores) == 0:
        log_info("can't parse confidence score - answer: {}".format(response_text), type="error")
        score = round(0.2 + (random.random() - random.random()) * 0.2, 4)
        return score
    
    prob = float(scores[-1])
    if len(scores) > 1: logging.warning("more than one confidence score - using last: {}".format(response_text))
    if prob > 1: logging.warning("confidence score > 1: {}".format(response_text))
    return prob 


def parse_yes_no(response_text):
    temp_processed_response = response_text.lower().replace('.','').replace(',','').replace(';','').replace(':','').split("DECISION:")[-1].strip()
    yes_answer = "yes" in temp_processed_response
    no_answer = "no" in temp_processed_response
    if yes_answer == no_answer:
        yes_choice = "NO"
        log_info("can't parse yes/no abstain answer: {}".format(response_text), type="error")
    if yes_answer: yes_choice = "YES"
    elif no_answer: yes_choice = "NO"
    return yes_choice


def parse_atomic_question(response_text):
    questions = []
    for line in response_text.split("\n"):
        if '?' in line:
            questions.append(line.split(":")[-1].strip())
        
    if len(questions) == 0:
        log_info("can't find question in answer: {}".format(response_text), type="error")
        return None
            
    atomic_question = questions[-1].replace("'", "").replace('"', "").strip()
    return atomic_question


def expert_response_question(messages, **kwargs):
    """
    Get follow-up question
    """
    log_info(f"++++++++++++++++++++ Start of Question Generator [py:expert_response_question()] ++++++++++++++++++++")
    log_info(f"[<QUESTION GENERATOR PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")
    response_text = get_response(messages, **kwargs)
    if not response_text: 
        log_info("[<QUESTION GENERATOR LM RES>]: " + "No response.")
        return "No response.", None
    log_info("[<QUESTION GENERATOR LM RES>]: " + response_text)

    atomic_question = parse_atomic_question(response_text)
    if atomic_question:
        log_info("[<QUESTION GENERATOR PARSED>]: " + atomic_question)
    else:
        log_info("[<QUESTION GENERATOR PARSED>]: " + "FAILED TO PARSE.")
    
    return response_text, atomic_question


def question_generation(messages, task_prompt, **kwargs):
    messages.append({"role": "user", "content": task_prompt})

    response_text, atomic_question = expert_response_question(messages, **kwargs)
    log_info_expert(f"[ATOMIC QUESTION PROMPT]: {messages}")
    log_info_expert(f"[ATOMIC QUESTION RESPONSE]: {atomic_question}\n")
    messages.append({"role": "assistant", "content": atomic_question})

    log_info_expert(f"[ATOMIC QUESTION RETURN]: {atomic_question}\n")
    return {
        "atomic_question": atomic_question,
        "messages": messages,
    }


class Args:
    def __init__(self, program_params):
        # Benchmark settings
        self.max_questions = int(program_params["max_questions"])
        self.self_consistency = int(program_params["self_consistency"])
        
        # Abstention settings
        # If abstain_threshold is None, each expert uses its own internal default
        # (e.g. PROB_THRESHOLD in numerical_cutoff, SCALE_THRESHOLD in scale_abstain).
        # self.self_consistency = 1
        self.abstain_threshold = None
        
        # Model inference settings
        self.temperature = 0.6
        self.top_p = 0.9
        self.max_tokens = 256
        self.top_logprobs = 0


def run_mediq(expert_class, seed=None, program_params=None) -> Tuple[List, List[str], List[int], List[str], Dict[str, Dict[str, int]]]:
    if not isinstance(program_params, dict):
        raise ValueError("PROGRAM_PARAMS must be a dict")
    for name, (min_v, max_v) in PROGRAM_PARAM_RANGES.items():
        val = int(program_params.get(name, -1))
        if val < min_v or val > max_v:
            raise ValueError(f"PROGRAM_PARAMS['{name}']={val} out of range [{min_v}, {max_v}]")

    data_path = DEFAULT_DATA_PATH
    max_cases = MAX_CASES
    
    args = Args(program_params=program_params)
    patient_data = load_data(data_path)

    batch_indices: Optional[List[int]] = None
    composer: Optional[BatchComposer] = None

    if max_cases and max_cases < len(patient_data):
        if seed is not None and BATCH_MODE == "composed":
            composer = BatchComposer(len(patient_data))
            batch_indices = composer.compose_batch(seed)
            patient_data = [patient_data[i] for i in batch_indices]
        elif seed is not None:
            rng = random.Random(seed)
            patient_data = rng.sample(patient_data, k=max_cases)
        else:
            patient_data = random.sample(patient_data, k=max_cases)
    
    dialogues = []
    diagnoses = []
    case_ids = []
    ground_truth = []
    all_expert_responses: List[List[str]] = []
    for i, sample in enumerate(patient_data):
        _expert_responses.clear()
        letter_choice, questions, answers, temp_additional_info, sample_info = run_patient_interaction(
            expert_class=expert_class,
            patient_class=FactSelectPatient,
            args=args,
            sample=sample
        )
        
        dialogue = []
        for q, a in zip(questions, answers):
            dialogue.append(("doctor", q))
            dialogue.append(("patient", a))
        dialogue.append(("doctor", f"Final answer: {letter_choice}"))
        
        dialogues.append(dialogue)
        diagnoses.append(letter_choice)
        case_ids.append(i)
        ground_truth.append(sample["answer_idx"])
        all_expert_responses.append(list(_expert_responses))

    if composer is not None and batch_indices is not None and seed is not None:
        correct_mask = [d == gt for d, gt in zip(diagnoses, ground_truth)]
        composer.update_results(batch_indices, correct_mask, seed)

    run_metadata = {
        "program_params": {"max_questions": args.max_questions, "self_consistency": args.self_consistency},
        "expert_responses": all_expert_responses,
    }
    return dialogues, diagnoses, case_ids, ground_truth, run_metadata


##### SELF-CONSISTENCY #####
### BINARY ###
# def expert_response_yes_no(messages, **kwargs):
#     """
#     Binary Abstain
#     """
#     log_info(f"++++++++++++++++++++ Start of YES/NO Decision [py:expert_response_yes_no()] ++++++++++++++++++++")
#     log_info(f"[<YES/NO PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

#     yes_no_responses, response_texts = [], {}
#     for i in range(1):
#         log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
#         response_text = get_response(messages, **kwargs)
#         if not response_text: 
#             log_info("[<YES/NO LM RES>]: " + "No response.")
#         log_info("[<YES/NO LM RES>]: " + response_text)

#         yes_choice = parse_yes_no(response_text)
#         log_info("[<YES/NO PARSED>]: " + yes_choice)
#         yes_no_responses.append(yes_choice)
#         response_texts[yes_choice] = response_text
    
#     if yes_no_responses.count("YES") > yes_no_responses.count("NO"):
#         yes_choice = "YES"
#     else:
#         yes_choice = "NO"
#     confidence = yes_no_responses.count("YES") / len(yes_no_responses)
#     log_info(f"[<YES/NO RETURN>]: yes_choice: {yes_choice}, confidence: {confidence}")
#     return response_texts[yes_choice], yes_choice, confidence

### NO SELF-CONSISTENCY ###
# def expert_response_yes_no(messages, **kwargs):
#     """
#     Binary Abstain
#     """
#     log_info(f"++++++++++++++++++++ Start of YES/NO Decision [py:expert_response_yes_no()] ++++++++++++++++++++")
#     log_info(f"[<YES/NO PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

#     response_text = get_response(messages, **kwargs)
#     if not response_text:
#         log_info("[<YES/NO LM RES>]: " + "No response.")
#     log_info("[<YES/NO LM RES>]: " + response_text)

#     yes_choice = parse_yes_no(response_text)
#     log_info("[<YES/NO PARSED>]: " + yes_choice)
#     confidence = 1.0 if yes_choice == "YES" else 0.0
#     log_info(f"[<YES/NO RETURN>]: yes_choice: {yes_choice}, confidence: {confidence}")
#     return response_text, yes_choice, confidence


### IMPLICIT ###
# def expert_response_choice_or_question(messages, options_dict, **kwargs):
#     """
#     Implicit Abstain
#     """
#     log_info(f"++++++++++++++++++++ Start of Implicit Abstention [py:expert_response_choice_or_question()] ++++++++++++++++++++")
#     log_info(f"[<IMPLICIT ABSTAIN PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")
#     answers, questions, response_texts = [], [], {}
#     for i in range(3):
#         log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
#         response_text = get_response(messages, **kwargs)
#         if not response_text: 
#             log_info("[<IMPLICIT ABSTAIN LM RES>]: " + "No response --> Re-prompt")
#             continue
#         log_info("[<IMPLICIT ABSTAIN LM RES>]: " + response_text)
#         response_text = response_text.replace("Confident --> Answer: ", "").replace("Not confident --> Doctor Question: ", "")

#         if "?" not in response_text:
#             letter_choice = parse_choice(response_text, options_dict)
#             if letter_choice:
#                 log_info("[<IMPLICIT ABSTAIN PARSED>]: " + letter_choice)
#                 answers.append(letter_choice)
#                 response_texts[letter_choice] = response_text
#         else:
#             # not a choice, parse as question
#             atomic_question = parse_atomic_question(response_text)
#             if atomic_question:
#                 log_info("[<IMPLICIT ABSTAIN PARSED>]: " + atomic_question)
#                 questions.append(atomic_question)
#                 response_texts[atomic_question] = response_text
            
#             else:
#                 log_info("[<IMPLICIT ABSTAIN PARSED>]: " + "FAILED TO PARSE --> Re-prompt")

#     if len(answers) + len(questions) == 0:
#         log_info("[<IMPLICIT ABSTAIN SC-PARSED>]: " + "No response.")
#         return "No response.", None, None, 0.0

#     conf_score = len(answers) / (len(answers) + len(questions))
#     if len(answers) > len(questions): 
#         final_answer = max(set(answers), key = answers.count)
#         response_text = response_texts[final_answer]
#         atomic_question = None
#     else:
#         final_answer = None
#         rand_id = random.choice(range(len(questions)))
#         atomic_question = questions[rand_id]
#         response_text = response_texts[atomic_question]
#     log_info(f"[<IMPLICIT ABSTAIN RETURN>]: atomic_question: {atomic_question}, final_answer: {final_answer}, conf_score: {conf_score} ([{len(answers)} : {len(questions)}])")
#     return response_text, atomic_question, final_answer, conf_score

### NO SELF-CONSISTENCY ###
# def expert_response_choice_or_question(messages, options_dict, **kwargs):
#     """
#     Implicit Abstain
#     """
#     log_info(f"++++++++++++++++++++ Start of Implicit Abstention [py:expert_response_choice_or_question()] ++++++++++++++++++++")
#     log_info(f"[<IMPLICIT ABSTAIN PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

#     response_text = get_response(messages, **kwargs)
#     if not response_text:
#         log_info("[<IMPLICIT ABSTAIN LM RES>]: " + "No response.")
#         return "No response.", None, None, 0.0
#     log_info("[<IMPLICIT ABSTAIN LM RES>]: " + response_text)
#     response_text = response_text.replace("Confident --> Answer: ", "").replace("Not confident --> Doctor Question: ", "")

#     if "?" not in response_text:
#         letter_choice = parse_choice(response_text, options_dict)
#         if letter_choice:
#             log_info("[<IMPLICIT ABSTAIN PARSED>]: " + letter_choice)
#             log_info(f"[<IMPLICIT ABSTAIN RETURN>]: atomic_question: None, final_answer: {letter_choice}, conf_score: 1.0")
#             return response_text, None, letter_choice, 1.0
#     else:
#         atomic_question = parse_atomic_question(response_text)
#         if atomic_question:
#             log_info("[<IMPLICIT ABSTAIN PARSED>]: " + atomic_question)
#             log_info(f"[<IMPLICIT ABSTAIN RETURN>]: atomic_question: {atomic_question}, final_answer: None, conf_score: 0.0")
#             return response_text, atomic_question, None, 0.0
#         log_info("[<IMPLICIT ABSTAIN PARSED>]: " + "FAILED TO PARSE")

#     log_info("[<IMPLICIT ABSTAIN RETURN>]: No valid parse.")
#     return "No response.", None, None, 0.0


### NUMERICAL ###
# def expert_response_confidence_score(messages, **kwargs):
#     """
#     Numerical Abstain
#     """
#     log_info(f"++++++++++++++++++++ Start of Numerical Confidence Score [py:expert_response_confidence_score()] ++++++++++++++++++++")
#     log_info(f"[<CONF SCORE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

#     conf_scores, response_texts = [], {}
#     for i in range(3):
#         log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
#         response_text = get_response(messages, **kwargs)
#         if not response_text: 
#             log_info("[<CONF SCORE LM RES>]: " + "No response.")
#             continue
#         log_info("[<CONF SCORE LM RES>]: " + response_text)

#         conf_score = parse_confidence_score(response_text)
#         conf_scores.append(conf_score)
#         response_texts[conf_score] = response_text
#         log_info(f"[<CONF SCORE PARSED>]: {conf_score}")
    
#     if len(conf_scores) > 0:
#         avg_conf_score = sum(conf_scores) / len(conf_scores)
#         # response_text = "CONFIDENCE SCORE: " + str(avg_conf_score)
#         temp = [abs(r - avg_conf_score) for r in conf_scores]
#         response_text = response_texts[conf_scores[temp.index(min(temp))]]
#     else:
#         avg_conf_score, response_text = 0, "No response."
#     log_info(f"[<CONF SCORE RETURN>] (average conf score): {avg_conf_score}")
#     return response_text, avg_conf_score

### NO SELF-CONSISTENCY ###
# def expert_response_confidence_score(messages, **kwargs):
#     """
#     Numerical Abstain
#     """
#     log_info(f"++++++++++++++++++++ Start of Numerical Confidence Score [py:expert_response_confidence_score()] ++++++++++++++++++++")
#     log_info(f"[<CONF SCORE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

#     response_text = get_response(messages, **kwargs)
#     if not response_text:
#         log_info("[<CONF SCORE LM RES>]: " + "No response.")
#         return "No response.", 0.0
#     log_info("[<CONF SCORE LM RES>]: " + response_text)

#     conf_score = parse_confidence_score(response_text)
#     log_info(f"[<CONF SCORE PARSED>]: {conf_score}")
#     log_info(f"[<CONF SCORE RETURN>] (conf score): {conf_score}")
#     return response_text, conf_score


### SCALE ###
# def expert_response_scale_score(messages, **kwargs):
#     """
#     Scale Abstain
#     """
#     log_info(f"++++++++++++++++++++ Start of Scale Confidence Score [py:expert_response_scale_score()] ++++++++++++++++++++")
#     log_info(f"[<SCALE SCORE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

#     conf_scores, response_texts = [], {}
#     for i in range(3):
#         log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
#         response_text = get_response(messages, **kwargs)
#         if not response_text:
#             log_info("[<SCALE SCORE LM RES>]: " + "No response.")
#             continue
#         log_info("[<SCALE SCORE LM RES>]: " + response_text)

#         conf_score = parse_likert_scale(response_text)
#         conf_scores.append(conf_score)
#         response_texts[conf_score] = response_text
#         log_info("[<SCALE SCORE PARSED>]: " + str(conf_score))
    
#     if len(conf_scores) > 0:
#         avg_conf_score = sum(conf_scores) / len(conf_scores)
#         temp = [abs(r - avg_conf_score) for r in conf_scores]
#         response_text = response_texts[conf_scores[temp.index(min(temp))]]
#     else:
#         avg_conf_score, response_text = 0, "No response."
#     log_info(f"[<SCALE SCORE RETURN>] (average conf score]): {avg_conf_score}")
#     return response_text, avg_conf_score

### NO SELF-CONSISTENCY ###
# def expert_response_scale_score(messages, **kwargs):
#     """
#     Scale Abstain
#     """
#     log_info(f"++++++++++++++++++++ Start of Scale Confidence Score [py:expert_response_scale_score()] ++++++++++++++++++++")
#     log_info(f"[<SCALE SCORE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

#     response_text = get_response(messages, **kwargs)
#     if not response_text:
#         log_info("[<SCALE SCORE LM RES>]: " + "No response.")
#         return "No response.", 0.0
#     log_info("[<SCALE SCORE LM RES>]: " + response_text)

#     conf_score = parse_likert_scale(response_text)
#     log_info("[<SCALE SCORE PARSED>]: " + str(conf_score))
#     log_info(f"[<SCALE SCORE RETURN>] (conf score): {conf_score}")
#     return response_text, conf_score