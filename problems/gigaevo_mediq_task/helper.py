import json
import os
import sys
import logging
import re
from typing import List, Tuple
import random


# Add mediQ src to path
sys.path.append("/media/ssd-3t/isviridov/alphaevolve/mediQ/src")

# Default data path
DEFAULT_DATA_PATH = "/media/ssd-3t/isviridov/alphaevolve/mediQ/data/all_craft_md.jsonl"

# Models specification
MEDIQ_EXPERT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MEDIQ_EXPERT_MODEL_QG = MEDIQ_EXPERT_MODEL
MEDIQ_PATIENT_MODEL = MEDIQ_EXPERT_MODEL
# URL for MedIQ LLM (patient/doctor). Set by run.sh; default for local vLLM on 44003.
MEDIQ_VLLM_URL = os.environ.get("MEDIQ_VLLM_URL")

# Other fixed params
MAX_CASES = 50


def log_info(message, logger_name="detail_logger", print_to_std=False, type="info"):
    # if type(logger) == str and logger in logging.getLogger().manager.loggerDict:
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


class ModelCacheHTTP:
    """ModelCache that uses vLLM server via OpenAI-compatible HTTP API."""
    
    def __init__(self, model_name, base_url, **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        self.args = kwargs
        self._init_client()
    
    def _init_client(self):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="dummy-key",  # vLLM doesn't validate API key
        )
        log_info(f"[ModelCacheHTTP] Initialized client for {self.model_name} at {self.base_url}")
    
    def generate(self, messages):
        log_info(f"[{self.model_name}][INPUT]: {messages}")
        
        temperature = self.args.get("temperature", 0.6)
        max_tokens = self.args.get("max_tokens", 128)
        top_p = self.args.get("top_p", 0.9)
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
            response_text = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        except Exception as e:
            log_info(f"[{self.model_name}][ERROR]: {e}", type="error")
            raise
        
        # Trimmed response for logging to avoid huge lines
        log_info(f"[{self.model_name}][OUTPUT]: " +
                 f"{{'response_text': '{response_text[:100]}...', 'usage': {usage}}}")
        # Return only text and usage; log-probs are not used in this task.
        return response_text, usage


def get_response(messages, model_name, **kwargs):
    
    model_cache = models.get(model_name, None)
    if model_cache is None:
        log_info(f"[get_response] Initializing HTTP client for {model_name} at {MEDIQ_VLLM_URL}")
        model_cache = ModelCacheHTTP(model_name, base_url=MEDIQ_VLLM_URL, **kwargs)
        models[model_name] = model_cache
    
    return model_cache.generate(messages)


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
        self.facts = sample.get('facts') or sample.get('atomic_facts')  # To store atomic facts after initial processing, you can choose to store this somewhere locally to avoid repeated processing

        self.max_length = 50  # Maximum length of the response (different from the expert system)

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
        if max_length is None: max_length = self.max_length
        return get_response(messages, self.model_name, max_length=max_length)
    
    def respond(self, question):
        raise NotImplementedError


class FactSelectPatient(Patient):
    def respond(self, question):
        if not self.facts:
            system_prompt = "You are a truthful medical assistant that understands the patient's information."
            user_prompt = f"Break the following patient information into a list of independent atomic facts, with one piece of information in each statement. Each fact should only include the smallest unit of information, but should be self-contained.\n\"{self.context_para}\"\nResponse with the list of atomic facts and nothing else. Write each fact on a new line starting with a dash (-). No numbered lists allowed."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response_text, num_tokens = self.get_response(messages, max_length=1000)
            response_text = [s.strip().lstrip('- ').strip() for s in response_text.splitlines() if s.strip()]
            self.facts = response_text
        
        facts_prompt = "\n".join(self.facts)
        system_prompt = "You are a truthful medical assistant that understands the patient's information, and you are trying to answer questions from a medical doctor about the patient given a list of factual statements describing the patient. Please return the facts that answer the doctor's question verbatim without any additional information. Do not prefix the answer with numbers. If none of the facts answer the question, simply say \"The patient cannot answer this question, please do not ask this question again.\""
        prompt = f"List of facts:\n{facts_prompt}\n\nDoctor's question: \"{question}\"\n\nStatements that answer the question:"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response, num_tokens = self.get_response(messages)
        
        self.update_state(question, response)
        return response


def run_patient_interaction(expert_class, patient_class, args, sample):
    expert_system = expert_class(args, sample["question"], sample["options"])
    patient_system = patient_class(args, sample)  # Assuming the patient_system is initialized with the sample which includes necessary context
    temp_choice_list = []
    temp_additional_info = []  # To store optional data like confidence scores

    while len(patient_system.get_questions()) < args.max_questions:
        patient_state = patient_system.get_state()
        response_dict = expert_system.respond(patient_state)
        
        # Optional return values for analysis, e.g., confidence score, logprobs
        temp_additional_info.append({k: v for k, v in response_dict.items() if k not in ["type", "letter_choice", "question"]})

        if response_dict["type"] == "question":
            # still make the Expert generate a choice based on the current state for intermediate evaluation, log the question as an intermediate choice
            temp_choice_list.append(response_dict["letter_choice"])
            # Patient generates an answer based on the last question asked, and add to memory
            patient_response = patient_system.respond(response_dict["question"])
            log_info(f"[Patient System]: {patient_response}")

        elif response_dict["type"] == "choice":
            expert_decision = response_dict["letter_choice"]
            temp_choice_list.append(expert_decision)
            sample_info = {
                "initial_info": patient_system.initial_info,
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "question": sample["question"],
                "options": sample["options"],
                "context": sample["context"],
                "facts": patient_system.facts, # if the FactSelectPatient patient module is used, this will store the atomic facts the patient used to answer questions for reproducibility
            }
            return expert_decision, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list, temp_additional_info, sample_info
        
        else:
            raise ValueError("Invalid response type from expert_system.")
    
    # If max questions are reached and no final decision has been made
    log_info(f"==================== Max Interaction Length ({args.max_questions} turns) Reached --> Force Final Answer ====================")
    patient_state = patient_system.get_state()
    response_dict = expert_system.respond(patient_state)
    log_info(f"[Expert System]: {response_dict}")
    stuck_response = response_dict["letter_choice"]
    # Optional return values for analysis, e.g., confidence score, logprobs
    temp_additional_info.append({k: v for k, v in response_dict.items() if k != "letter_choice"})

    sample_info = {
        "initial_info": patient_system.initial_info,
        "correct_answer": sample["answer"],
        "correct_answer_idx": sample["answer_idx"],
        "question": sample["question"],
        "options": sample["options"],
        "context": sample["context"],
        "facts": patient_system.facts, # if the FactSelectPatient patient module is used, this will store the atomic facts the patient used to answer questions for reproducibility
    }
    
    return stuck_response, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list + [stuck_response], temp_additional_info, sample_info


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
            "rationale_generation": self.args.rationale_generation,
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
    response_text, num_tokens = get_response(messages, **kwargs)
    if not response_text: 
        log_info("[<CHOICE LM RES>]: " + "No response.")
        return "No response.", None, num_tokens
    log_info("[<CHOICE LM RES>]: " + response_text)

    letter_choice = parse_choice(response_text, options_dict)
    if letter_choice:
        log_info("[<CHOICE PARSED>]: " + letter_choice)
    else:
        log_info("[<CHOICE PARSED>]: " + "FAILED TO PARSE.")
    
    return response_text, letter_choice, num_tokens


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
    response_text, num_tokens = get_response(messages, **kwargs)
    if not response_text: 
        log_info("[<QUESTION GENERATOR LM RES>]: " + "No response.")
        return "No response.", None, num_tokens
    log_info("[<QUESTION GENERATOR LM RES>]: " + response_text)

    atomic_question = parse_atomic_question(response_text)
    if atomic_question:
        log_info("[<QUESTION GENERATOR PARSED>]: " + atomic_question)
    else:
        log_info("[<QUESTION GENERATOR PARSED>]: " + "FAILED TO PARSE.")
    
    return response_text, atomic_question, num_tokens


def question_generation(messages, task_prompt, **kwargs):
    messages.append({"role": "user", "content": task_prompt})

    response_text, atomic_question, num_tokens = expert_response_question(messages, **kwargs)
    log_info_expert(f"[ATOMIC QUESTION PROMPT]: {messages}")
    log_info_expert(f"[ATOMIC QUESTION RESPONSE]: {atomic_question}\n")
    messages.append({"role": "assistant", "content": atomic_question})

    log_info_expert(f"[ATOMIC QUESTION RETURN]: {atomic_question}, usage: {num_tokens}\n")
    return {
        "atomic_question": atomic_question,
        "messages": messages,
        "usage": num_tokens,
    }


class Args:
    def __init__(self):
        
        # Benchmark settings
        self.max_questions = 10
        
        # Abstention settings
        self.rationale_generation = False
        self.self_consistency = 1
        self.abstain_threshold = 0.8
        
        # Model inference settings
        self.temperature = 0.6
        self.top_p = 0.9
        self.max_tokens = 128
        self.top_logprobs = 0


def run_mediq(expert_class) -> Tuple[List, List[str], List[int], List[str]]:
    
    data_path = DEFAULT_DATA_PATH
    max_cases = MAX_CASES
    
    args = Args()
    patient_data = load_data(data_path)
    
    # Limit cases if needed
    if max_cases and max_cases < len(patient_data):
        patient_data = patient_data[:max_cases]
    
    dialogues = []
    diagnoses = []
    case_ids = []
    ground_truth = []  # Collect correct answers directly from data
    
    for i, sample in enumerate(patient_data):
        # Run interaction exactly like mediQ_benchmark.py
        letter_choice, questions, answers, temp_choice_list, temp_additional_info, sample_info = run_patient_interaction(
            expert_class=expert_class,
            patient_class=FactSelectPatient,
            args=args,
            sample=sample
        )
        
        # Build dialogue in format for validate(): List[(name, replic)]
        dialogue = []
        for q, a in zip(questions, answers):
            dialogue.append(("doctor", q))
            dialogue.append(("patient", a))
        dialogue.append(("doctor", f"Final answer: {letter_choice}"))
        
        dialogues.append(dialogue)
        diagnoses.append(letter_choice)
        case_ids.append(i)
        ground_truth.append(sample["answer_idx"])  # Store correct answer
    
    return dialogues, diagnoses, case_ids, ground_truth