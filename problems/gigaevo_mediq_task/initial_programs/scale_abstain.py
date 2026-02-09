from helper import log_info, log_info_expert, expert_response_choice, Expert, question_generation, get_response, run_mediq
from typing import List, Tuple


SCALE_THRESHOLD = 4.0


expert_system = {
    "system_msg": "You are a medical doctor trying to reason through a real-life clinical case. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, respond according to the task specified by the user. Base your response on the current and standard practices referenced in medical guidelines.",

    "question_word": "Doctor Question",
    "answer_word": "Patient Response",

    "abstention_prompt": """Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. How confident are you to pick the correct option to the problem factually using the conversation log? Choose between the following ratings:
"Very Confident" - The correct option is supported by all evidence, and there is enough evidence to eliminate the rest of the answers, so the option can be confirmed conclusively.
"Somewhat Confident" - I have reasonably enough information to tell that the correct option is more likely than other options, more information is helpful to make a conclusive decision.
"Neither Confident or Unconfident" - There are evident supporting the correct option, but further evidence is needed to be sure which one is the correct option.
"Somewhat Unconfident" - There are evidence supporting more than one options, therefore more questions are needed to further distinguish the options.
"Very Unconfident" - There are not enough evidence supporting any of the options, the likelihood of picking the correct option at this point is near random guessing.\n\nThink carefully step by step, respond with the chosen confidence rating ONLY and NOTHING ELSE.""",

    "abstention_prompt_RG": """Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. How confident are you to pick the correct option to the problem factually using the conversation log? Choose between the following ratings:
"Very Confident" - The correct option is supported by all evidence, and there is enough evidence to eliminate the rest of the answers, so the option can be confirmed conclusively.
"Somewhat Confident" - I have reasonably enough information to tell that the correct option is more likely than other options, more information is helpful to make a conclusive decision.
"Neither Confident or Unconfident" - There are evident supporting the correct option, but further evidence is needed to be sure which one is the correct option.
"Somewhat Unconfident" - There are evidence supporting more than one options, therefore more questions are needed to further distinguish the options.
"Very Unconfident" - There are not enough evidence supporting any of the options, the likelihood of picking the correct option at this point is near random guessing.\n\nAnswer in the following format:\nREASON: a one-sentence explanation of why you are or are not confident and what other information is needed.\nDECISION: chosen rating from the above list.""",
    
    "question_prompt": "If there are missing features that prevent you from picking a confident and factual answer to the inquiry, consider which features are not yet asked about in the conversation log; then, consider which missing feature is the most important to ask the patient in order to provide the most helpful information toward a correct medical decision. You can ask about any relevant information about the patient’s case, such as family history, tests and exams results, treatments already done, etc. Consider what are the common questions asked in the specific subject relating to the patient’s known symptoms, and what the best and most intuitive doctor would ask. Ask ONE SPECIFIC ATOMIC QUESTION to address this feature. The question should be bite-sized, and NOT ask for too much at once. Make sure to NOT repeat any questions from the above conversation log. Answer in the following format:\nATOMIC QUESTION: the atomic question and NOTHING ELSE.\nATOMIC QUESTION: ",

    "answer": "Assume that you already have enough information from the above question-answer pairs to answer the patient inquiry, use the above information to produce a factual conclusion. Respond with the correct letter choice (A, B, C, or D) and NOTHING ELSE.\nLETTER CHOICE: ",

    "curr_template": """A patient comes into the clinic presenting with a symptom as described in the conversation log below:
    
PATIENT INFORMATION: {}
CONVERSATION LOG:
{}
QUESTION: {}
OPTIONS: {}
YOUR TASK: {}"""
    
}


def parse_likert_scale(response_text):
    temp_processed_response = response_text.lower().replace('.','').replace(',','').replace(';','').replace(':','')
    if "very confident" in temp_processed_response:
        conf_score = 5
    elif "somewhat confident" in temp_processed_response:
        conf_score = 4
    elif "neither confident nor unconfident" in temp_processed_response:
        conf_score = 3
    elif "neither confident or unconfident" in temp_processed_response:
        conf_score = 3
    elif "somewhat unconfident" in temp_processed_response:
        conf_score = 2
    elif "very unconfident" in temp_processed_response:
        conf_score = 1
    else:
        conf_score = 0
        log_info("can't parse likert confidence score: {}".format(response_text), type="error")
    return conf_score


def expert_response_scale_score(messages, self_consistency=1, **kwargs):
    """
    Scale Abstain
    """
    log_info(f"++++++++++++++++++++ Start of Scale Confidence Score [py:expert_response_scale_score()] ++++++++++++++++++++")
    log_info(f"[<SCALE SCORE PROMPT>] [len(messages)={len(messages)}] (messages[-1]):\n{messages[-1]['content']}")

    conf_scores, response_texts = [], {}
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    for i in range(self_consistency):
        log_info(f"-------------------- Self-Consistency Iteration {i+1} --------------------")
        response_text, num_tokens = get_response(messages, **kwargs)
        total_tokens["input_tokens"] += num_tokens["input_tokens"]
        total_tokens["output_tokens"] += num_tokens["output_tokens"]
        if not response_text:
            log_info("[<SCALE SCORE LM RES>]: " + "No response.")
            continue
        log_info("[<SCALE SCORE LM RES>]: " + response_text)

        conf_score = parse_likert_scale(response_text)
        conf_scores.append(conf_score)
        response_texts[conf_score] = response_text
        log_info("[<SCALE SCORE PARSED>]: " + str(conf_score))
    
    if len(conf_scores) > 0:
        avg_conf_score = sum(conf_scores) / len(conf_scores)
        temp = [abs(r - avg_conf_score) for r in conf_scores]
        response_text = response_texts[conf_scores[temp.index(min(temp))]]
    else:
        avg_conf_score, response_text = 0, "No response."
    log_info(f"[<SCALE SCORE RETURN>] (average conf score]): {avg_conf_score}")
    return response_text, avg_conf_score, total_tokens


def scale_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, abstain_threshold, **kwargs):
    """
    Likert abstention strategy based on the current patient state.
    This function prompts the model to produce a likert scale confidence score of how confident it is in its decision, then decide abstention based on a cutoff
    """
    if not abstain_threshold: abstain_threshold = SCALE_THRESHOLD

    # Get the response from the expert system
    prompt_key = "abstention_prompt_RG" if rationale_generation else "abstention_prompt"
    abstain_task_prompt = expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{expert_system['question_word']}: {qa['question']}\n{expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": expert_system["system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, conf_score, num_tokens = expert_response_scale_score(
        messages, abstain_threshold=abstain_threshold, **kwargs
    )
    abstain_decision = conf_score < abstain_threshold
    log_info_expert(f"[ABSTENTION PROMPT]: {messages}")
    log_info_expert(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    prompt_answer = expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": expert_system["system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens_answer = expert_response_choice(messages_answer, options_dict, **kwargs)
    num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info_expert(f"[SCALE ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
    }


class ScaleExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = scale_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"], expert_system["question_prompt"], question_generation)
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


def entrypoint() -> Tuple[List, List[str], List[int], List[str]]:
    return run_mediq(ScaleExpert)