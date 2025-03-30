# Copied from Master
alpaca_template_oneline = """Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {} ### Response:"""

def doc_to_text(doc) -> str:
    """
    Question: <question>
    Choices:
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    Answer:
    """
    choices = [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]
    option_choices = {
        "A": choices[0],
        "B": choices[1],
        "C": choices[2],
        "D": choices[3],
    }

    instruction = "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in option_choices.items():
        instruction += f"{choice.upper()}. {option}\n"
    
    prompt = alpaca_template_oneline.format(instruction)
    return prompt

def test_doc_to_text():
    doc = {
        "question": "What is the capital of France?",
        "opa": "Paris",
        "opb": "London",
        "opc": "Berlin",
        "opd": "Madrid",
    }
    
    print(doc_to_text(doc))
    
    
if __name__ == "__main__":
    test_doc_to_text()