alpaca_template_oneline = """Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {} ### Response:"""

def doc_to_text(doc) -> str:
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    
    instruction = f"Question: {doc['sent1']}\n{answers}" 
    prompt = alpaca_template_oneline.format(instruction) 
    
    return prompt

def doc_to_target(doc) -> int:
    return doc["label"]



def test_doc_to_text():
    doc = {
        "sent1": "This is a question",
        "ending0": "Option 1",
        "ending1": "Option 2",
        "ending2": "Option 3",
        "ending3": "Option 4",
    }
    
    prompt = doc_to_text(doc)
    print("Prompt:", prompt)
    
if __name__ == "__main__":
    test_doc_to_text()
    