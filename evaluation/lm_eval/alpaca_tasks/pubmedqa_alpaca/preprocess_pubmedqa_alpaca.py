

alpaca_template_oneline = """Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {} ### Response:"""


def doc_to_text(doc) -> str:
    ctxs = "\n".join(doc["CONTEXTS"])
    
    instruction = "Abstract: {}\nQuestion: {}".format(
        ctxs,
        doc["QUESTION"],
    )
    
    prompt = alpaca_template_oneline.format(instruction)
    
    return prompt


def test_doc_to_text():
    doc = {
        "CONTEXTS": ["This is the first context.", "This is the second context."],
        "QUESTION": "This is the question.",
    }
    
    prompt = doc_to_text(doc)
    print("Prompt:", prompt)

if __name__ == "__main__":
    test_doc_to_text()