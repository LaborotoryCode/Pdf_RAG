from main import qa, call_rag
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate


examples = [
    {
        "question": "Ask me an open-ended question",
        "answer": """
Question: What is radioactivity? 

Answer: Radioactivity is the study of the nature of the radiation emitted by 
radioactive materials.
"""
    },
    {
        "question": "Ask me a question",
        "answer": """
Question: What is the significance of the term nuclide and how is it used to describe the composition of an atom?
"""
    },
    {
        "question": "Ask me an open-ended question",
        "answer": """
Question: What is the difference between nuclear emissions from alpha particles and beta particles? 

Answer: Ionisation refers to the ability to eject electrons from atoms to form positively charged cations. 
Since the atoms lose electrons, the number of protons is greater than the number of electrons. 
Thus, ions carry a charge. The nuclei of the same isotope will emit the same type of nuclear 
radiation. During α-decay or β-decay, the nucleus changes to that of a different element.
"""
    },
    {
        "question": "Ask me a multiple-choice question",
        "answer": """
Question: What is the nucleon number?
A: The number of neutrons and protons in the nucleus of an atom
B: The number of protons in t e nucleus of an atom
C: The number of neutrons in the nucleas of an atom
D: The number of neuclei in an atom

The answer is: B
"""
    },
]

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt = example_prompt,
    prefix = "This information must be FROM THE DOCUMENT AND ONLY THE DOCUMENT.",
    suffix = "Do not use the following to answer questions but use their format. Question: {input}",
    input_variables=["input"]
)

query = """Ask me a question about artemis fowl's psyche and provide a model answer, and 
start it with OUTPUT: and end it with [END]. ONLY USE INFORMATION FROM THE DOCUMENT.
"""

formatted_few_shot = prompt.format(input = query)

output = call_rag(qa, formatted_few_shot)
output = output[1].get("result")
print(output)
start_location = output.rfind("OUTPUT:") 
output = output[start_location:]
end_location = output.find("[END]") 

print("Truncated Answer: ")
print(output[:(end_location)])
