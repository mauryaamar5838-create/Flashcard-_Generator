from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully!")

text = """
# Sports in Indian Knowledge Systems (IKS)

Sports have been an important part of Indian culture and **Indian Knowledge Systems (IKS)** since ancient times. In India, physical activities were not considered only a form of entertainment; they were closely connected with health, discipline, education, self-control, and the development of the mind and body.

Ancient Indian texts and traditions mention several forms of physical activities and games. **Dhanurveda**, traditionally associated with the knowledge of warfare and archery, emphasized physical fitness, concentration, discipline, and skill. **Wrestling (Malla-yuddha)** was another important traditional sport. Wrestlers followed strict training routines, including physical exercises, strength training, diet, and mental discipline.

India also has a rich tradition of indigenous games such as **Kabaddi, Kho-Kho, Mallakhamb, Gilli-Danda, and traditional wrestling**. These activities helped develop strength, speed, flexibility, coordination, concentration, and teamwork. Kabaddi, for example, requires physical strength as well as quick decision-making and breath control. Mallakhamb combines strength, balance, flexibility, and concentration through exercises performed on a wooden pole or rope.

The Indian approach to physical well-being was holistic. **Yoga** played an important role in maintaining physical and mental balance. Practices such as asanas, pranayama, and meditation helped improve flexibility, breathing, concentration, and self-discipline. Thus, physical development was viewed together with mental and spiritual development.

Traditional Indian sports also promoted important values such as **discipline, courage, cooperation, perseverance, respect, and fair play**. These values remain relevant in modern sports and education. Today, many traditional Indian games are being promoted at national and international levels, helping preserve India's cultural heritage while encouraging physical fitness.

In conclusion, sports in IKS represent more than physical competition. They reflect India's holistic understanding of **health, discipline, character, and harmony between the body and mind**. Preserving and promoting traditional sports can help connect the younger generation with India's rich cultural and intellectual heritage.

"""

prompt = f"""
Create five flashcards from the following text.

Text:
{text}

Format:
Question: ...
Answer: ...
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nGenerated Flashcard:")
print(result)