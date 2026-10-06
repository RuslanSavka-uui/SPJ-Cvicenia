import re

current_tokens = []
sentence_text = ""
reference_text = ""
skip_count = 0
all_sentences = []

with open("sk_snk-ud-dev.conllu", encoding="utf-8") as file:
    for line in file:
        if line.startswith("#"):
            if line.startswith("# text = "):
                reference_text = line[len("# text = "):].strip()
            else:
                continue
        elif line.strip() == "":
            for token in current_tokens:
                if "SpaceAfter=No" in token[1]:
                    sentence_text += token[0]
                else:
                    sentence_text += token[0] + " "
            sentence_text = sentence_text.strip()
            if sentence_text != reference_text:
                print("MISMATCH (reconstruction):")
                print(f"got: {sentence_text}")
                print(f"expected: {reference_text}")
            gold_token = []
            for form, misc in current_tokens:
                gold_token.append(form)
            all_sentences.append((sentence_text, gold_token))
            sentence_text = ""
            current_tokens = []
        else:
            columns = line.split("\t")
            if skip_count > 0:
                skip_count -= 1
                continue
            if "-" in columns[0]:
                form = columns[1]
                misc = columns[-1]
                current_tokens.append((form, misc))
                skip_count = (int(columns[0].split("-")[1]) - int(columns[0].split("-")[0])) + 1
                continue
            if "." in columns[0]:
                continue
            form = columns[1]
            misc = columns[-1]
            current_tokens.append((form, misc))

with open("output.txt", "w", encoding="utf-8") as file:
    for sentence, tokens in all_sentences:
        file.write(f"{sentence}\t{tokens}\n")

# Tokenizer
pattern = r"\w+|[^\w\s]"

total_tokens = 0          
matched_tokens = 0        
mismatched_tokens = 0     
fully_matched_sentences = 0    
mismatched_sentences = 0       

for sentence_text, gold_token in all_sentences:
    tokens = re.findall(pattern, sentence_text)

    total_tokens += len(gold_token)

    if tokens == gold_token:
        matched_tokens += len(gold_token)
        fully_matched_sentences += 1
    else:
        mismatched_sentences += 1
        min_len = min(len(tokens), len(gold_token))
        for i in range(min_len):
            if tokens[i] == gold_token[i]:
                matched_tokens += 1
            else:
                mismatched_tokens += 1
        mismatched_tokens += abs(len(tokens) - len(gold_token))

accuracy = matched_tokens / total_tokens * 100

print()
print("Results")
print(f"Total tokens: {total_tokens}")
print(f"Matched tokens: {matched_tokens}")
print(f"Mismatched tokens: {mismatched_tokens}")
print(f"Fully matched sentences: {fully_matched_sentences}")
print(f"Mismatched sentences: {mismatched_sentences}")
print(f"Accuracy: {accuracy:.2f}%")
