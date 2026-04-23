import os
import json
import math
import random
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

descriptions_folder = "style_descriptions"
labels_file = "style_labels.json"
output_file = "style_predictions.json"

START_FROM_FILE = None  # Set to file_id to resume


def load_labels():
    with open(labels_file, "r") as f:
        return json.load(f)


def load_predictions():
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            return json.load(f)
    return {}


def save_predictions(preds):
    with open(output_file, "w") as f:
        json.dump(preds, f, indent=2)


def build_examples(labels, descriptions_folder, example_fids):
    """Build example strings from 15% of each group."""
    examples = []
    for group_name, entries in labels.items():
        group_example_fids = [fid for fid in example_fids if fid in entries]
        for fid in group_example_fids:
            desc_path = os.path.join(descriptions_folder, fid + ".txt")
            if os.path.exists(desc_path):
                with open(desc_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                label = entries[fid]
                examples.append(f'"{text}" -> [{group_name}, {label}]')
    return "\n".join(examples)


def select_example_fids(labels):
    """Select 15% (rounded up) of files from each group as examples."""
    example_fids = set()
    for group_name, entries in labels.items():
        fids = list(entries.keys())
        n = math.ceil(len(fids) * 0.15)
        selected = random.sample(fids, min(n, len(fids)))
        example_fids.update(selected)
    return example_fids


def process_file(text_content, examples):
    """Send text to API, get top 3 verb/noun groups + specificity."""
    prompt = f"""
Given a text which predominantly describes a lower-body (referred to as gait) motion using human body parts movement, your goal is to determine what this motion would be semantically described as i.e. give it a verb/noun group which describes its gait locomotion.
Pick top 3 verb/noun groups that best describe it. Also identify the specifity that makes that motion different from a default verb/noun group that you chose.

Rules:
- Lowercase
- Output should be like [verb1/noun1, verb2/noun2, verb3/noun3, specifity]. The specifity can be a word or phrase [does not matter what grammatic structure (i.e. noun, verb, adj, etc..) as long as it distinguishes the motion from its group]. But use basic forms ex. lemma for verb
- Verbs are usually preferred over nouns unless nouns better describe the gait locomotion.
- Follow formats of specifity extraction, which provides things to look out for that will lead to the output... Look at examples of each format for reasoning.
- In some cases a format that has been found is not suited for our situation... They have been listed in 'Formats to remove" and in this case, you should return REMOVE
- In cases of multiple formats present, the most dominant one should be followed through.
- Note that that the specifity extraction is being done for lower-body distinctiveness, not upper-body.

Formats of specifity extraction:
- Gait modifier (general)
- Style-induced gait (should be a significant style, not something casual) (more implied than general)
- Limb specific bias
- Stability
- Gait intensity

Examples of formats of specificity extraction:
- Gait modifier: if a text describes salsa -> salsa
- Gait modifier: if a text describes jumping jacks -> jumping jacks
- Style-induced gait: if a text describes a sway -> sway
- Style-induced gait: if a text describes a dramatic walk -> dramatic
- Style-induced gait: if a text describes a cautious walk -> cautious
- Style-induced gait: if a text describes a brisk walk -> brisk
- Limb specific bias: if a text describes kicking with left leg, do not pick 'left leg' as kicking already implies that
- Limb specific bias: if a text describes stepping forward with left leg, do not pick 'left leg' as walking already implies that 
- Limb specific bias: if a text describes limping while holding right knee-> hold right knee
- Limb specific bias: if a text describes hopping on one foot -> one foot
- Limb specific bias: if a text describes walking while dragging feet -> foot drop
- Limb specific bias: if a text describes jumping on toes-> toes
- Stability: if a text describes losing balance -> lose balance
- Gait intensity: if a text describes a large jump -> large
- Gait intensity: if a text describes a long gait-> long gait

Formats to not use for specifity extraction:
- Gait pace (something like fast or slow would already be differentiated by for example, sprint group vs jog group)
- Gait trajectory
- Environmental constraint (focus is on flat ground)

Examples: you are expected to give three verb/noun groups and 1 specifity but in examples, we only give 1 verb/noun group and 1 specifity
{examples}

Text:
{text_content}
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text.strip().lower()


# --- Main loop ---

labels = load_labels()
predictions = load_predictions()

# Select 15% examples from each group
example_fids = select_example_fids(labels)

# All files in style_labels
all_fids = set()
for entries in labels.values():
    all_fids.update(entries.keys())

# Files to process = all files minus examples (examples are used as few-shot, not queried)
process_fids = sorted(all_fids - example_fids)

# Build examples string once
examples_str = build_examples(labels, descriptions_folder, example_fids)

api_call_count = 0
started = START_FROM_FILE is None

for file_id in process_fids:
    if not started:
        if file_id == START_FROM_FILE:
            started = True
        else:
            continue

    # Skip if already predicted
    if file_id in predictions:
        continue

    desc_path = os.path.join(descriptions_folder, file_id + ".txt")
    if not os.path.exists(desc_path):
        print(f"[SKIP] {file_id} - no description file")
        continue

    with open(desc_path, "r", encoding="utf-8") as f:
        text_content = f.read().strip()

    try:
        output = process_file(text_content, examples_str)
        api_call_count += 1

        predictions[file_id] = output
        save_predictions(predictions)
        print(f"[{api_call_count}] {file_id}: {output}")

    except Exception as e:
        print(f"[ERROR] {file_id}: {e}")
        continue

print(f"\nDone. API calls made: {api_call_count}")
print(f"Total predictions: {len(predictions)}")
