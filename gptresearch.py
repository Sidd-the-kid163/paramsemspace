import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

motion_folder = "style_motions"
texts_folder = "style_texts"
labels_file = "style_labels.json"

START_FROM_GROUP = None
ONLY_GROUPS = ["backpedal", "run", "jog", "spin"]  # Only process these groups


def load_labels():
    with open(labels_file, "r") as f:
        return json.load(f)


def save_labels(labels):
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)


def deduplicate_group(group_name, group_data):
    """Send group's {file: label} to API, get back cleaned version."""
    group_json = json.dumps(group_data, indent=2)

    prompt = f"""
Given the json structure, go through the labels of each file number and remove any that are analogous to or synonymous with one or more of the other labels.
Do not remove all the analogous/synonymous labels, leave one. Return the same modified json structure.

The focus of the labels is on lower-body motion. You are not to aggressively remove content and only if they are the same by definition.

{group_json}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    raw = response.output_text.strip()

    # Extract JSON from response (may have markdown fences or trailing text)
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        raw = raw[start:end]
    elif "{" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        raw = raw[start:end]

    return json.loads(raw)


# --- Main loop ---

labels = load_labels()
group_names = list(labels.keys())
api_call_count = 0
started = START_FROM_GROUP is None

for group_name in group_names:
    if ONLY_GROUPS and group_name not in ONLY_GROUPS:
        continue

    if not started:
        if group_name == START_FROM_GROUP:
            started = True
        else:
            continue

    labels = load_labels()
    if group_name not in labels:
        continue

    group_data = labels[group_name]
    before_count = len(group_data)

    if before_count <= 1:
        print(f"[SKIP] {group_name}: only {before_count} file(s)")
        continue

    try:
        cleaned = deduplicate_group(group_name, group_data)
        api_call_count += 1

        # Find removed files
        removed_files = set(group_data.keys()) - set(cleaned.keys())

        # Don't delete from disk yet — just update labels
        # for fid in removed_files:
        #     for ext, folder in [(".txt", texts_folder), (".npy", motion_folder)]:
        #         path = os.path.join(folder, fid + ext)
        #         if os.path.exists(path):
        #             os.remove(path)

        # Update labels
        labels = load_labels()
        labels[group_name] = cleaned
        save_labels(labels)

        after_count = len(cleaned)
        print(f"[{group_name}] {before_count} -> {after_count} (removed {len(removed_files)}: {removed_files if removed_files else 'none'})")

    except Exception as e:
        print(f"[ERROR] {group_name}: {e}")
        continue

print(f"\nDone. API calls made: {api_call_count}")
