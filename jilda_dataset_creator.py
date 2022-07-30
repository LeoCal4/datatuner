import json

TAGS_INDEX = 3

def create_jilda_d2t_dataset(dataset_path: str, translations_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        parsed_text = json.load(f)
    with open(translations_path, "r", encoding="utf-8") as f:
        values_translations = json.load(f)
    dataset = []
    for tags in parsed_text:
        if not tags[TAGS_INDEX]:
            continue
        #* skip all dialogue acts spoken by users and greetings 
        dialogue_act = tags[TAGS_INDEX][0][0]
        if dialogue_act.endswith("_greet") or dialogue_act.startswith("usr_"):
            continue
        input_data = ""
        previous_dialogue_act = ""
        for entry in tags[TAGS_INDEX]:
            if not entry:
                continue
            current_dialogue_act = entry[0]
            try:
                current_dialogue_act = values_translations[current_dialogue_act]
            except KeyError:
                print("Greet found, removing the following sentence")
                print(' '.join(tags[0]))
                continue
            slot_name = entry[1]
            slot_name = values_translations[slot_name]
            slot_value = entry[2]
            if slot_value in values_translations:
                slot_value = values_translations[slot_value]
            if previous_dialogue_act == "" or previous_dialogue_act != current_dialogue_act: 
                if previous_dialogue_act != "":
                    input_data += "), "
                input_data += f"<{current_dialogue_act}> {current_dialogue_act.replace('_', ' ')} (<{slot_name}> {slot_name.replace('_', ' ')}: {slot_value}"
            elif previous_dialogue_act == current_dialogue_act:
                input_data += f", <{slot_name}> {slot_name.replace('_', ' ')}: {slot_value}"
            else:
                raise ValueError("This should not happen")
            previous_dialogue_act = current_dialogue_act
        input_data += ")"
        input_data = '"' + input_data + '"'
        current_text = ""
        for text_token in tags[0]:
            #* do not add space before punctuation
            if text_token in ",.()!?':;-":
                current_text += text_token
            else:
                current_text += " " + text_token
        current_text = '"' + current_text.strip() + '"'
        dataset.append(f"{input_data}, {current_text}\n")
    with open("jilda_datatuner_test.csv", "w", encoding="utf-8") as f:
        f.write("mr,ref\n")
        f.writelines(dataset)



def jilda_dataset_analysis(dataset_path: str):
    with open(dataset_path, "r") as f:
        parsed_text = json.load(f)
    tags_types = []
    tags_subtypes =  {}
    for tags in parsed_text:
        if not tags[TAGS_INDEX]:
            continue
        for entry in tags[TAGS_INDEX]:
            if not entry: continue
            current_tag = entry[0]
            tag_subtype = entry[1]
            tags_types.append(current_tag)
            if not current_tag in tags_subtypes:
                tags_subtypes[current_tag] = [tag_subtype]
            else:
                if tag_subtype not in tags_subtypes[current_tag]: 
                    tags_subtypes[current_tag].append(tag_subtype)
    tags_types = set(tags_types)
    print("================== TAGS TYPES ===================")
    print(tags_types)
    print("================== TAGS ===================")
    for tag in tags_subtypes.keys():
        print(f"TAG {tag}:")
        for subtype in tags_subtypes[tag]:
            print(f"\t- {subtype}")
    print(f"There are {len(tags_subtypes.keys())} tags")



if __name__ == "__main__":
    dataset_path = r"C:\Users\Leo\Desktop\Tesi\datasets\jilda\train_data.json"
    translation_path = r"C:\Users\Leo\Documents\PythonProjects\Tesi\datatuner\src\datatuner\lm\custom\ignore\jilda_translations.json"
    create_jilda_d2t_dataset(dataset_path, translation_path)
