import json
from tqdm import tqdm
from openai import OpenAI
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from constants import API_KEY

prompts = {
    "political_bias" : {
        "system" : ("És especialista em política Portuguesa. Vais interpretar texto e identificar enviesamento político. "
                    "Em Portugal, o Partido Socialista (PS), Bloco de Esquerda (BE), Livre, e Partido Comunista Português (PCP) são os partidos de esquerda. "
                    "O Partido Social Democrata (PSD), Iniciativa Liberal (IL), CDS, e Chega são os partidos de direita."),
        "user" : ('És um modelo treinado para identificar a viés política de artigos de opinião ou notícias. '
                'A tua tarefa é classificar o artigo fornecido como tendo uma viés de "direita" ou "esquerda".\n'
                'Um texto enviesado à direita tende a enfatizar valores como o livre mercado, a redução do papel do Estado, conservadorismo, nacionalismo e críticas a políticas progressistas.\n'
                'Um texto enviesado à esquerda tende a valorizar a intervenção do Estado na economia, a justiça social e direitos dos trabalhadores, a redistribuição de riqueza e críticas ao neoliberalismo.\n'
                'A saída deve ser sempre um JSON com o seguinte formato:\n'
                '{"classe": "direita"} ou {"classe": "esquerda"}\n'
                'O artigo abaixo está **claramente enviesado**. Classifica o viés do seguinte texto e devolve apenas a resposta formatada corretamente:\n\n')
    },
    
    "reliability" : {
        "system" : ('Tu és um assistente especializado em avaliar a fiabilidade de artigos com base na sua linguagem, estrutura e transparência. '
                    'A tua tarefa é classificar a fiabilidade e apresentar a resposta num formato JSON bem definido.\n'
                    'Diretrizes:\n'
                    '- Considera **apenas elementos formais** do texto (linguagem, tom, estrutura, uso de fontes). **Não avalies a veracidade do conteúdo.**\n'
                    '- A resposta deve ter o seguinte formato: {"classe": "classificação dada"}\n'
                    "Quando receberes um artigo, analisa a sua forma e avalia o nível de fiabilidade, seguindo rigorosamente o formato especificado."),
        "user" : ('És um modelo treinado para identificar a fiabilidade de artigos de opinião ou notícias. '
                'A tua tarefa é classificar o artigo fornecido como sendo "fiável" ou "não fiável", com base na sua linguagem, estrutura e transparência, sem avaliar a veracidade do conteúdo.\n'
                'Um texto fiável tende a apresentar fontes verificáveis, usar linguagem neutra e objetiva, ter uma estrutura clara e sem erros, e evitar exageros e sensacionalismo.\n'
                'Um texto não fiável tende a usar linguagem sensacionalista e emocionalmente carregada, ter erros gramaticais ou estruturais, e exagerar ou distorcer factos para apoiar um ponto de vista.\n'
                'A saída deve ser sempre um JSON com o seguinte formato: {"classe":"fiável"} ou {"classe":"não fiável"}\n'
                'O artigo abaixo está **claramente classificado** como uma dessas opções. Classifica a fiabilidade do seguinte texto e devolve **apenas a resposta formatada corretamente**:\n\n')
    },

    "objectivity" : {
        "system" : ('Tu és um assistente especializado em avaliar a objetividade de artigos com base na sua linguagem, estrutura e transparência. '
                    'A tua tarefa é classificar a objetividade e apresentar a resposta num formato JSON bem definido.\n'
                    'Diretrizes:\n'
                    '- Considera **apenas elementos formais** do texto (linguagem, terminologia, tom, estrutura). **Não avalies a veracidade do conteúdo.**\n'
                    '- A resposta deve ter o seguinte formato: {"classe": "classificação dada"}\n'
                    "Quando receberes um artigo, analisa a sua forma e avalia o nível de objetividade, seguindo rigorosamente o formato especificado."),
        "user" : ('És um modelo treinado para identificar a objetividade de artigos de opinião ou notícias. '
                'A tua tarefa é classificar o artigo fornecido como sendo "objetivo" ou "subjetivo", com base na sua linguagem, estrutura e transparência, sem avaliar a veracidade do conteúdo.\n'
                'Um texto objetivo tende a usar linguagem neutra e descritiva, evita expressar opiniões, apresenta dados verificáveis, ou mantém um tom imparcial e estrutura clara.\n'
                'Um texto subjetivo tende a utilizar linguagem opinativa ou emocional, expressa juízos de valor, favorece um ponto de vista sem apresentar contrapontos, ou faz suposições sem suporte em dados verificáveis.\n'
                'A saída deve ser sempre um JSON com o seguinte formato: {"classe":"objetivo"} ou {"classe":"subjetivo"}\n'
                'O artigo abaixo está **claramente classificado** como uma dessas opções. Classifica a objetividade do seguinte texto e devolve **apenas a resposta formatada corretamente**:\n\n')
    }
}

label_map = {
    "political_bias" : {
        "esquerda" : 0,
        "direita" : 1
    },

    "reliability" : {
        "não fiável" : 0,
        "fiável" : 1
    },

    "objectivity" : {
        "subjetivo" : 0,
        "objetivo" : 1
    }
}

deep_infra_key = API_KEY

openai = OpenAI(
    api_key=deep_infra_key,
    base_url="https://api.deepinfra.com/v1/openai",
)

def process_output(output, axis):
    json_object = json.loads(re.findall(r'\{.*?\}', output)[0])
    a,b = label_map[axis].keys()
    if json_object['classe'] == a:
        return 0
    elif json_object['classe'] == b:
        return 1
    else:
        return "inconclusivo"

def classify_llama(test_set, axis): 
    zero_shot_classifications = []

    for article in tqdm(test_set):

        system_prompt = prompts[axis]["system"]
        user_prompt = prompts[axis]["user"] + article['text']

        chat_completion = openai.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
        )

        output = chat_completion.choices[0].message.content
        
        try:
            processed_classification = process_output(output)
        except:
            processed_classification = output

        zero_shot_classifications.append({"title" : article["title"], "label" : article["label"], "classification" : processed_classification})

        # saving at every iteration
        with open(f"llama_zero_shot_{axis}.json", "w", encoding="utf-8") as file:
            json.dump(zero_shot_classifications, file, indent=4, ensure_ascii=False)

def classify_eurollm(test_set, axis):
    model_id = "utter-project/EuroLLM-9B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    zero_shot_classifications = []

    for article in tqdm(test_set):
        prompt_structure = [
            {"role": "system", "content": '{system_prompt}'},
            {"role": "user", "content": '{user_prompt}'}]
        
        system_prompt = prompts[axis]["system"]
        user_prompt = prompts[axis]["user"] + article['text']
        
        prompt_structure[0]["content"] = prompt_structure[0]["content"].format(system_prompt = system_prompt)
        prompt_structure[1]["content"] = prompt_structure[1]["content"].format(user_prompt = user_prompt)

        # tokenizing
        formatted_news = tokenizer.apply_chat_template(prompt_structure, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_news, return_tensors="pt", padding=True, truncation=True)

        # moving tensors to GPU
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=256,
                do_sample=False)

        output = tokenizer.decode(outputs[0], skip_special_tokens=True).split('assistant\n')[1]

        try:
            processed_classification = process_output(output)
        except:
            processed_classification = output

        zero_shot_classifications.append({"title" : article["title"], "label" : article["label"], "classification" : processed_classification})

        with open(f"eurollm_zero_shot_{axis}.json", "w", encoding="utf-8") as file:
            json.dump(zero_shot_classifications, file, indent=4, ensure_ascii=False)

def eval(axis, model):

    with open(f"{model}_zero_shot_{axis}.json", "r", encoding="utf-8") as file:
        zero_shot_classifications = json.load(file)

    true_labels = [classif['label'] for classif in zero_shot_classifications]
    predicted_labels = [classif['classification'] for classif in zero_shot_classifications]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print("Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")