import json
import re
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm
from openai import OpenAI
from constants import API_KEY

prompts = { "political_bias" : {
                "system" : ("És especialista em política Portuguesa. Vais interpretar texto e identificar enviesamento político. "
                      "Em Portugal, o Partido Socialista (PS), Bloco de Esquerda (BE), Livre, e Partido Comunista Português (PCP) são os partidos de esquerda. "
                      "O Partido Social Democrata (PSD), Iniciativa Liberal (IL), CDS, e Chega são os partidos de direita."),
                "user" : ('Um indicador de viés político é aqui definido como um rótulo descritivo que represente a presença de enviesamento político num texto.\n '
                    'O que vais detetar é especificamente enviesamento político. \n\n'
                    'A saída deve ser sempre um JSON com o seguinte formato:\n'
                    '{"indicador": "descrição do viés identificado"}'
                    'Exemplos:\n'
                    '{"indicador": "Refere-se à imigração como algo negativo para a sociedade."}\n'
                    '{"indicador": "Apoia a intervenção do governo na economida e sociedade."}\n'
                    '{"indicador": "Menciona as vantagens de impostos baixos para o crescimento económico."}\n'
                    '{"indicador": "Critica o capitalismo."}\n'
                    '{"indicador": "Desvaloriza o estado social e valoriza o mérito individual."}\n'
                    '{"indicador": "Critica negativamente a posição de certos partidos de direita."}\n\n'
                    'Extrai **apenas um novo indicador** de viés político do seguinte texto e formata a saída **exatamente como nos exemplos**:\n\n')
            },

            "reliability" : {
                "system" : ('Tu és um assistente especializado em avaliar a fiabilidade de artigos com base na sua linguagem, estrutura e transparência. '
                    'A tua tarefa é identificar indicadores de fiabilidade e apresentar a resposta num formato JSON bem definido.\n'
                    'Diretrizes:\n'
                    '- Considera **apenas elementos formais** do texto (linguagem, tom, estrutura, uso de fontes). **Não avalies a veracidade do conteúdo.**\n'
                    '- Identifica um único indicador por vez.\n'
                    '- A resposta deve ter o seguinte formato: {"indicador": "descrição do indicador identificado"}\n'
                    "Quando receberes um artigo, analisa a sua forma e extrai apenas um indicador do nível de fiabilidade, seguindo rigorosamente o formato especificado."),
                "user" : ('Um indicador de fiabilidade é aqui definido como um rótulo descritivo que represente a presença de elementos num artigo que podem indicar maior ou menor credibilidade.\n'
                    'O que vais detetar são especificamente sinais linguísticos, estruturais ou estilísticos que afetam credibilidade percebida do artigo.\n'
                    'A saída deve ser sempre um JSON com o seguinte formato:\n'
                    '{"indicador": "descrição do indicador identificado"}\n'
                    'Exemplos:\n'
                    '{"indicador": "Apresenta fontes verificáveis para as informações mencionadas."}\n'
                    '{"indicador": "Utiliza linguagem sensacionalista para atrair atenção."}\n'
                    '{"indicador": "Utiliza uma linguagem neutra e objetiva."}\n'
                    '{"indicador": "Faz afirmações fortes sem citar fontes verificáveis."}\n'
                    '{"indicador": "Evita exageros ou distorções ao apresentar os fatos."}\n'
                    '{"indicador": "Apresenta erros gramaticais e ortográficos."}\n'
                    '{"indicador": "O texto está bem estruturado e sem erros gramaticais."}\n'
                    '{"indicador": "Uso excessivo de linguagem emocional e adjetivos carregados."}\n\n'
                    'Extrai **apenas um novo indicador** do nível de fiabilidade do seguinte texto e formata a saída **exatamente como nos exemplos**:\n')
            },

            "objectivity" : {
                "system" : ('Tu és um assistente especializado em avaliar objetividade de artigos com base na sua linguagem, estrutura e transparência. '
                    'A tua tarefa é identificar indicadores de objetividade/subjetividade e apresentar a resposta num formato JSON bem definido.\n'
                    'Diretrizes:\n'
                    '- Considera **apenas elementos formais** do texto (linguagem, terminologia, tom, estrutura). **Não avalies a veracidade do conteúdo.**\n'
                    '- Identifica um único indicador por vez.\n'
                    '- A resposta deve ter o seguinte formato: {"indicador": "descrição do indicador identificado"}\n'
                    "Quando receberes um artigo, analisa a sua forma e extrai apenas um indicador de objetividade/subjetividade, seguindo rigorosamente o formato especificado."),
                "user" : ('Um indicador de objetividade é aqui definido como um rótulo descritivo que represente a presença de elementos num artigo que contribuem para a sua maior ou menor imparcialidade e rigor.\n'
                    'O que vais detetar são especificamente sinais linguísticos, estruturais ou estilísticos que afetam a objetividade percebida do artigo.\n'
                    'A saída deve ser sempre um JSON com o seguinte formato:\n'
                    '{"indicador": "descrição do indicador identificado"}\n'
                    'Exemplos:\n'
                    '{"indicador": "Apresenta dados concretos e verificáveis para fundamentar as informações."}\n'
                    '{"indicador": "Utiliza linguagem opinativa, expressando juízos de valor."}\n'
                    '{"indicador": "Evita linguagem emocional ou adjetivos subjetivos."}\n'
                    '{"indicador": "Revela preferência explícita por um ponto de vista sem apresentar contrapontos."}\n'
                    '{"indicador": "Utiliza um tom neutro e descritivo, sem expressar opinião."}\n'
                    '{"indicador": "Inclui suposições ou generalizações sem suporte em dados verificáveis."}\n'
                    '{"indicador": "Inclui referências a fontes credíveis e verificáveis."}\n'
                    '{"indicador": "Apresenta argumentos persuasivos em vez de informações neutras."}\n\n'
                    'Extrai **apenas um novo indicador** de objetividade do seguinte texto e formata a saída **exatamente como nos exemplos**:\n')
            }
        }

openai = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

def process_descriptor_json(descriptor):
    json_object = json.loads(re.findall(r'\{.*?\}', descriptor)[0])
    return json_object["indicador"]

def get_score(descriptor, threshold, chroma_db, k):
    # Search for a similar sentence
    results = chroma_db.similarity_search_with_score(descriptor, k=k)

    # filtering out indicators with similarity lower than threshold
    results_filtered = [indi for indi in results if (1 - indi[1]) > threshold]
    
    # Axis score
    sim_scores = [1 - indi[1] for indi in results_filtered]
    class_scores = [indi[0].metadata['score'] for indi in results_filtered]
    axis_score = sum(sim_scores[i] * class_scores[i] for i in range(len(sim_scores))) / sum(sim_scores)

    return axis_score

# for political_bias, objectivity, and reliability
def get_descriptors_llama(test_set, axis):

    descriptors = []
    for article in tqdm(test_set):
        system_prompt = prompts[axis]["system"]
        user_prompt = prompts[axis]["user"]

        user_prompt_complete = user_prompt + article["text"]

        chat_completion = openai.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_complete},
        ],
        temperature=0
        )

        descriptor = chat_completion.choices[0].message.content

        try:
            processed_descriptor = process_descriptor_json(descriptor)
        except:
            processed_descriptor = descriptor
        
        descriptors.append({"title" : article["title"], "descriptor" : processed_descriptor, "label" : article['label']})

        # saving at every iteration
        with open(f"descriptors_llama_{axis}.json", "w", encoding="utf-8") as file:
            json.dump(descriptors, file, indent=4, ensure_ascii=False)

def get_descriptor_eurollm(test_set, axis):

    descriptors = []
    
    model_id = "utter-project/EuroLLM-9B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for article in tqdm(test_set):

        system_prompt = prompts[axis][str(article["label"])]["system"]
        user_prompt = prompts[axis][str(article["label"])]["user"] + article['text']

        prompt_structure = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
        ]

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
                do_sample=False
            )

        output_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        descriptor = output_decoded.split('assistant\n')[1]

        try:
            processed_descriptor = process_descriptor_json(descriptor)
        except:
            processed_descriptor = descriptor

        descriptors.append({"title" : article["title"], "descriptor" : processed_descriptor, "label" : article['label']})

        # saving at every iteration
        with open(f"descriptors_eurollm_{axis}.json", "w", encoding="utf-8") as file:
            json.dump(descriptors, file, indent=4, ensure_ascii=False)