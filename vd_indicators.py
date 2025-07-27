import json
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import re
from tqdm import tqdm
from openai import OpenAI
from constants import API_KEY

prompts = {
    "political_bias" : {
        "0" : {
            "system" : ("És especialista em política Portuguesa. Vais interpretar texto e identificar enviesamento político. "
                      "Em Portugal, o Partido Socialista (PS), Bloco de Esquerda (BE), Livre, e Partido Comunista Português (PCP) são os partidos de esquerda. "
                      "O Partido Social Democrata (PSD), Iniciativa Liberal (IL), CDS, e Chega são os partidos de direita."),
            "user" : ('Um indicador de viés político é aqui definido como um rótulo descritivo que represente a presença de enviesamento político num texto.\n '
                    'O que vais detetar é especificamente enviesamento político à esquerda. \n\n'
                    'A saída deve ser sempre um JSON com o seguinte formato:\n'
                    '{"bias": "descrição do viés identificado"}'
                    'Exemplos:\n'
                    '{"bias": "Refere-se à imigração como algo importante para a sociedade."}\n'
                    '{"bias": "Apoia a intervenção do governo na economia e sociedade."}\n'
                    '{"bias": "Critica o capitalismo."}\n'
                    '{"bias": "Fala de alterações climáticas como um problema ugente."}\n'
                    '{"bias": "Critica negativamente a posição de certos partidos de direita."}\n\n'
                    'Extrai **apenas um novo indicador** de viés político à esquerda do seguinte texto e formata a saída **exatamente como nos exemplos**:\n\n')
        },
        "1" : {
            "system" : ("És especialista em política Portuguesa. Vais interpretar texto e identificar enviesamento político. "
                      "Em Portugal, o Partido Socialista (PS), Bloco de Esquerda (BE), Livre, e Partido Comunista Português (PCP) são os partidos de esquerda. "
                      "O Partido Social Democrata (PSD), Iniciativa Liberal (IL), CDS, e Chega são os partidos de direita."),
            "user" : ('Um indicador de viés político é aqui definido como um rótulo descritivo que represente a presença de enviesamento político num texto.\n '
                    'O que vais detetar é especificamente enviesamento político à direita. \n\n'
                    'A saída deve ser sempre um JSON com o seguinte formato:\n'
                    '{"bias": "descrição do viés identificado"}'
                    'Exemplos:\n'
                    '{"bias": "Refere-se à imigração como algo negativo para a sociedade."}\n'
                    '{"bias": "Menciona as vantagens de impostos baixos para o crescimento económico."}\n'
                    '{"bias": "Apoia que o governo deve ter pouco envolvimento na economia e sociedade."}\n'
                    '{"bias": "Desvaloriza o estado social e valoriza o mérito individual."}\n'
                    '{"bias": "Critica negativamente a posição de certos partidos de esquerda."}\n\n'
                    'Extrai **apenas um novo indicador** de viés político à direita do seguinte texto e formata a saída **exatamente como nos exemplos**:\n\n')
        }
    },

    "reliability" : {
        "0" : {
            "system" : ('Tu és um assistente especializado em avaliar não fiabilidade de artigos com base na sua linguagem, estrutura e transparência. '
                        'A tua tarefa é identificar indicadores de não fiabilidade e apresentar a resposta num formato JSON bem definido.\n'
                        'Diretrizes:\n'
                        '- Considera **apenas elementos formais** do texto (linguagem, tom, estrutura, uso de fontes). **Não avalies a veracidade do conteúdo.**\n'
                        '- Identifica um único indicador por vez.\n'
                        '- A resposta deve ter o seguinte formato: {"não fiável": "descrição do indicador identificado"}\n'
                        "Quando receberes um artigo, analisa a sua forma e extrai apenas um indicador de não fiabilidade, seguindo rigorosamente o formato especificado."),
            "user" : ('Um indicador de não fiabilidade é aqui definido como um rótulo descritivo que represente a presença de elementos num artigo que podem indicar menor credibilidade.\n'
                    'O que vais detetar são especificamente sinais linguísticos, estruturais ou estilísticos que afetam negativamente a fiabilidade percebida do artigo.\n'
                    'A saída deve ser sempre um JSON com o seguinte formato:\n'
                    '{"não fiável": "descrição do indicador identificado"}\n'
                    'Exemplos:\n'
                    '{"não fiável": "Utiliza linguagem sensacionalista para atrair atenção."}\n'
                    '{"não fiável": "Faz afirmações fortes sem citar fontes verificáveis."}\n'
                    '{"não fiável": "Apresenta erros gramaticais e ortográficos."}\n'
                    '{"não fiável": "Inclui exageros ou distorções de factos para apoiar um ponto de vista."}\n'
                    '{"não fiável": "Uso de termos vagos e indefinidos como \'muitos especialistas dizem\'."}\n'
                    '{"não fiável": "Uso excessivo de linguagem emocional e adjetivos carregados."}\n\n'
                    'Assumindo o seguinte texto como sendo não fiável, extrai **apenas um novo indicador** de não fiabilidade do seguinte texto e formata a saída **exatamente como nos exemplos**:\n')
        },
        "1" : {
            "system" : ('Tu és um assistente especializado em avaliar a fiabilidade de artigos com base na sua linguagem, estrutura e transparência. '
                        'A tua tarefa é identificar indicadores de fiabilidade e apresentar a resposta num formato JSON bem definido.\n'
                        'Diretrizes:\n'
                        '- Considera **apenas elementos formais** do texto (linguagem, tom, estrutura, uso de fontes). **Não avalies a veracidade do conteúdo.**\n'
                        '- Identifica um único indicador por vez.\n'
                        '- A resposta deve ter o seguinte formato: {"fiável": "descrição do indicador identificado"}\n'
                        "Quando receberes um artigo, analisa a sua forma e extrai apenas um indicador de fiabilidade, seguindo rigorosamente o formato especificado."),
            "user" : ('Um indicador de fiabilidade é aqui definido como um rótulo descritivo que represente a presença de elementos num artigo que podem indicar maior credibilidade.\n'
                    'O que vais detetar são especificamente sinais linguísticos, estruturais ou estilísticos que afetam positivamente a fiabilidade percebida do artigo.\n'
                    'A saída deve ser sempre um JSON com o seguinte formato:\n'
                    '{"fiável": "descrição do indicador identificado"}\n'
                    'Exemplos:\n'
                    '{"fiável": "Apresenta fontes verificáveis para as informações mencionadas."}\n'
                    '{"fiável": "Utiliza uma linguagem neutra e objetiva."}\n'
                    '{"fiável": "Evita exageros ou distorções ao apresentar os fatos."}\n'
                    '{"fiável": "O texto está bem estruturado e sem erros gramaticais."}\n'
                    '{"fiável": "Fornece múltiplos pontos de vista sobre o tema abordado."}\n'
                    '{"fiável": "Apresenta estatísticas que comprovam o argumento."}\n\n'
                    'Assumindo o seguinte texto como sendo fiável, extrai **apenas um novo indicador** de fiabilidade do seguinte texto e formata a saída **exatamente como nos exemplos**:\n')
        }
    },

    "objectivity" : {
        "0" : {
            "system" : ('Tu és um assistente especializado em avaliar subjetividade de artigos com base na sua linguagem, estrutura e transparência. '
                        'A tua tarefa é identificar indicadores de subjetividade e apresentar a resposta num formato JSON bem definido.\n'
                        'Diretrizes:\n'
                        '- Considera **apenas elementos formais** do texto (linguagem, terminologia, tom, estrutura). **Não avalies a veracidade do conteúdo.**\n'
                        '- Identifica um único indicador por vez.\n'
                        '- A resposta deve ter o seguinte formato: {"subjetivo": "descrição do indicador identificado"}\n'
                        "Quando receberes um artigo, analisa a sua forma e extrai apenas um indicador de subjetividade, seguindo rigorosamente o formato especificado."),
            "user" : ('Um indicador de subjetividade é aqui definido como um rótulo descritivo que represente a presença de elementos num artigo que indicam parcialidade, opinião ou falta de rigor factual.\n'
                    'O que vais detetar são especificamente sinais linguísticos, estruturais ou estilísticos que reduzem a objetividade percebida do artigo.\n'
                    'A saída deve ser sempre um JSON com o seguinte formato:\n'
                    '{"subjetivo": "descrição do indicador identificado"}\n'
                    'Exemplos:\n'
                    '{"subjetivo": "Utiliza linguagem opinativa, expressando juízos de valor."}\n'
                    '{"subjetivo": "Revela preferência explícita por um ponto de vista sem apresentar contrapontos."}\n'
                    '{"subjetivo": "Inclui suposições ou generalizações sem suporte em dados verificáveis."}\n'
                    '{"subjetivo": "Faz uso frequente de adjetivos subjetivos e linguagem emocional."}\n'
                    '{"subjetivo": "Apresenta argumentos persuasivos em vez de informações neutras."}\n'
                    '{"subjetivo": "Emprega termos ambíguos ou vagos que dificultam a verificação das afirmações."}\n\n'
                    'Assumindo o seguinte texto como sendo subjetivo ou baseado em opinião, extrai **apenas um novo indicador** de subjetividade do seguinte texto e formata a saída **exatamente como nos exemplos**:\n')
        },
        "1" : {
            "system" : ('Tu és um assistente especializado em avaliar objetividade de artigos com base na sua linguagem, estrutura e transparência. '
                        'A tua tarefa é identificar indicadores de objetividade e apresentar a resposta num formato JSON bem definido.\n'
                        'Diretrizes:\n'
                        '- Considera **apenas elementos formais** do texto (linguagem, terminologia, tom, estrutura). **Não avalies a veracidade do conteúdo.**\n'
                        '- Identifica um único indicador por vez.\n'
                        '- A resposta deve ter o seguinte formato: {"objetivo": "descrição do indicador identificado"}\n'
                        "Quando receberes um artigo, analisa a sua forma e extrai apenas um indicador de objetividade, seguindo rigorosamente o formato especificado."),
            "user" : ('Um indicador de objetividade é aqui definido como um rótulo descritivo que represente a presença de elementos num artigo que contribuem para a sua imparcialidade e rigor.\n'
                    'O que vais detetar são especificamente sinais linguísticos, estruturais ou estilísticos que reforçam a objetividade percebida do artigo.\n'
                    'A saída deve ser sempre um JSON com o seguinte formato:\n'
                    '{"objetivo": "descrição do indicador identificado"}\n'
                    'Exemplos:\n'
                    '{"objetivo": "Apresenta dados concretos e verificáveis para fundamentar as informações."}\n'
                    '{"objetivo": "Evita linguagem emocional ou adjetivos subjetivos."}\n'
                    '{"objetivo": "Utiliza um tom neutro e descritivo, sem expressar opinião."}\n'
                    '{"objetivo": "Fornece múltiplas perspetivas sobre o tema, sem favorecer uma posição específica."}\n'
                    '{"objetivo": "Inclui referências a fontes credíveis e verificáveis."}\n'
                    '{"objetivo": "Estrutura clara e lógica, facilitando a compreensão da informação."}\n\n'
                    'Assumindo o seguinte texto como sendo objetivo e factual, extrai **apenas um novo indicador** de objetividade do seguinte texto e formata a saída **exatamente como nos exemplos**:\n')
        }
    }
}

json_types = {
    "political_bias" : {
        "0" : "bias",
        "1" : "bias"
    },

    "reliability" : {
        "0" : "não fiável",
        "1" : "fiável"
    },

    "objectivity" : {
        "0" : "subjetivo",
        "1" : "objetivo"
    }
}

openai = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

def process_indicator_json(indicator, axis, label):
    json_object = json.loads(re.findall(r'\{.*?\}', indicator)[0])
    json_type = json_types[axis][label]
    return json_object[json_type]

def get_indicators_llama(train_set, axis):
    indicators = {}

    for article in tqdm(train_set):

        system_prompt = prompts[axis][str(article["label"])]["system"]
        user_prompt = prompts[axis][str(article["label"])]["user"] + article['text']

        chat_completion = openai.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
        )

        indicator = chat_completion.choices[0].message.content

        try:
            processed_indicator = process_indicator_json(indicator, axis, str(article['label']))
        except:
            processed_indicator = indicator

        indicators[processed_indicator] = {'label' : article['label'], 'title' : article['title'], 'score' : article['score']}

        # saving at every iteration
        with open(f"indicators_llama_{axis}.json", "w", encoding="utf-8") as file:
            json.dump(indicators, file, indent=4, ensure_ascii=False)

def get_indicators_eurollm(train_set, axis):

    model_id = "utter-project/EuroLLM-9B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    indicators = {}
    for article in tqdm(train_set):

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
        indicator = output_decoded.split('assistant\n')[1]

        try:
            processed_indicator = process_indicator_json(indicator, objective=True)
        except:
            processed_indicator = indicator

        indicators[processed_indicator] = {'label' : article['label'], 'title' : article['title'], 'score' : article['score']}

        # saving at every iteration
        with open(f"indicators_eurollm_{axis}.json", "w", encoding="utf-8") as file:
            json.dump(indicators, file, indent=4, ensure_ascii=False)
    

def create_db(axis, model):
    # When running for the first time to setup the db's
    with open(f"indicators_{model}_{axis}.json", "r", encoding="utf-8") as file:
        indicators = json.load(file)

    indicators_doc = [
        Document(page_content=ind, metadata=meta) for ind, meta in indicators.items()
    ]

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    Chroma.from_documents(indicators_doc, embedding_model, collection_metadata={"hnsw:space": "cosine"}, persist_directory=f"./{model}_{axis}_db")