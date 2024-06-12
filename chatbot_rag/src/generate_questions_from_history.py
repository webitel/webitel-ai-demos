from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticToolsParser,PydanticOutputParser
import pandas as pd
import json
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

#read data
examples = json.load(open("prompts/examples/examples.json"))['examples']


with open('data/raw/conversations_6m.json', 'r') as f:
    data = f.read()
    

data_new = data.replace('//', '/')

#format data
old = "\\\\"
new = "\\"
data_new = data_new.replace(old, new)
conversations = json.loads(data_new)

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

class ExtractedQA(BaseModel):
    """Extracted question and answer from the conversation."""

    question: str = Field(
        ...,
        description=" general question.",
    )
    
    answer: str = Field(
        ...,
        description=" corresponding answer to question",
    
    )

prefix_prompt = f"""
###TASK###
You are worker in SHOP domain called АТБ and your role is to generate syntetic data to train our models.
You will see conversation between user and worker, your aim is to extract general questions that user asked worker, you also need to extract answers that worker provided.

###RULES###
Some conversations do not have general answer, they require additional information from our side, you have to skip such conversations.(do not extract questions and answers from them)
Conversations that talk about specific event in specific date or time are also not suitable for our task.
There might be multiple question and answers extracted from one conversation, you need to extract all of them if they are suitable for our task.
Basically conversation answers to which are temporary or require additional information are not suitable for our task, for instance phone number of some place/service is fine, but event that is held on specifc date is not.
The extracted question must be a question from user and the answer should be from worker.

###EXAMPLES###
Here is an example of conversations.
Conversation 1:
{examples[0]['body']}
{examples[0]['postfix']}

Conversation 2:
{examples[1]['body']}
{examples[1]['postfix']}


Conversation 3:
{examples[2]['body']}
{examples[2]['postfix']}


Conversation 4:
{examples[3]['body']}
{examples[3]['postfix']}


Conversation 5:
{examples[4]['body']}
{examples[4]['postfix']}

"""

postfix_prompt = """
Now let's generate general questions and answers from the conversations, remember we need to skip conversations where we do not have general answer that is suitable for everyone.

###CONVERSATION###
Here is the conversation:
{example}


Start extracting the questions and answers in required format:"""

template = prefix_prompt + postfix_prompt

prompt = PromptTemplate.from_template(template)
llm_with_tools = llm.bind_tools([ExtractedQA])

syntetitc_data_generator = prompt | llm_with_tools


def process_chat(chat_history):
    new_chat = {'client_info': '', 'messages': []}
    
    operator_joined = True
    last_sender = ''
    last_message = ''
    conversation = ''
    if len(chat_history) <= 2 :
        raise ValueError('Too big or small chat history')
    
    for message in chat_history:
        if message['operator'] == True:
            operator_joined = True
            
        if message['name'] == 'bot' and operator_joined == False:
            new_chat['client_info']  += message['text'] + ' '
        elif operator_joined:
            new_sender = message['operator']
            new_message = message['text']
            
            if new_sender == last_sender:
                last_message += ' ' +  new_message 
            else:
                if last_sender == False:
                    conversation += 'User: ' + last_message + '\n'
                elif last_sender == True:
                    conversation += 'worker: ' + last_message + '\n'
                last_message = new_message            
            
            last_sender = message['operator']
    if last_sender == False:
        conversation += 'User: ' + last_message + '\n'
    elif last_sender == True:
        conversation += 'worker: ' + last_message + '\n'

    return conversation

extracted_questions_and_answers = []
total_count = 0

#кожен 5й чат процеситься, приблизно за 1000 чатів ціна 20$ (саме запроцешених) 
for chats in conversations[:100]:
    try:
        extracted_info_list = []
        chat_history = chats['messages']
        extracted_info = syntetitc_data_generator.invoke({"example":process_chat(chat_history)})
        # print()
        print("Extracted info:")
        total_count +=1
        print(extracted_info)
        for tool_call in extracted_info.additional_kwargs['tool_calls']:
            extracted_info_list.append(tool_call['function']['arguments'])
        extracted_questions_and_answers.append((process_chat(chat_history), extracted_info_list))
    except Exception as e:
        print(str(e))
        pass

print('Total processed chats:', total_count)
print(extracted_questions_and_answers)
pd.DataFrame(extracted_questions_and_answers,columns=['chat_history','extracted_qa']).to_csv('generated_qa/extracted_questions_and_answers5.csv', index=False)
