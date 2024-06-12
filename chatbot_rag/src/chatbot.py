import logging
from sentence_transformers import CrossEncoder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
import json
import dotenv
from src.utils import load_prompt
from datetime import datetime
import os 

dotenv.load_dotenv()

openai_model = os.environ.get("OPENAI_MODEL")
api_key = os.environ.get("OPENAI_API_KEY")
device = os.environ.get("DEVICE")
print("DEVICE in chatbot", device)


logname = 'app.logs'
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('UI')


class ParaphrasedQuery(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""

    paraphrased_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )
    
class Reranker():
    def __init__(self, model_name="DiTy/cross-encoder-russian-msmarco"):
        #
        self.model = CrossEncoder(model_name, max_length=512, device=device)#
    
    def get_rank(self, queries, docs):
        return self.model.rank(queries, docs)
    
    def rerank(self, queries, docs):
        # return self.get_rank(queries, docs)
        raise NotImplementedError("Reranking is not implemented yet.")

class RagAgent():
    
    def __init__(self,vector_retriever):
        self.reranker = create_reranker()
        self.query_analyzer = create_query_analyzer()
        self.history_aware_retriever = get_history_aware_retriever(vector_retriever)
        self.question_answer_chain = create_question_answer_chain()
        self.retriever = vector_retriever
        self.refiner = refiner()
        self.full_name = None
        self.contract = None
        self.overdue = None
        self.phone = None
        self.debt = None
    
    def change_user(self,full_name = 'Назар Андрушко', contract = '№12314412', overdue = '12 днів', phone = '380123456789', debt = '15000 грн', user_category = 'default',startDate = '2021-01-01', dueDate ='2021-02-01'):
        self.full_name = full_name
        self.contract = contract
        self.overdue = overdue
        self.phone = phone
        self.user_category = user_category
        self.debt = debt
        self.question_answer_chain = create_question_answer_chain(full_name, contract, overdue, phone, debt,startDate,dueDate)
    
    def change_reranker(self,model_name):
        if model_name == "default":
            return
        logger.info('Cganging reranker model to: ' + model_name)
        self.reranker = Reranker(model_name='output/'+model_name+'-latest')
        
    def advanced_rag(self,input_query, chat_history = []):
        #keep only 6 last messages (3 user messages and 3 ai messages)
        chat_history = chat_history[-6:]
        
        #step 1 generate new queries with similar meaning
        new_queries = self.query_analyzer.invoke({"question":input_query})
        ## add original query to the list
        new_queries.append(ParaphrasedQuery(paraphrased_query=input_query))
        logger.info(f'Generated new queries with similar meaning: {new_queries}' + '\n')

        #step 2 run default retriever with memory 
        prefix_query = "Питання: "
        if len(chat_history) >= 2:
            prefix_query = f"Питання:{chat_history[-2]} Відповідь: {chat_history[-1]} Питання: "
        context_docs = []
        for query in new_queries:
            # context_docs.extend(self.history_aware_retriever.invoke({"input":prefix_query + query.paraphrased_query,"chat_history":chat_history}))
            context_docs.extend(self.retriever.invoke(prefix_query + query.paraphrased_query))
            context_docs.extend(self.history_aware_retriever.invoke({"input":query.paraphrased_query,"chat_history":chat_history}))
            context_docs.extend(self.retriever.invoke(query.paraphrased_query))

        logger.info(f'Context docs: {context_docs}' + '\n')

        # context_docs = history_aware_retriever.invoke({"input":input_query,"chat_history":[HumanMessage("Вопрос: Когда мне виплатят деньги?"), AIMessage("Мы пока не знаем когда вам отдадут деньги") ]})
        #step 3 run ranker and select best context docs
        
        ## remove duplicated docs in context_docs
        deduped_docs = []
        for doc in context_docs:
            # for doc2 in deduped_docs:
            #     if doc.page_content.split('Ответ')[1] in doc2.page_content.split('Ответ')[1]:
            #         continue

            if doc not in deduped_docs:
                deduped_docs.append(doc)
        ## run ranker
        ranks = self.reranker.get_rank(input_query, [doc.page_content for doc in deduped_docs])
            
        #select 5 best ranked docs
        best_ranked_docs = []
        for rank in ranks[:5]:
            best_ranked_docs.append(deduped_docs[rank['corpus_id']])
        
        
        logger.info(f'Used docs for answer: {best_ranked_docs}' + '\n')
        
        #step 4 run default question answer chain
        context = ""
        for i,doc in enumerate(best_ranked_docs):
            context += f"Context {i}: " +doc.page_content + "\n"
        
        answer = self.question_answer_chain.invoke({"input":input_query,"context":best_ranked_docs, "chat_history":chat_history})
        retries = 0
        while 'я не знаю відповіді' in answer and retries <3:
            answer = self.question_answer_chain.invoke({"input":input_query,"context":best_ranked_docs, "chat_history":chat_history})
            retries += 1
        logger.info(f'Original question :{input_query} Answer: {answer}' + '\n')
        
        #step 5 refine 
        # refined_answer = self.refiner.invoke({"input":answer,"chat_history":chat_history,'context':best_ranked_docs,'ai_response':answer})
        
        
        return answer, best_ranked_docs
    
    


    
def create_query_analyzer():
    system = load_prompt('prompts/query_analyzer_prompt.txt')
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model=openai_model, temperature=0,)
    llm_with_tools = llm.bind_tools([ParaphrasedQuery])
    query_analyzer = prompt | llm_with_tools | PydanticToolsParser(tools=[ParaphrasedQuery])
    
    return query_analyzer

def get_history_aware_retriever(retriever):
    llm = ChatOpenAI(model=openai_model)
    contextualize_q_system_prompt = load_prompt('prompts/history_aware_retriever_prompt.txt')
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def create_question_answer_chain(full_name=None, contract=None, overdue=None, phone= None, debt= None,startDate= None,dueDate= None):
    llm = ChatOpenAI(model=openai_model,temperature=0.4)
    currentDate = datetime.today().strftime('%Y-%m-%d')
    prefix = load_prompt('prompts/question_answer_chain.txt')
    prefix = prefix.format(full_name=full_name, contract=contract, overdue=overdue, phone=phone, debt=debt,startDate=startDate,dueDate=dueDate,currentDate=currentDate)
    postfix = """Here is the context that you can use to answer the question and provide accrurate answer, you can rephrase answer from the context to make it more natural and accurate.:
    {context}"""
    
    qa_system_prompt = prefix + postfix
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return question_answer_chain

def refiner():
    llm = ChatOpenAI(model=openai_model,)
    
    refiner_prompt = """You are an assistant for Miloan company to help users with their queries.
    You will receive AI response, context and chat history you aim is to refine the AI response to make it more natural and accurate.
    If the answer is not accurate and there is better answer in the context you should provide this answer in natural manner.
    If the answer is good, then just make it more natural.
    If you don't know the answer say 'Вибачте, я не знаю відповіді на це питання, я підключу вас до оператора.'.
    Rely on the context information more than on your knowledge.
    If you don't know the answer, simply say you don't know.
    There might be multiple fields "Ответ" in the context, you should use the one that is next to the question - 'Вопрос' that is similar to the user's question.
    The question in context may be phrased in a different way, but if it is the question with the same meaning, you should answer it in similar way.
    Try not to repeat answers that have already been given.
    Try not to repeat yourself, so that user will be engaged in the conversation and will not be disappointed, you as assistant will be granted a tip for this, but you must give only relevant answers, if the answer is not relevant you should retutn 'Вибачте, я не знаю відповіді на це питання, я підключу вас до оператора.'
    Avoid repeating the same response multiple times; instead, rephrase it.
    Rely more on the initial examples than the latter ones, as they are more relevant.
    Your answers must be in ukrainian language.

    Proposed AI response:{ai_response}. 
    As output either provide refined AI response to user query or 'Вибачте, я не знаю відповіді на це питання, я підключу вас до оператора.'

    Use the following extracted contextual fragments to respond to the question.

    {context}"""
    refiner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", refiner_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    refiner_chain = create_stuff_documents_chain(llm, refiner_prompt)
    return refiner_chain

def create_reranker():
    return Reranker()

def generate_questions(question, answer):
    llm = ChatOpenAI(model=openai_model, temperature=0.4,)


    template = load_prompt('prompts/generate_similar_questions_prompt.txt')

    prompt = PromptTemplate.from_template(template)
    llm_with_tools = llm.bind_tools([ParaphrasedQuery])
    syntetitc_data_generator = prompt | llm_with_tools
    res = syntetitc_data_generator.invoke({"example": question + " Відповідь: " + answer})
    generated_questions = ""
    for i,tool_call in enumerate(res.additional_kwargs['tool_calls']):
        generated_question = json.loads(tool_call['function']['arguments'])['paraphrased_query']
        generated_questions += f"Q{i}: {generated_question} \n" 
    return generated_questions