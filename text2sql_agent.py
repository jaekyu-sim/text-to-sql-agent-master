import sqlite3


from langgraph.prebuilt import create_react_agent
# from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from langchain.tools import tool
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# 벡터 스토어: langchain_chroma + OllamaEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document  # LangChain Document

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("SKCC_AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("SKCC_AZURE_ENDPOINT"),
    openai_api_type="azure",
    api_version="2024-02-15-preview",
)

SQL_MEMORY_PERSIST_DIR = "./sql_memory_chroma_db"
SQL_MEMORY_COLLECTION_NAME = "sql_memory"

def load_or_build_sql_memory_store():
    """
    SQL 질문/쿼리 캐시용 Vector Store (Chroma, langchain_chroma 버전)
    디렉토리가 있든 없든 동일하게 Chroma 객체 생성.
    add_documents() 호출 시 내부적으로 컬렉션이 만들어짐.
    """
    os.makedirs(SQL_MEMORY_PERSIST_DIR, exist_ok=True)

    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name=SQL_MEMORY_COLLECTION_NAME,
        persist_directory=SQL_MEMORY_PERSIST_DIR,
    )
    return vector_store


vector_store = load_or_build_sql_memory_store()

# Azure OpenAI 클라이언트 초기화
llm_selector = AzureChatOpenAI(
    azure_deployment=os.getenv("SKCC_AZURE_MODEL"),#"gpt-4o",
    azure_endpoint=os.getenv("SKCC_AZURE_ENDPOINT"),
    api_key=os.getenv("SKCC_AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.7
)
#llm_selector = ChatOllama(model="qwen3:4b", temperature=0.2, format="json")
import json



@dataclass
class TableSchema:
    table_name: str
    columns: List[str]
    descriptions: Dict[str, str] = None
    samples: List[Dict[str, Any]] = None

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", s.lower())

def _score(text: str, q_tokens: List[str], keyword_map: Dict[str, List[str]], relation_bonus_map: Dict[str, List[str]]) -> int:
    base_score = sum(1 for qt in q_tokens if qt in text.lower())

    # keyword_map = {
    #     "판매": ["invoice", "invoiceline", "order", "payment", "track", "album"],
    #     "앨범": ["album", "track", "invoice"],
    #     "음악": ["track", "artist", "album"],
    #     "고객": ["customer", "invoice"],
    #     "직원": ["employee", "supportrep"],
    # }
    # relation_bonus_map = {
    #     "invoice": ["invoiceline", "track"],
    #     "track": ["album", "invoiceline"],
    #     "album": ["track", "artist"],
    # }

    semantic_bonus = 0
    for qt in q_tokens:
        for k, synonyms in keyword_map.items():
            if qt == k and any(word in text.lower() for word in synonyms):
                semantic_bonus += 2

    for key, related in relation_bonus_map.items():
        if key in text.lower() and any(r in text.lower() for r in related):
            semantic_bonus += 1

    return base_score + semantic_bonus


def _table_text_blob(ts: TableSchema) -> str:
    col_part = " ".join(ts.columns)
    desc_part = " ".join([(ts.descriptions or {}).get(c, "") for c in ts.columns])
    return f"{ts.table_name} {col_part} {desc_part}".lower()

class FindTablesInput(BaseModel):
    query: str
    top_k: int = 5
    full_schema_json: List[Dict[str, Any]]

class FindTablesOutput(BaseModel):
    table_name: str
    score: float

@tool("find_relevant_tables", args_schema=FindTablesInput, return_direct=False)
def find_relevant_tables(query: str, full_schema_json: List[Dict[str, Any]], top_k: int = 5) -> List[FindTablesOutput]:
    """
    사용자 질문(query)과 DB 스키마(db_schema)를 입력받아 관련도가 높은 테이블 top_k개를 반환
    반환 형식: [{"table_name": "...", "score": 7}, ...]
    """

    q_tokens = _tokenize(query)
    scored = []

    for t in full_schema_json:
        ts = TableSchema(
            table_name=t["table_name"],
            columns=t.get("column_names", []),
            descriptions={c: d for c, d in zip(t.get("column_names", []), t.get("description", []) or [])},
            samples=t.get("sample_rows", []),
        )
        sc = _score(_table_text_blob(ts), q_tokens)
        scored.append({"table_name": ts.table_name, "score": sc})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

class GetColumnsInput(BaseModel):
    table_name: str
    full_schema_json: List[Dict[str, Any]]


@tool("get_table_columns", args_schema=GetColumnsInput, return_direct=False)
def get_table_columns(table_name: str, full_schema_json: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    단일 테이블의 컬럼, 타입, 설명을 반환.
    """
    for t in full_schema_json:
        if t["table_name"].lower() == table_name.lower():
            return {
                "table_name": t["table_name"],
                "columns": t.get("column_names", []),
                "types": t.get("column_types", []),
                "descriptions": t.get("description", []),
                "samples": t.get("sample_rows", [])
            }
    return {"table_name": table_name, "error": "not found"}

SELECTOR_SYSTEM_PROMPT = """
당신은 SQLite Text-to-SQL용 MAC-SQL Selector Agent입니다.
당신의 임무는 **사용자 질문과 전체 DB 스키마(JSON 리스트)**를 기반으로 관련된 테이블과 컬럼만 선택하는 것입니다.
입력으로 주어진 'query'에 대해 'full_schema_json', 'table_relation', 'join_hint', 'keyword_map', 'relation_bonus_map' 를 참고하여 어떤 데이터가 연결되어야 하는지 고려하세요.


당신은 다음 두 가지 도구를 사용할 수 있습니다:
1) find_relevant_tables: 관련 테이블 Top-K 후보 검색
2) get_table_columns: 특정 테이블의 컬럼 및 설명 확인

모든 결과는 아래 형식의 JSON으로만 반환해야 합니다:
{
  "selected_tables": ["TBL1", "TBL2", ...],
  "selected_columns": {
    "TBL1": ["col_a", "col_b"],
    "TBL2": ["col_x"]
  },
  "rationale": "왜 이 테이블과 컬럼을 선택했는지 한국어로 설명",
  "confidence": 0.0_to_1.0
}

규칙:
- 정확성을 우선합니다 (적지만 핵심적인 테이블만 선택)
- DB 스키마에 존재하지 않는 테이블/컬럼은 절대 생성하지 마세요
- 출력은 반드시 JSON 형식만 (백틱, 문장 불가)
- 모든 설명은 한국어로 작성하세요.
"""

selector_tools = [find_relevant_tables, get_table_columns]

selector_agent = create_react_agent(
    llm_selector,
    tools=selector_tools,
    prompt=SELECTOR_SYSTEM_PROMPT,
)




llm_decomposer = AzureChatOpenAI(
    azure_deployment=os.getenv("SKCC_AZURE_MODEL"),#"gpt-4o",
    azure_endpoint=os.getenv("SKCC_AZURE_ENDPOINT"),
    api_key=os.getenv("SKCC_AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.4
)


DECOMPOSER_SYSTEM_PROMPT = """
당신은 Text-to-SQL 시스템의 Decomposer Agent입니다.
입력된 자연어 질문을 SQL로 작성하기 위한 **논리적 단계(step)** 로 분해하세요.

입력으로 주어진 'related_tables'를 참고하여 어떤 데이터가 연결되어야 하는지 고려하세요.

출력 형식(JSON만 허용):
{
  "decomposition_steps": [
    {"step": 1, "sub_question":"첫 번째 단계 설명", "subquery": "첫 번째 단계 설명을 구현한 SQL Query"},
    {"step": 2, "sub_question":"두 번째 단계 설명", "subquery": "두 번째 단계 설명을 구현한 SQL Query"},
    ...
  ],
  "reasoning": "왜 이런 단계로 분해했는지",
  "confidence": 0.0_to_1.0
}

규칙:
- 모든 단계는 SQL 작성의 논리적 순서(집계 → 정렬 → 필터 → 출력)에 따라 작성하세요.
- 불필요한 말이나 백틱(``) 없이, JSON만 출력하세요.
- 관련 테이블 간의 조인을 명확히 고려하세요.
- 모든 설명은 한국어로 작성하세요.
"""

decomposer_agent = create_react_agent(
    llm_decomposer,
    tools=[],  
    prompt=DECOMPOSER_SYSTEM_PROMPT,
)


llm_composer = AzureChatOpenAI(
    azure_deployment=os.getenv("SKCC_AZURE_MODEL"),#"gpt-4o",
    azure_endpoint=os.getenv("SKCC_AZURE_ENDPOINT"),
    api_key=os.getenv("SKCC_AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.4
)

COMPOSER_PROMPT = """
당신은 Text-to-SQL Composer Agent입니다.
입력된 자연어 질문(query), decomposition_steps, 그리고 related_tables, schema 정보를 기반으로
SQLite 데이터베이스에서 실행 가능한 SQL 쿼리를 작성하세요.

[중요 규칙 - 반드시 지켜야 합니다]

1. 사용할 수 있는 테이블과 컬럼은 오직 입력으로 주어지는 schema에 포함된 것만 사용합니다.
   - schema에는 다음과 같은 형식의 정보가 들어 있습니다:
     [
       {
         "table_name": "Album",
         "columns": ["AlbumId", "Title", "ArtistId"],
         "types": ["INTEGER", "NVARCHAR(160)", "INTEGER"]
       },
       ...
     ]
   - schema에 존재하지 않는 테이블 이름(예: "sales", "orders", "table_name")이나
     존재하지 않는 컬럼 이름("year", "city" 등)을 절대 만들지 마세요.
   - 스키마는 Chinook 예제 DB이며, 대표적인 테이블은 Album, Artist, Track, Customer, Invoice, InvoiceLine, Employee, Genre 등입니다.

2. related_tables는 selector가 고른, 이번 질의와 특히 연관성이 높은 테이블 목록입니다.
   - 우선적으로 related_tables와 그에 해당하는 schema 정보를 사용해 SQL을 구성하세요.

3. SQLite 문법을 따르세요.
   - 키워드는 대문자(SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY, LIMIT 등)를 사용합니다.
   - 문자열 결합은 `||` 연산자를 사용합니다. (예: FirstName || ' ' || LastName)
   - 날짜 비교는 TEXT 타입(YYYY-MM-DD 등)을 사용하는 것을 기본으로 합니다.

4. 질문에 '가장 많이 판매된', '상위', 'Top', '최대', '최소' 등이 포함되면
   - 반드시 집계 함수(SUM, COUNT 등) + GROUP BY + ORDER BY + LIMIT 절을 적절히 사용하세요.

출력은 반드시 JSON 형식만으로 출력하며,
코드블록( ``` )이나 'json'이라는 문자열, 그 밖의 설명 문장은 절대 포함하지 마세요.

출력 형식(JSON만 허용):
{
  "sql": "SELECT ...",
  "rationale": "쿼리 구성 이유를 한국어로 간단히 설명",
  "confidence": 0.0~1.0
}
"""
composer_agent = create_react_agent(llm_composer, tools=[], prompt=COMPOSER_PROMPT)


llm_refiner = AzureChatOpenAI(
    azure_deployment=os.getenv("SKCC_AZURE_MODEL"),#"gpt-4o",
    azure_endpoint=os.getenv("SKCC_AZURE_ENDPOINT"),
    api_key=os.getenv("SKCC_AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.4
)

REFINER_PROMPT = """
당신은 SQL Refiner Agent입니다.
입력된 SQL Query 를 점검하고, 문법적 오류와 누락된 JOIN, GROUP BY, LIMIT 등을 찾고 이를 개선하세요.
반드시 아래 형식의 **JSON만 출력**하세요.
코드블록(```)이나 추가 문장은 절대 포함하지 마세요.

규칙 : 
- 입력된 SQL 쿼리를 점검하여 문법적 오류나 누락된 조인, GROUP BY, LIMIT 등을 개선하세요.
- JOIN 누락으로 인한 컬럼 reference 오류를 검증하세요.
- 코드블록(```)이나 추가 문장은 절대 포함하지 마세요.

출력 형식(JSON만):
{
  "refined_sql": "...",
  "changes": ["어떤 개선을 했는지 목록"],
  "confidence": 0.0~1.0
}
"""
refiner_agent = create_react_agent(llm_refiner, tools=[], prompt=REFINER_PROMPT)

llm_reuse = AzureChatOpenAI(
    azure_deployment=os.getenv("SKCC_AZURE_MODEL"),
    azure_endpoint=os.getenv("SKCC_AZURE_ENDPOINT"),
    api_key=os.getenv("SKCC_AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    temperature=0.4,
)

SQL_REUSE_SYSTEM_PROMPT = """
당신은 SQL Rewrite Agent입니다.

입력으로 JSON 형태의 하나의 메시지를 받습니다. 이 JSON에는
- previous_question: 이전 자연어 질문
- previous_sql: 이전 질문에 대해 이미 잘 동작했던 SQL
- new_question: 사용자의 새 질문 (앞선 대화의 연속일 수 있음)
이 포함되어 있습니다.

new_question은 종종 "그럼 2025년은?", "2025년 Newyork은?", "이번엔 2025년"처럼
이전 질문을 축약해서 말하는 follow-up 형태일 수 있습니다.
이 경우, previous_question과 previous_sql을 적극적으로 참고하여
누락된 정보를 보완해야 합니다.

특히 다음과 같은 경우에는 반드시 can_reuse=true 로 설정하고,
가능한 한 previous_sql을 수정해서 new_sql을 만들어야 합니다:
- new_question이 이전 질문과 거의 동일하고, 연도/날짜/기간/지역/상태 등의
  조건만 달라진 경우
- new_question에서 일부 정보가 생략되어 있지만, previous_question을 보면
  의도가 명확하게 이어지는 경우

규칙:
1. 먼저 previous_sql을 분석하여 어떤 조건(연도, 날짜, 도시, 고객, 제품 등)이 사용되는지 파악합니다.
2. new_question을 읽고, previous_question과 비교하여
   - 어떤 조건이 바뀌었는지
   - 어떤 조건은 유지해야 하는지를 결정합니다.
3. 가능하다면 previous_sql의 구조를 최대한 유지한 채 WHERE 절 등의 조건만 수정하여 new_sql을 만드세요.
4. previous_question 및 previous_sql과 전혀 다른 테이블/지표/목표를 요구할 때만
   can_reuse=false 로 판단합니다.
5. 출력은 반드시 "순수 JSON 한 덩어리"로만 출력합니다.
   코드블록( ``` ), 'json'이라는 단어, 설명 문장을 절대 추가하지 마세요.

출력 형식:
{
  "can_reuse": true_or_false,
  "new_sql": "수정된 SQL 또는 재사용이 불가능한 경우 빈 문자열",
  "changes": ["어떤 조건을 어떻게 수정했는지 한국어로 설명한 문자열 목록"],
  "confidence": 0.0~1.0
}
"""

# SQL_REUSE_SYSTEM_PROMPT = """
# 당신은 SQL Rewrite Agent입니다.
# previous_question, previous_sql, new_question을 비교하여:
# - 조건만 바뀐다면 SQL을 수정하여 new_sql을 만들고
# - 재사용이 불가하다면 can_reuse=false 로 설정합니다.

# 출력 JSON 형식:
# {
#   "can_reuse": true_or_false,
#   "new_sql": "",
#   "changes": ["..."],
#   "confidence": 0.0~1.0
# }
# """

reuse_agent = create_react_agent(
    llm_reuse,
    tools=[],
    prompt=SQL_REUSE_SYSTEM_PROMPT,
)

def save_sql_memory(vector_store, question: str, sql: str, summary: str):
    doc = Document(
        page_content=question,
        metadata={
            "sql": sql,
            "summary": summary,
            "question": question,
        },
    )
    vector_store.add_documents([doc])


def try_reuse_sql_from_memory(vector_store, new_question: str, min_confidence: float = 0.7):
    results = vector_store.similarity_search(new_question, k=1)
    if not results:
        return None

    results = results[0]
    prev_question = results.metadata.get("question")
    prev_sql = results.metadata.get("sql")

    if not prev_question or not prev_sql:
        return None

    reuse_result = reuse_agent.invoke({
        "messages": [
            HumanMessage(content=json.dumps({
                "previous_question": prev_question,
                "previous_sql": prev_sql,
                "new_question": new_question
            }, ensure_ascii=False))
        ]
    })

    raw_content = reuse_result["messages"][-1].content
    if not isinstance(raw_content, str):
        print("reuse_agent 반환 타입 이상:", type(raw_content))
        return None

    raw_content_stripped = raw_content.strip()
    try:
        data = json.loads(raw_content_stripped)
    except json.JSONDecodeError as e:
        print("reuse_agent JSON 파싱 실패:", e)
        print("raw_content:", raw_content_stripped)
        return None

    if not data.get("can_reuse"):
        return None

    if data.get("confidence", 0) < min_confidence:
        return None

    print("저장된 sql load 완료 : ", data.get("new_sql"))
    return data.get("new_sql")

# def try_reuse_sql_from_memory(vector_store, new_question: str, min_confidence: float = 0.7):


#     results = vector_store.similarity_search(new_question, k=1)
#     if not results:
#         return None

#     results = results[0]
#     prev_question = results.metadata.get("question")
#     prev_sql = results.metadata.get("sql")

#     if not prev_question or not prev_sql:
#         return None

#     reuse_result = reuse_agent.invoke({
#         "messages": [
#             HumanMessage(content=json.dumps({
#                 "previous_question": prev_question,
#                 "previous_sql": prev_sql,
#                 "new_question": new_question
#             }, ensure_ascii=False))
#         ]
#     })

#     data = json.loads(reuse_result["messages"][-1].content)

#     if not data.get("can_reuse"):
#         return None

#     if data.get("confidence", 0) < min_confidence:
#         return None
#     print("저장된 sql load 완료 : ", data.get("new_sql"))
#     return data.get("new_sql")


from langgraph.graph import MessagesState, StateGraph, START, END
class AgentState(MessagesState):
    full_schema_json:str
    table_relations:List[str]
    table_join_hint:List[str]
    keyword_map:Dict[str, List[str]]
    relation_bonus_map:Dict[str, List[str]]
    question:str
    selected_tables:Any
    dec_step:Any
    sql:str
    refined_sql:str
    from_memory: bool

def memory_retrieve_node(state: AgentState) -> AgentState:
    new_question = state["question"]
    reused_sql = try_reuse_sql_from_memory(vector_store, new_question)

    if reused_sql:
        print("#memory_retrieve_node : SQL 을 불러 왔습니다")
        return {"sql": reused_sql, "from_memory": True}
    else:
        print("#memory_retrieve_node : SQL 을 불러 오지 않았습니다.")
        return {"from_memory": False}

def selector_node(state: AgentState) -> AgentState:
    nl_question = state['question']
    table_relations = state['table_relations']
    table_join_hint = state['table_join_hint']
    keyword_map = state['keyword_map']
    relation_bonus_map = state['relation_bonus_map']
    full_schema_json = state['full_schema_json']

    selector_result = selector_agent.invoke(
        {
            "messages": [
                HumanMessage(content=f"""
                    query: {nl_question}
                    table_relation: {table_relations}
                    join_hint: {table_join_hint}
                    keyword_map: {keyword_map}
                    relation_bonus_map: {relation_bonus_map}
                    full_schema_json: {full_schema_json}
                """)
            ]
        }
    )

    raw = selector_result["messages"][-1].content.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    # JSON 파싱
    try:
        data = json.loads(raw)
    except Exception as e:
        return {
            "selected_tables": [],
            "question": nl_question
        }

    selected_tables = data.get("selected_tables", [])
    return {
        "question": nl_question,
        "selected_tables": selected_tables
    }
    
def decomposer_node(state:AgentState)->AgentState:
    nl_question = state['question']
    selected_tables = state['selected_tables']


    decomposer_result = decomposer_agent.invoke({
        "messages": [
            HumanMessage(content=f"query: {nl_question}\nrelated_tables: {selected_tables}")
        ]
    })

    dec_steps = json.loads(decomposer_result["messages"][-1].content)
    #print("decomposer : ", dec_steps)
    return {"dec_step":dec_steps}
def composer_node(state: AgentState) -> AgentState:
    nl_question = state['question']
    selected_tables = state['selected_tables']
    dec_steps = state['dec_step']
    full_schema_json = state['full_schema_json']

    schema_snippet = []
    for t in full_schema_json:
        if not selected_tables or t["table_name"] in selected_tables:
            schema_snippet.append({
                "table_name": t["table_name"],
                "columns": t.get("column_names", []),
                "types": t.get("column_types", []),
            })

    composer_input = {
        "query": nl_question,
        "related_tables": selected_tables,
        "decomposition_steps": dec_steps,
        "schema": schema_snippet,
    }

    composer_result = composer_agent.invoke({
        "messages": [
            HumanMessage(content=json.dumps(composer_input, ensure_ascii=False))
        ]
    })

    raw = composer_result["messages"][-1].content.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    data = json.loads(raw)
    sql = data["sql"]
    print("#composer_node : SQL 이 생성 되었습니다.", sql)
    return {"sql": sql}
# def composer_node(state:AgentState)->AgentState:
#     nl_question = state['question']
#     selected_tables = state['selected_tables']
#     dec_steps = state['dec_step']

#     composer_result = composer_agent.invoke({"messages": [
#         HumanMessage(content=f"query: {nl_question}\nrelated_tables: {selected_tables}\ndecomposition_steps: {dec_steps}")
#     ]})
#     #print("#2 ", composer_result)

#     sql = json.loads(composer_result["messages"][-1].content)["sql"]
#     print("#composer_node : SQL 이 생성 되었습니다.", sql)
#     return {"sql": sql}

def refiner_node(state:AgentState)->AgentState:

    nl_question = state['question']
    sql = state['sql']

    refiner_result = refiner_agent.invoke({"messages": [
        HumanMessage(content=f"sql: {sql}\nquestion: {nl_question}")
    ]})
    
    res = json.loads(refiner_result['messages'][-1].content)
    print("#refiner_node : SQL 이 조율 되었습니다.", res)
    return {"refined_sql":res}


def memory_save_node(state: AgentState) -> AgentState:
    question = state["question"]
    refined = state["refined_sql"]
    refined_sql = refined.get("refined_sql")

    selected_tables = state.get("selected_tables", [])

    if selected_tables:
        tables_str = ", ".join(selected_tables)
        summary = f"[테이블: {tables_str}] {question}"
    else:
        summary = f"[테이블 정보 없음] {question}"

    if refined_sql:
        print("#memory_save_node : SQL 이 저장되었습니다.",refined_sql, summary)
        save_sql_memory(vector_store, question, refined_sql, summary)

    return state


def make_query(agent_input:AgentState):
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("memory_retrieve_node", memory_retrieve_node)
    graph_builder.add_node("selector_node", selector_node)
    graph_builder.add_node("decomposer_node", decomposer_node)
    graph_builder.add_node("composer_node", composer_node)
    graph_builder.add_node("refiner_node", refiner_node)
    graph_builder.add_node("memory_save_node", memory_save_node)

    graph_builder.add_edge(START, "memory_retrieve_node")

    def route_memory(state: AgentState):
        if state.get("from_memory"):
            return "refiner_node"
        else:
            return "selector_node"

    graph_builder.add_conditional_edges(
        "memory_retrieve_node",
        route_memory,
        {
            "refiner_node": "refiner_node",
            "selector_node": "selector_node",
        }
    )
    graph_builder.add_edge("selector_node", "decomposer_node")
    graph_builder.add_edge("decomposer_node", "composer_node")
    graph_builder.add_edge("composer_node", "refiner_node")
    graph_builder.add_edge("refiner_node", "memory_save_node")
    graph_builder.add_edge("memory_save_node", END)

    graph = graph_builder.compile()

    #agent_input
    #input_value = {"question": nl_question}
    result = graph.invoke(agent_input)

    return result
    

    