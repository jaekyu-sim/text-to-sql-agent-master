# server.py
from mcp.server.fastmcp import FastMCP
from text2sql_agent import make_query, AgentState
from exec_utils import try_execute_with_refine
from dotenv import load_dotenv
import json
import os
import yaml

# 새로 추가
from db_proxy import DBProxy
from schema_tools import get_schema_cache, get_relations_cache

load_dotenv()

# 기존 코드의 에이전트 초기화 등 그대로 사용

mcp = FastMCP("TextToSQL Server")

with open('Chinook_Column_Config.json', 'r', encoding='utf-8') as f:

    json_data = json.load(f)

# global full_schema_json

full_schema_json = json_data

table_relations = ["Album.ArtistId → Artist.ArtistId"
                    , "Track.AlbumId → Album.AlbumId"
                    , "Track.GenreId → Genre.GenreId"
                    , "InvoiceLine.TrackId → Track.TrackId"
                    , "InvoiceLine.InvoiceId → Invoice.InvoiceId"
                    , "Invoice.CustomerId → Customer.CustomerId"]

table_join_hint = [""" "판매", "주문", "결제", "구매" 관련 질문 → Invoice 또는 InvoiceLine 관련 """
                    , """ "판매", "주문", "결제", "구매" 관련 질문 → Invoice 또는 InvoiceLine 관련 """
                    , """ "앨범", "음악", "트랙", "곡", "아티스트" 관련 질문 → Album, Track, Artist 관련 """
                    , """ "고객", "국가", "도시" 관련 → Customer, Invoice """]

keyword_map = {
    "판매": ["invoice", "invoiceline", "order", "payment", "track", "album"],
    "앨범": ["album", "track", "invoice"],
    "음악": ["track", "artist", "album"],
    "고객": ["customer", "invoice"],
    "직원": ["employee", "supportrep"],
}

relation_bonus_map = {
    "invoice": ["invoiceline", "track"],
    "track": ["album", "invoiceline"],
    "album": ["track", "artist"],
}

with open("config/security.yaml", "r", encoding="utf-8") as f:
    SECURITY = yaml.safe_load(f)

DB_PATH = "Chinook_Sqlite.sqlite"
db_proxy = DBProxy(dsn=DB_PATH, security_conf=SECURITY, dialect="sqlite")

@mcp.tool()
def get_schema() -> str:
    """
    전체 컬럼 스키마를 JSON 문자열로 반환
    """
    #return json.dumps(full_schema_json, ensure_ascii=False)
    return json.dumps(get_schema_cache(), ensure_ascii=False) # cache 적용 버전

@mcp.tool()
def get_relations() -> str:
    """
    테이블 간 릴레이션을 JSON 문자열로 반환
    """
    return json.dumps(table_relations, ensure_ascii=False)


@mcp.tool()
def generate_sql(query: str) -> str:
    """자연어 질의를 입력받아 대응되는 SQL 쿼리를 반환합니다."""
    # agent_state = AgentState(
    #     question=query,
    #     full_schema_json=full_schema_json,
    #     table_relations=table_relations,
    #     table_join_hint=table_join_hint,
    #     keyword_map=keyword_map,
    #     relation_bonus_map=relation_bonus_map,
    # )

    agent_state = AgentState(
        question=query,
        full_schema_json=get_schema_cache(),
        table_relations=get_relations_cache(),
        table_join_hint=table_join_hint,
        keyword_map=keyword_map,
        relation_bonus_map=relation_bonus_map,
    )

    result_state = make_query(agent_state)
    final_sql = result_state["refined_sql"]["refined_sql"]



    result_state = make_query(agent_state)
    refined_sql = result_state["refined_sql"]["refined_sql"]
    DB_PATH = "Chinook_Sqlite.sqlite"

    print("refined sql : ", refined_sql)

    # (옵션) 생성 직후 미리보기 실행 + 자동 보정 루프
    final_sql, report = try_execute_with_refine(
        db_path=DB_PATH,
        question=query,
        db_desc={"dialect": "sqlite"},
        sql=refined_sql,
        max_try=3,
        preview_limit=50
    )

    print("final sql : ", final_sql)

    # 운영 정책에 따라 반환값 선택
    # 1) SQL만 반환
    return final_sql
    #return final_sql
@mcp.tool()
def run_sql_proxy(sql: str, params_json: str = "{}") -> str:
    """
    SQL 실행 전용 툴 (가드레일+타임아웃+LIMIT 포함)
    - params_json 예: {"CustomerId": 5, "Country": "USA"}
    반환: {"columns":[...], "rows":[[...],...], "row_count": n}
    """
    params = json.loads(params_json or "{}")
    cols, rows = db_proxy.execute_safe(sql, params)
    return json.dumps({"columns": cols, "rows": rows, "row_count": len(rows)}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")