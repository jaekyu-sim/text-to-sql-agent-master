import streamlit as st
from text2sql_agent import AgentState, make_query
import json
import pandas as pd
from exec_utils import execute_preview_sqlite

# ----------------------------------------
# 기본 설정
# ----------------------------------------
st.set_page_config(page_title="Text2SQL (SQLite · Chinook)", layout="wide")
st.title("🧠 Text2SQL for SQLite (Chinook)")

# ----------------------------------------
# 스키마 / 힌트 세팅
# ----------------------------------------
with open("Chinook_Column_Config.json", "r", encoding="utf-8") as f:
    full_schema_json = json.load(f)

table_relations = [
    "Album.ArtistId → Artist.ArtistId",
    "Track.AlbumId → Album.AlbumId",
    "Track.GenreId → Genre.GenreId",
    "InvoiceLine.TrackId → Track.TrackId",
    "InvoiceLine.InvoiceId → Invoice.InvoiceId",
    "Invoice.CustomerId → Customer.CustomerId",
]

table_join_hint = [
    '"판매/주문/결제/구매" → Invoice, InvoiceLine',
    '"앨범/음악/트랙/곡/아티스트" → Album, Track, Artist',
    '"고객/국가/도시" → Customer, Invoice',
]

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

# ----------------------------------------
# 세션 상태 초기화 (채팅 + SQL)
# ----------------------------------------
if "messages" not in st.session_state:
    # [{"role": "user" | "assistant", "content": "..."}, ...]
    st.session_state["messages"] = []

if "last_sql" not in st.session_state:
    st.session_state["last_sql"] = None

# ----------------------------------------
# 과거 채팅 히스토리 렌더링
# ----------------------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------------
# 유저 입력 (채팅 방식)
# ----------------------------------------
user_query = st.chat_input("자연어로 질문을 입력해 주세요. 예) 2024년 New York의 최대 앨범 판매량은?")

if user_query:
    # 1) 유저 메시지 히스토리에 추가
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2) Agent 입력 상태 구성
    agent_state = {
        "messages": [],  # 필요 시 LangGraph 쪽 대화 컨텍스트로 확장 가능
        "question": user_query,
        "full_schema_json": full_schema_json,
        "table_relations": table_relations,
        "table_join_hint": table_join_hint,
        "keyword_map": keyword_map,
        "relation_bonus_map": relation_bonus_map,
    }

    # 3) Text2SQL 그래프 실행
    result = make_query(agent_state)
    refined_sql = result["refined_sql"]["refined_sql"]
    st.session_state["last_sql"] = refined_sql

    # 4) 어시스턴트 응답 메시지 생성 (SQL 포함)
    assistant_text = f"다음 SQL을 생성했어요:\n```sql\n{refined_sql}\n```"
    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

    with st.chat_message("assistant"):
        st.markdown(assistant_text)

# ----------------------------------------
# 아래 영역: SQL 미리보기 / 힌트
# ----------------------------------------
st.divider()
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🧩 생성된 SQL & 미리보기 실행 (SQLite)")

    if st.session_state.get("last_sql"):
        st.code(st.session_state["last_sql"], language="sql")

        if st.button("▶️ 미리보기 실행 (SQLite)", use_container_width=True):
            rep = execute_preview_sqlite("Chinook_Sqlite.sqlite", st.session_state["last_sql"])
            if rep.ok:
                st.success(f"미리보기 성공 · rows={rep.row_count}")
                df = pd.DataFrame(rep.rows_preview)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.error(f"실행 실패: {rep.err_type} · {rep.err_message}")
    else:
        st.info("위 채팅창에 먼저 자연어 질의를 입력하면, 생성된 SQL을 여기에서 확인하고 실행할 수 있어요.")

with col2:
    st.subheader("📚 힌트")
    st.write("- SELECT 전용, 자동 LIMIT 50 적용 (구현 로직에 맞게 조정 가능)")
    st.write("- 스키마는 Chinook 기준")
    st.write("- “2024년 New York 최대 판매량” → 연도/도시 바꿔가며 후속 질의 가능")
    st.write("- 이전 질의/SQL은 벡터 DB에 저장되어, 재사용 후보로 활용됩니다.")